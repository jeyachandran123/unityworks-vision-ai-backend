"""Recorders — the DVRs and NVRs cameras plug into.

A recorder owns everything about *reaching* a camera that is the same for every
channel on it: address, port, account, password and the URL convention of its
vendor. Cameras reference it and own only what differs per camera.

### The password only ever goes in

`create` and `set_credential` accept a plaintext password and seal it before it
reaches the row; nothing in this module, and nothing that calls `to_wire`, can
read one back. The wire says whether a credential is configured and how it is
stored, and nothing else — not the value, not the ciphertext, not a masked
rendering of the real thing. Changing a password means typing a new one.

### Tenant scoping

Every entry point takes `organization_id` and constructs its query already
narrowed. A recorder in another organization is `NotFoundError`, never a
refusal, because a refusal would confirm it exists.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Camera, Recorder
from app.domain.recorder_brands import (
    CUSTOM,
    Brand,
    BrandError,
    brand_for,
    custom_brand,
    load_brands,
)
from app.domain.recorder_secrets import (
    MissingMasterKeyError,
    SealedSecret,
    SecretSealError,
    master_key,
    seal,
)
from app.errors import ConflictError, DependencyUnavailableError, NotFoundError, ValidationError
from app.vision.recorder_credentials import RECORDER_SCHEME, recorder_reference

if TYPE_CHECKING:  # pragma: no cover
    from app.configuration.settings import Settings

#: A hostname or an IPv4 address, and nothing else.
#:
#: The host is interpolated into `rtsp://user:pass@HOST:port/path`. A value
#: carrying `/`, `@`, `?` or whitespace would rewrite that URL — `@` alone would
#: move the credentials onto a host of the caller's choosing — so the charset is
#: closed rather than escaped.
_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.\-]{0,252}[A-Za-z0-9])?$")

#: Custom stream numbers. Every vendor here uses a single digit; two leaves room.
_STREAM_NUMBER_RANGE = range(0, 100)

_MAX_PASSWORD = 256


class RecorderInUseError(ConflictError):
    """The recorder still has cameras, so it can be deactivated but not deleted."""


# ── the service ──────────────────────────────────────────────────────────────


class RecorderService:
    """Recorder configuration. Tenant-scoped at every entry point."""

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(
        self, *, organization_id: str, limit: int = 25, offset: int = 0
    ) -> tuple[list[Recorder], int]:
        total = (
            await self._session.execute(
                select(func.count())
                .select_from(Recorder)
                .where(Recorder.organization_id == organization_id)
            )
        ).scalar_one()
        rows = (
            (
                await self._session.execute(
                    select(Recorder)
                    .where(Recorder.organization_id == organization_id)
                    # Active first, then by name: what somebody is looking for is
                    # almost always a box that is running.
                    .order_by(Recorder.is_active.desc(), Recorder.name)
                    .limit(max(1, min(limit, 200)))
                    .offset(max(0, offset))
                )
            )
            .scalars()
            .all()
        )
        return list(rows), int(total)

    async def camera_counts(self, *, organization_id: str) -> dict[str, tuple[int, int]]:
        """`{recorder_id: (cameras, cameras switched on)}` in one query."""
        rows = (
            await self._session.execute(
                select(
                    Camera.recorder_id,
                    func.count(),
                    func.sum(case((Camera.enabled.is_(True), 1), else_=0)),
                )
                .where(Camera.organization_id == organization_id)
                .group_by(Camera.recorder_id)
            )
        ).all()
        return {recorder_id: (int(total), int(on or 0)) for recorder_id, total, on in rows}

    async def get(self, *, organization_id: str, recorder_id: str) -> Recorder:
        recorder = (
            await self._session.execute(
                select(Recorder).where(
                    Recorder.organization_id == organization_id,
                    Recorder.id == str(recorder_id or ""),
                )
            )
        ).scalar_one_or_none()
        if recorder is None:
            raise NotFoundError("no such recorder")
        return recorder

    async def cameras(self, *, organization_id: str, recorder_id: str) -> list[Camera]:
        return list(
            (
                await self._session.execute(
                    select(Camera)
                    .where(
                        Camera.organization_id == organization_id,
                        Camera.recorder_id == recorder_id,
                    )
                    .order_by(Camera.channel, Camera.camera_key)
                )
            )
            .scalars()
            .all()
        )

    async def create(
        self,
        *,
        organization_id: str,
        name: str,
        host: str,
        rtsp_port: int,
        username: str,
        brand: str,
        settings: Settings,
        password: str | None = None,
        credential_ref: str | None = None,
        path_template: str = "",
        stream_main: int | None = None,
        stream_sub: int | None = None,
    ) -> Recorder:
        """Add a recorder. Its password is sealed before the row exists.

        `credential_ref` is for callers inside the application that keep a
        password in the environment (`env:`/`file:`). Nothing reachable from an
        HTTP request passes one: a client that could name a reference could make
        the server read a secret it was never given.
        """
        clean = clean_connection(
            name=name,
            host=host,
            rtsp_port=rtsp_port,
            username=username,
            brand=brand,
            path_template=path_template,
            stream_main=stream_main,
            stream_sub=stream_sub,
        )
        await self._name_is_free(organization_id, clean["name"])

        if password is None and not credential_ref:
            raise ValidationError(
                "a password is required: the recorder cannot be signed in to without one"
            )
        if credential_ref and not credential_ref.startswith(("env:", "file:")):
            raise ValidationError("a credential reference must be env: or file:")

        # Sealed before the row exists, so a server that cannot seal - no master
        # key configured - refuses cleanly and leaves nothing half-written.
        sealed = _seal(password, settings) if password is not None else None

        # The id is assigned here rather than at flush because the credential
        # reference names it.
        recorder = Recorder(id=uuid.uuid4().hex, organization_id=organization_id, **clean)
        if sealed is not None:
            _apply_sealed(recorder, sealed)
        else:
            recorder.credential_ref = str(credential_ref)
        self._session.add(recorder)
        await self._session.flush()
        return recorder

    async def update(
        self, *, organization_id: str, recorder_id: str, **changes: Any
    ) -> tuple[Recorder, list[str]]:
        """Change how the recorder is reached. Returns the fields that changed.

        Never the password: that is `set_credential`, a separate act with its
        own audit row, so "who changed the password" is always answerable.
        """
        recorder = await self.get(organization_id=organization_id, recorder_id=recorder_id)
        editable = (
            "name",
            "host",
            "rtsp_port",
            "username",
            "brand",
            "path_template",
            "stream_main",
            "stream_sub",
        )
        proposed = {field: getattr(recorder, field) for field in editable}
        for field in editable:
            if field in changes and changes[field] is not None:
                proposed[field] = changes[field]
        # Leaving `custom` clears what only `custom` uses, so a stale template
        # cannot linger behind a brand that ignores it.
        if proposed["brand"] != CUSTOM:
            proposed.update(path_template="", stream_main=None, stream_sub=None)

        clean = clean_connection(**proposed)
        if clean["name"] != recorder.name:
            await self._name_is_free(organization_id, clean["name"])

        changed = [field for field in editable if getattr(recorder, field) != clean[field]]
        for field in changed:
            setattr(recorder, field, clean[field])
        if changed:
            recorder.updated_at = datetime.now(UTC)
        return recorder, changed

    async def set_credential(
        self,
        *,
        organization_id: str,
        recorder_id: str,
        password: str,
        settings: Settings,
    ) -> Recorder:
        """Replace the password. The previous one is overwritten, never read."""
        recorder = await self.get(organization_id=organization_id, recorder_id=recorder_id)
        _apply_sealed(recorder, _seal(password, settings))
        recorder.updated_at = datetime.now(UTC)
        return recorder

    async def set_active(self, *, organization_id: str, recorder_id: str, active: bool) -> Recorder:
        recorder = await self.get(organization_id=organization_id, recorder_id=recorder_id)
        recorder.is_active = bool(active)
        recorder.updated_at = datetime.now(UTC)
        return recorder

    async def delete(self, *, organization_id: str, recorder_id: str) -> None:
        """Delete a recorder that has never held a camera. Refused otherwise.

        A recorder with cameras is part of the record of how they were watched.
        Deactivating it stops them and keeps that record; deleting would need
        every camera removed first, which is a different and heavier decision
        taken camera by camera.
        """
        recorder = await self.get(organization_id=organization_id, recorder_id=recorder_id)
        count = (
            await self._session.execute(
                select(func.count())
                .select_from(Camera)
                .where(Camera.organization_id == organization_id, Camera.recorder_id == recorder.id)
            )
        ).scalar_one()
        if count:
            noun = "camera" if count == 1 else "cameras"
            raise RecorderInUseError(
                f"'{recorder.name}' still has {count} {noun}. Deactivate it instead, or "
                f"move or remove its cameras first.",
                details={"camera_count": int(count)},
            )
        await self._session.delete(recorder)

    async def _name_is_free(self, organization_id: str, name: str) -> None:
        taken = (
            await self._session.execute(
                select(Recorder.id).where(
                    Recorder.organization_id == organization_id,
                    func.lower(Recorder.name) == name.lower(),
                )
            )
        ).scalar_one_or_none()
        if taken is not None:
            raise ConflictError(f"a recorder called '{name}' already exists")


# ── brands ───────────────────────────────────────────────────────────────────


def brand_of(recorder: Recorder) -> Brand:
    """How this recorder builds its stream URLs."""
    if recorder.brand == CUSTOM:
        return custom_brand(
            path_template=recorder.path_template,
            main=int(recorder.stream_main if recorder.stream_main is not None else 0),
            sub=int(recorder.stream_sub if recorder.stream_sub is not None else 1),
            default_port=recorder.rtsp_port or 554,
        )
    return brand_for(recorder.brand)


def brand_label(recorder: Recorder) -> str:
    if recorder.brand == CUSTOM:
        return "Other (custom)"
    try:
        return brand_for(recorder.brand).label
    except BrandError:
        return recorder.brand


def brands_to_wire() -> list[dict[str, Any]]:
    """What the brand dropdown needs: a key, a label and a sensible port.

    No templates. A person choosing a brand is answering "what does it say on
    the box", and the path it implies is the server's business.
    """
    return [
        {"key": brand.key, "label": brand.label, "default_port": brand.default_port}
        for brand in sorted(load_brands().values(), key=lambda b: b.label.lower())
    ]


# ── the wire ─────────────────────────────────────────────────────────────────


def to_wire(recorder: Recorder, *, camera_count: int = 0, cameras_on: int = 0) -> dict[str, Any]:
    """A recorder for the API. Carries nothing that could recover the password."""
    custom = recorder.brand == CUSTOM
    return {
        "id": recorder.id,
        "name": recorder.name,
        "host": recorder.host,
        "rtsp_port": recorder.rtsp_port,
        "username": recorder.username,
        "brand": recorder.brand,
        "brand_label": brand_label(recorder),
        # Only for `custom`, where the person typed it themselves. A known brand's
        # template is the server's concern and is not shown.
        "path_template": recorder.path_template if custom else None,
        "stream_main": recorder.stream_main if custom else None,
        "stream_sub": recorder.stream_sub if custom else None,
        "is_active": recorder.is_active,
        "credential_configured": credential_configured(recorder),
        "credential_scheme": credential_scheme(recorder.credential_ref),
        "camera_count": camera_count,
        "cameras_on": cameras_on,
        "created_at": _iso(recorder.created_at),
        "updated_at": _iso(recorder.updated_at),
    }


def credential_configured(recorder: Recorder) -> bool:
    if (recorder.credential_ref or "").startswith(RECORDER_SCHEME):
        return recorder.secret_ciphertext is not None and recorder.secret_nonce is not None
    return bool(recorder.credential_ref)


def credential_scheme(credential_ref: str) -> str:
    """`recorder`, `env` or `file` — how the password is kept, never where or what."""
    scheme, separator, _ = (credential_ref or "").partition(":")
    return scheme if separator else ""


# ── sealing ──────────────────────────────────────────────────────────────────


def sealed_of(recorder: Recorder) -> SealedSecret | None:
    """The sealed password on this row, when it keeps one."""
    if not (recorder.credential_ref or "").startswith(RECORDER_SCHEME):
        return None
    if recorder.secret_ciphertext is None or recorder.secret_nonce is None:
        return None
    return SealedSecret(
        ciphertext=bytes(recorder.secret_ciphertext),
        nonce=bytes(recorder.secret_nonce),
        key_id=recorder.secret_key_id or "",
    )


async def load_sealed(session: AsyncSession) -> dict[str, SealedSecret]:
    """Every sealed recorder password in the database, by recorder id.

    Across organizations on purpose: the credential store serves the whole
    process, and each camera still names exactly one recorder of its own.
    """
    rows = (
        (
            await session.execute(
                select(Recorder).where(Recorder.credential_ref.like(f"{RECORDER_SCHEME}%"))
            )
        )
        .scalars()
        .all()
    )
    return {row.id: sealed for row in rows if (sealed := sealed_of(row)) is not None}


def sync_credential(store: Any, recorder: Recorder) -> None:
    """Bring the process-wide credential store up to date with this recorder.

    Called whenever a recorder's password changes or one of its cameras is
    about to be dialled, so a camera started from the product never uses a
    stale password and never waits for a restart to learn a new one.
    """
    if store is None:
        return
    store.put(recorder.id, sealed_of(recorder))


def _seal(password: str, settings: Settings) -> SealedSecret:
    if not isinstance(password, str) or not password:
        raise ValidationError("a password is required")
    if len(password) > _MAX_PASSWORD:
        raise ValidationError(f"a password may be at most {_MAX_PASSWORD} characters")
    try:
        key = master_key(settings)
    except MissingMasterKeyError as exc:
        # A deployment step was missed, not a user mistake. Said so, in words
        # an administrator can act on, rather than as a generic server error.
        raise DependencyUnavailableError(
            "Recorder passwords cannot be saved on this server yet: its encryption "
            "key (RECORDER_SECRET_KEY) is not configured. Ask whoever runs the "
            "server to set it.",
            details={"setting": "RECORDER_SECRET_KEY"},
        ) from exc
    except SecretSealError as exc:
        raise DependencyUnavailableError(
            "Recorder passwords cannot be saved on this server: its encryption key "
            "(RECORDER_SECRET_KEY) is not valid.",
            details={"setting": "RECORDER_SECRET_KEY"},
        ) from exc
    return seal(password, key=key, key_id=settings.recorder_secret_key_id or "k1")


def _apply_sealed(recorder: Recorder, sealed: SealedSecret) -> None:
    recorder.secret_ciphertext = sealed.ciphertext
    recorder.secret_nonce = sealed.nonce
    recorder.secret_key_id = sealed.key_id
    recorder.credential_ref = recorder_reference(recorder.id)


# ── validation ───────────────────────────────────────────────────────────────


def clean_connection(
    *,
    name: Any,
    host: Any,
    rtsp_port: Any,
    username: Any,
    brand: Any,
    path_template: Any = "",
    stream_main: Any = None,
    stream_sub: Any = None,
) -> dict[str, Any]:
    """Validate everything about reaching a recorder, and return it normalised.

    Each message says what to do, because this text is shown to the person who
    typed the value.
    """
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValidationError("give the recorder a name, such as “Kitchen DVR”")
    if len(clean_name) > 255:
        raise ValidationError("a recorder name may be at most 255 characters")

    clean_host = str(host or "").strip()
    if not clean_host:
        raise ValidationError("enter the recorder's IP address or hostname")
    if not _HOST.match(clean_host):
        raise ValidationError(
            "enter just the address, such as 192.168.1.20 or dvr.example.com — "
            "without rtsp://, a port, or a path"
        )

    try:
        port = int(rtsp_port)
    except (TypeError, ValueError) as exc:
        raise ValidationError("the port must be a number, usually 554") from exc
    if not 1 <= port <= 65535:
        raise ValidationError("the port must be between 1 and 65535, usually 554")

    clean_username = str(username or "").strip()
    if len(clean_username) > 128:
        raise ValidationError("a username may be at most 128 characters")

    clean_brand = str(brand or "").strip().lower()
    template = ""
    main: int | None = None
    sub: int | None = None
    if clean_brand == CUSTOM:
        template = str(path_template or "").strip()
        if not template.startswith("/") or any(ch.isspace() for ch in template):
            raise ValidationError(
                "a custom stream path starts with / and contains no spaces, such as "
                "/live/ch{channel}/{subtype}"
            )
        main, sub = _stream_number(stream_main, "main"), _stream_number(stream_sub, "sub")
        try:
            custom_brand(path_template=template, main=main, sub=sub)
        except BrandError as exc:
            raise ValidationError(_friendly_brand_error(exc)) from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise ValidationError(
                "the custom stream path may only use {channel} and {subtype}"
            ) from exc
        _check_template_renders(template)
    else:
        try:
            brand_for(clean_brand)
        except BrandError as exc:
            raise ValidationError("choose the recorder's brand from the list") from exc

    return {
        "name": clean_name,
        "host": clean_host,
        "rtsp_port": port,
        "username": clean_username,
        "brand": clean_brand,
        "path_template": template,
        "stream_main": main,
        "stream_sub": sub,
    }


def _stream_number(value: Any, which: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"enter the number this device uses for its {which} stream") from exc
    if number not in _STREAM_NUMBER_RANGE:
        raise ValidationError(f"the {which} stream number must be between 0 and 99")
    return number


def _check_template_renders(template: str) -> None:
    """A template with a placeholder we do not supply fails at connect time,
    with an error about a camera rather than about the template. Caught here."""
    try:
        template.format(channel=1, subtype=0)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValidationError(
            "the custom stream path may only use {channel} and {subtype}"
        ) from exc


def _friendly_brand_error(exc: BrandError) -> str:
    text = str(exc)
    if "{channel}" in text:
        return "the custom stream path must include {channel}, or every camera would show the same picture"
    if "both main and sub" in text:
        return "the main and sub stream numbers must be different"
    return "the custom stream settings are not valid"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


__all__ = [
    "RecorderInUseError",
    "clean_connection",
    "RecorderService",
    "brand_label",
    "brand_of",
    "brands_to_wire",
    "credential_configured",
    "credential_scheme",
    "load_sealed",
    "sealed_of",
    "sync_credential",
    "to_wire",
]
