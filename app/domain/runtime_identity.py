"""The globally unique name a camera answers to inside the runtime.

`Camera.camera_key` is unique **per organization** — that is what
`uq_camera_key` says, and it is the right constraint for a tenant-scoped
table. It is the wrong thing to key a process-wide registry on, and this
module exists because the runtime was doing exactly that.

### The failure this closes

`CameraWall._streams` was a `dict[str, CameraStream]` keyed on the bare
`camera_key`, and `FileObservationLog` names its partition file after the
`CameraId` it is handed. `cam-01` is the single most likely camera key any
customer will choose, so the first two organizations to run cameras would
have collided on it — and a collision here is not a crash. It is one
organization's operator receiving another organization's frames from
`wall.get("cam-01")`, and both organizations' observations appended to one
partition file. Neither failure announces itself.

The stream ticket did not save us either. It is HMAC-signed over
`(tenant, camera, subject)` and cannot be forged, so organization B's
operator could only ever mint a ticket for organization B's `cam-01` — and
then `wall.get("cam-01")` would hand them organization A's stream. The
signature was sound and the lookup it authorized was not.

### The identity

    organization_id ":" camera_key

Ordered that way deliberately: the tenant is the outer scope, so the string
sorts and reads the way the hierarchy does. Composed rather than a fresh
opaque id because it stays legible in a log line and in a partition filename,
which is most of what these identifiers are for in practice.

### Why the parse is unambiguous, structurally

Two charsets are enforced here and at every write boundary:

* an organization id is `[A-Za-z0-9-]+` — no `:` and, importantly, **no `_`**
* a camera key is `[A-Za-z0-9][A-Za-z0-9_-]*`

The organization half therefore cannot contain the separator, so
`split(":", 1)` always recovers exactly the pair that was composed, whatever
the camera key contains.

The `_` exclusion looks arbitrary and is not. `FileObservationLog._path`
rewrites every character outside `[A-Za-z0-9_-]` to `_`, so on disk the
identity becomes `<org>_<key>`. If an organization id could contain `_`,
`a_` + `b` and `a` + `_b` would both land in `a___b.jsonl` — the same
collision one layer down, reintroduced by the filesystem rather than by the
registry. Forbidding `_` in the organization half makes the first `_` in a
partition filename the separator, and the round trip provable.
"""

from __future__ import annotations

import re

from app.errors import ValidationError

#: Separates the tenant from the camera. Legal in neither half.
SEPARATOR = ":"

#: See the module docstring — the `_` exclusion is load-bearing on disk.
ORGANIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
CAMERA_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_organization_id(organization_id: str) -> str:
    """The tenant half. Rejected here rather than discovered as a collision."""
    value = (organization_id or "").strip()
    if not ORGANIZATION_ID.match(value):
        raise ValidationError(
            "an organization id must be letters, digits and hyphens, starting "
            "with a letter or digit; '_' and ':' are reserved because runtime "
            "camera identity is composed from this value",
            details={"organization_id": value},
        )
    return value


def validate_camera_key(camera_key: str) -> str:
    """The camera half."""
    value = (camera_key or "").strip()
    if not value:
        raise ValidationError("a camera must have a key; identity is never inferred")
    if not CAMERA_KEY.match(value):
        raise ValidationError(
            "a camera key must be letters, digits, hyphens and underscores, "
            "starting with a letter or digit; ':' is reserved as the separator "
            "between the organization and the camera in runtime identity",
            details={"camera_key": value},
        )
    return value


def runtime_camera_id(organization_id: str, camera_key: str) -> str:
    """The name this camera answers to in every process-wide registry.

    Every runtime lookup goes through this — the wall registry, the stream
    ticket, the observation partition, the RTSP session. A caller that reaches
    a registry with a bare `camera_key` has the bug this module exists to make
    unwriteable.
    """
    return (
        f"{validate_organization_id(organization_id)}"
        f"{SEPARATOR}"
        f"{validate_camera_key(camera_key)}"
    )


def for_camera(camera) -> str:
    """The runtime id of a `Camera` row, without restating both halves."""
    return runtime_camera_id(camera.organization_id, camera.camera_key)


def split_runtime_camera_id(runtime_id: str) -> tuple[str, str]:
    """`(organization_id, camera_key)`. The exact inverse of composition.

    Total, because the organization half cannot contain the separator — see
    the module docstring. A value with no separator at all is a bare camera
    key from before this identity existed, and is refused rather than
    guessed at: guessing its tenant is the failure this module closes.
    """
    organization_id, separator, camera_key = (runtime_id or "").partition(SEPARATOR)
    if not separator:
        raise ValidationError(
            "this is a bare camera key, not a runtime camera id; a camera key "
            "is unique only within its organization, so it cannot name a "
            "camera on its own",
            details={"value": runtime_id},
        )
    return organization_id, camera_key


__all__ = [
    "CAMERA_KEY",
    "ORGANIZATION_ID",
    "SEPARATOR",
    "for_camera",
    "runtime_camera_id",
    "split_runtime_camera_id",
    "validate_camera_key",
    "validate_organization_id",
]
