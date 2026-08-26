"""The pass that turns live observations into compliance findings and incidents.

### This composes; it does not implement

A full compliance engine already exists at the repository's top level in
`compliance/` — `RuleSet`, `ComplianceEvaluator`, `ObservationReader`, the
four-state semantics and their own test suite. Nothing had ever *called* it.
That is the same shape as every other gap this project has found: the capability
was built, and the composition root never wired it.

So this file holds no rule logic, no operator table and no verdict semantics.
It reads Vision State through `ObservationReader`, hands the views to
`ComplianceEvaluator`, and maps the findings it gets back onto the
application's Incident domain — which is the one part that genuinely did not
exist anywhere.

### Where the layer boundary sits

`compliance/` knows about observations and rules and nothing about users,
tenants, databases or incidents. `app/domain/incidents.py` knows about incidents
and nothing about rules. This module is the only place that knows both, which
is what keeps either of them replaceable.

### Why a timer rather than a subscription

Vision State publishes deltas, and subscribing would be tidier. It would also
put a database write on the platform's own publish path, where a slow commit
becomes backpressure on synthesis — and this project has already been bitten
twice by CPU-bound work landing on a hot path (Phases 6B.2 and 6B.3). A timer
reading a snapshot cannot do that: a slow pass simply starts the next one later.
The cost is latency bounded by the interval, and the deduplication in
`IncidentService.open` makes re-reading the same violation harmless.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger


@dataclass(slots=True)
class CompliancePass:
    """What one evaluation pass did. Counters only — no imagery, no identity."""

    subjects: int = 0
    findings: int = 0
    compliant: int = 0
    violations: int = 0
    unknown: int = 0
    incidents_opened: int = 0
    incidents_updated: int = 0
    incidents_resolved: int = 0
    errors: int = 0
    evidence_captured: int = 0
    notifications_sent: int = 0
    notifications_failed: int = 0
    capability_gaps: tuple[str, ...] = ()
    by_rule: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, finding: Any) -> None:
        state = finding.state.value
        self.findings += 1
        bucket = self.by_rule.setdefault(
            finding.rule_id, {"compliant": 0, "violation": 0, "unknown": 0}
        )
        bucket[state] = bucket.get(state, 0) + 1

        # Named explicitly rather than `setattr(self, state, ...)`. The counter
        # for `violation` is `violations`, so the dynamic version resolved
        # correctly for `compliant` and `unknown` and raised `AttributeError`
        # on exactly the one state that matters — and only once a real
        # violation appeared, which is the worst possible time to find out.
        if state == "violation":
            self.violations += 1
        elif state == "compliant":
            self.compliant += 1
        else:
            self.unknown += 1

    def to_wire(self) -> dict[str, Any]:
        return {
            "subjects": self.subjects,
            "findings": self.findings,
            "compliant": self.compliant,
            "violations": self.violations,
            "unknown": self.unknown,
            "incidents_opened": self.incidents_opened,
            "incidents_updated": self.incidents_updated,
            "incidents_resolved": self.incidents_resolved,
            "errors": self.errors,
            "evidence_captured": self.evidence_captured,
            "notifications_sent": self.notifications_sent,
            "notifications_failed": self.notifications_failed,
            "capability_gaps": list(self.capability_gaps),
            "by_rule": self.by_rule,
        }


#: Severities that may raise an incident. `informational` rules exist so a
#: finding can accrue with full provenance while its accuracy is still
#: unmeasured — the shipped face-covering rule says exactly that about itself —
#: and must not page anyone on unscored evidence.
RAISES_INCIDENTS = frozenset({"low", "medium", "high", "critical"})


def _observed_at(finding: Any) -> datetime:
    """The finding's own evaluation time, not now.

    An incident is about when the violation was seen; stamping it with
    processing time makes a backlog look like a burst of fresh violations.
    """
    ns = getattr(getattr(finding, "evaluated_at", None), "ns", None)
    if isinstance(ns, int) and ns > 0:
        with contextlib.suppress(OverflowError, OSError, ValueError):
            return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)
    return datetime.now(UTC)


class ComplianceDriver:
    """Evaluates confirmed objects on a timer and moves the incident queue."""

    __slots__ = (
        "_database", "_evaluator", "_interval_s", "_last", "_notifier",
        "_reader", "_settings", "_task", "_vision", "_wall",
    )

    def __init__(
        self, *, settings: Any, vision: Any, database: Any, rules: Any,
        interval_s: float = 5.0, wall: Any = None, notifier: Any = None,
    ) -> None:
        from compliance import ComplianceEvaluator

        self._settings = settings
        self._vision = vision
        self._database = database
        self._evaluator = ComplianceEvaluator(rules)
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._last = CompliancePass()
        self._reader: Any = None
        self._wall = wall
        self._notifier = notifier

    @property
    def evaluator(self) -> Any:
        return self._evaluator

    @property
    def last_pass(self) -> CompliancePass:
        return self._last

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="compliance-driver")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval_s)
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad pass is not a dead driver
                logger.warning("compliance pass failed: {}: {}", type(exc).__name__, exc)

    # -- reading ------------------------------------------------------------

    def _observation_reader(self) -> Any:
        exposure = getattr(getattr(self._vision, "composition", None), "exposure", None)
        if exposure is None:
            return None
        if self._reader is None:
            from compliance import ObservationReader
            from vision_os.core.model.api import Principal
            from vision_os.core.model.ids import TenantId

            self._reader = ObservationReader(
                exposure.api,
                principal=Principal(
                    subject="compliance",
                    tenant_id=TenantId(self._settings.default_tenant_id),
                ),
            )
        return self._reader

    async def cameras(self) -> dict[str, str | None]:
        """`camera_key → restaurant_id` for this tenant's enabled cameras.

        From the database, because the incident needs the restaurant and the
        camera row is the only place that mapping is durable.
        """
        from sqlalchemy import select

        from app.domain.models import Camera

        async with self._database.session_scope() as session:
            rows = await session.execute(
                select(Camera.camera_key, Camera.restaurant_id).where(
                    Camera.organization_id == self._settings.default_tenant_id,
                    Camera.enabled.is_(True),
                )
            )
            return dict(rows.all())

    def snapshot(self, camera_keys: tuple[str, ...]) -> Any:
        """Confirmed objects the platform is currently willing to vouch for.

        `StateFilter` is left at its default, which excludes `PROVISIONAL`
        objects. Phase 6A.4 rejected widening it to populate a screen; widening
        it to raise an incident against an unconfirmed object would be worse.
        """
        reader = self._observation_reader()
        if reader is None or not camera_keys:
            return None

        from vision_os.core.model.api import Scope
        from vision_os.core.model.ids import CameraId, TenantId

        return reader.read(
            Scope(
                tenant_id=TenantId(self._settings.default_tenant_id),
                camera_ids=tuple(CameraId(c) for c in camera_keys),
            )
        )

    def evaluate(self, snapshot: Any) -> tuple[CompliancePass, tuple[Any, ...]]:
        """Findings for a snapshot. **Pure** — touches no database."""
        from vision_os.core.model.timebase import Instant

        run = CompliancePass()
        if snapshot is None or not snapshot.objects:
            return run, ()

        gaps = tuple(snapshot.capability_gaps)
        run.capability_gaps = gaps
        run.subjects = len(snapshot.objects)

        import time

        findings = self._evaluator.evaluate(
            snapshot.objects,
            now=Instant(time.time_ns()),
            coverage=snapshot.coverage,
            capability_gaps=gaps,
        )
        for finding in findings:
            run.record(finding)
        return run, findings

    # -- evidence and notification ------------------------------------------

    #: How far before a finding's observation instant a retained frame may sit
    #: and still be the frame that decision was made on. One analysis interval
    #: at the measured live rate is ~1.7 s; this allows for a slow pass without
    #: reaching back to an unrelated sighting. A frame *after* the observation
    #: is never accepted at any tolerance — that is the defect itself.
    DECISION_FRAME_TOLERANCE_NS = 30_000_000_000

    def _decision_frame(self, *, camera_key: str, finding: Any):
        """The retained frame this finding's subject was actually seen in.

        Keyed by object and observation time rather than by frame reference,
        because that is what a finding carries: the attribute's `observed_at`
        and the object it belongs to. Returns None when nothing suitable was
        retained, and the caller then falls back *and says so*.
        """
        try:
            from app.vision.decision_frames import DECISION_FRAMES

            object_id = str(finding.subject.object_id)
            observed_at = _observed_at_ns(finding)
            if observed_at:
                # Strictly at or before the observation. No fallback to "the
                # latest frame this object appeared in": that is the *later
                # room state* by another name, and the first version of this
                # method did exactly that — a real cam-13 incident came back
                # with a frame 25.8 s after its own observation, still labelled
                # a decision frame. If nothing suitable was retained, the
                # caller stores a context frame and says so.
                return DECISION_FRAMES.nearest_before(
                    camera_key, object_id, observed_at,
                    tolerance_ns=self.DECISION_FRAME_TOLERANCE_NS,
                )
            # No observation instant at all. The newest frame this object was
            # genuinely seen in is the best available claim, and it is still a
            # frame containing the subject rather than a photograph of the room.
            return DECISION_FRAMES.latest_for_object(camera_key, object_id)
        except Exception as exc:  # noqa: BLE001 - evidence is not the incident
            logger.debug(
                "decision frame lookup failed: {}: {}", type(exc).__name__, exc
            )
            return None

    async def _capture_evidence(
        self, session: Any, *, camera_key: str, finding: Any
    ) -> str:
        """Store **the frame the decision was made on** as durable evidence.

        **Off unless the deployment turns it on.** Storing images of
        identifiable people is a deployment decision, and `EVIDENCE_CAPTURE` is
        separate from `ALLOW_EVIDENCE`: writing a durable record and permitting
        it to be served are different authorisations, and enabling one must
        never silently enable the other.

        ### Why this no longer photographs the room

        This used to take the camera wall's *current* JPEG. A compliance pass
        runs on a timer, so the picture was routinely of a scene the subject had
        already left. Measured on camera 13:

            attribute observed at   14:04:40Z   hand_covering = none
            incident opened at      14:05:29Z
            evidence frame stamped  14:05:17Z   ← 37 s after the observation

        The verdict was true and the photograph was of an empty kitchen, which
        is indefensible for an operator being asked to act on it.

        `DECISION_FRAMES` retains the analysed frames and the object boxes cut
        from them, so this looks the subject up by **object and observation
        time** and stores the frame that object was actually seen in. The
        record's `captured_at` is that frame's capture time, not now, and its
        `frame_ref` names the frame — so the evidence can be joined back to the
        observation rather than merely trusted.

        **The fallback is labelled, never silent.** If the decision frame has
        aged out of the ring, the wall's current JPEG is stored with a purpose
        of `…:context-frame` instead of `…:decision-frame`, so nothing ever
        claims to be the decision frame without being it.

        A failure to capture never fails the incident. An incident without a
        picture is still a violation somebody must act on.
        """
        if not getattr(self._settings, "evidence_capture", False):
            return ""

        decision = self._decision_frame(camera_key=camera_key, finding=finding)
        if decision is not None:
            jpeg = decision.jpeg
            captured_at = datetime.fromtimestamp(
                decision.captured_at_ns / 1_000_000_000, tz=UTC
            )
            frame_ref = decision.frame_ref
            kind = "decision-frame"
        else:
            if self._wall is None:
                return ""
            stream = self._wall.get(camera_key)
            if stream is None:
                return ""
            _, jpeg = stream.latest(0, 0.0)
            if not jpeg:
                return ""
            captured_at = datetime.now(UTC)
            frame_ref = ""
            kind = "context-frame"

        from app.domain import evidence as evidence_domain

        ref = _evidence_ref_of(finding) or f"finding:{finding.finding_id}"
        store = evidence_domain.EvidenceStore(session, root=self._settings.evidence_path)
        exhibits = _exhibits(decision, finding=finding, evidence_ref=ref)
        try:
            # The frame first, always. Its manifest names the crops by handles
            # this function chose, so it is complete before any of them exists —
            # and a crop that then fails to store costs a thumbnail, not the
            # photograph the operator actually needs.
            await store.put(
                organization_id=self._settings.default_tenant_id,
                evidence_ref=ref,
                camera_key=camera_key,
                payload=bytes(jpeg),
                captured_at=captured_at,
                purpose=f"compliance:{finding.rule_id}:{kind}",
                retention_days=self._settings.evidence_retention_days,
                frame_ref=frame_ref,
                object_id=str(finding.subject.object_id),
                geometry=exhibits.manifest,
                media_type="image/jpeg",
            )
        except Exception as exc:  # noqa: BLE001 - evidence is not the incident
            logger.warning(
                "evidence capture failed for {} on {}: {}: {}",
                finding.rule_id, camera_key, type(exc).__name__, exc,
            )
            return ""

        for exhibit in exhibits.crops:
            try:
                await store.put(
                    organization_id=self._settings.default_tenant_id,
                    evidence_ref=exhibit.evidence_ref,
                    camera_key=camera_key,
                    payload=exhibit.jpeg,
                    # The crop was cut from this frame, so it was taken at this
                    # frame's instant. Stamping it `now` would put the picture
                    # and the picture-of-part-of-it minutes apart.
                    captured_at=captured_at,
                    purpose=f"compliance:{finding.rule_id}:decision-crop",
                    retention_days=self._settings.evidence_retention_days,
                    frame_ref=frame_ref,
                    object_id=exhibit.object_id,
                    geometry=exhibit.geometry,
                    media_type="image/jpeg",
                )
            except Exception as exc:  # noqa: BLE001 - one exhibit, not the set
                logger.warning(
                    "decision crop {} not stored: {}: {}",
                    exhibit.evidence_ref, type(exc).__name__, exc,
                )
        return ref

    async def _notify(self, incident: Any, finding: Any, run: CompliancePass) -> None:
        """Announce a new incident. **Never fails the incident.**

        A violation that happened is a fact; whether anyone was successfully
        told is a separate one, and rolling back the first because the second
        failed would lose the record of something that really occurred.
        """
        if self._notifier is None:
            return
        try:
            sent = await self._notifier.incident_opened(incident, finding)
            run.notifications_sent += int(bool(sent))
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            run.notifications_failed += 1
            logger.warning(
                "notification failed for incident {}: {}: {}",
                getattr(incident, "id", "?"), type(exc).__name__, exc,
            )

    # -- writing ------------------------------------------------------------

    async def run_once(self) -> CompliancePass:
        """One full pass: read, decide, persist. Safe to call from a route."""
        cameras = await self.cameras()
        run, findings = self.evaluate(self.snapshot(tuple(cameras)))
        self._last = await self.apply(findings, cameras=cameras, run=run)
        return self._last

    async def apply(
        self,
        findings: Any,
        *,
        cameras: dict[str, str | None],
        run: CompliancePass | None = None,
    ) -> CompliancePass:
        """Move the incident queue to match these findings.

        Separate from `run_once` because reading Vision State and deciding what
        an incident should look like are different jobs with different failure
        modes — and because this half is the part that had never existed, so it
        is the part worth being able to test on its own without a live platform.
        """
        run = run or CompliancePass()
        if not findings:
            return run

        from app.domain.incidents import IncidentService
        from compliance import ComplianceState

        async with self._database.session_scope() as session:
            service = IncidentService(session)
            for finding in findings:
                if finding.severity not in RAISES_INCIDENTS:
                    continue
                camera_key = str(finding.subject.camera_id)
                try:
                    if finding.state is ComplianceState.VIOLATION:
                        incident, created = await service.open(
                            organization_id=self._settings.default_tenant_id,
                            restaurant_id=cameras.get(camera_key),
                            camera_key=camera_key,
                            rule_id=finding.rule_id,
                            object_id=str(finding.subject.object_id),
                            observed_at=_observed_at(finding),
                            severity=finding.severity,
                            summary=_summary(finding),
                            ruleset_version=finding.ruleset_version,
                            # Frozen verbatim. A rule change must never rewrite
                            # what was decided about a person months ago.
                            finding=_finding_wire(finding),
                            track_id=str(finding.subject.object_id),
                        )
                        run.incidents_opened += int(created)
                        run.incidents_updated += int(not created)

                        # Evidence and notification happen **only on a genuinely
                        # new incident**. A violation that is still running would
                        # otherwise write a frame and page somebody on every
                        # pass — which is the same spam the incident
                        # deduplication exists to prevent, moved one layer out.
                        if created:
                            ref = await self._capture_evidence(
                                session, camera_key=camera_key, finding=finding
                            )
                            if ref:
                                incident.evidence_refs = ref
                                run.evidence_captured += 1
                            await self._notify(incident, finding, run)

                    elif finding.state is ComplianceState.COMPLIANT:
                        # Only a positive compliant observation closes anything.
                        # UNKNOWN deliberately falls through and changes nothing:
                        # a person who walked out of frame has not put a hairnet
                        # on, and closing on "we can no longer see the violation"
                        # is how a safety system learns to lie.
                        resolved = await service.resolve_by_observation(
                            organization_id=self._settings.default_tenant_id,
                            camera_key=camera_key,
                            object_id=str(finding.subject.object_id),
                            rule_id=finding.rule_id,
                        )
                        run.incidents_resolved += int(resolved is not None)

                except Exception as exc:  # noqa: BLE001 - one finding, not the pass
                    run.errors += 1
                    logger.warning(
                        "compliance could not apply {} for {}: {}: {}",
                        finding.rule_id,
                        finding.subject.object_id,
                        type(exc).__name__,
                        exc,
                    )

        if run.incidents_opened:
            logger.warning(
                "compliance opened {} incident(s) from {} violation(s)",
                run.incidents_opened,
                run.violations,
            )
        return run


#: How many crops one incident may keep. The alert subject first, then the
#: others in the frame. A gallery is a reading aid, and past a handful it stops
#: being one — while every extra crop is another image of an identifiable
#: person written to disk. Bounded here rather than left to the crowd.
MAX_CROPS_PER_INCIDENT = 6


@dataclass(slots=True)
class _CropExhibit:
    """One crop, ready to be stored as evidence in its own right."""

    evidence_ref: str
    object_id: str
    jpeg: bytes
    geometry: str


@dataclass(slots=True)
class _Exhibits:
    """What an incident is allowed to show, decided once."""

    manifest: str = ""
    crops: tuple[_CropExhibit, ...] = ()


def _exhibits(decision: Any, *, finding: Any, evidence_ref: str) -> _Exhibits:
    """The subject's box, its neighbours', and the crops. **Never raises.**

    ### Why this is decided here and frozen

    The boxes live in a bounded in-memory ring holding a couple of minutes of
    analysed frames. An incident is read for days. Whatever is not written down
    now cannot be recovered later — and the tempting recovery, running a
    detector over the stored JPEG, would highlight *a* person in the picture
    rather than the one the verdict was about.

    ### What may be shown as decision evidence

    Only objects of the **finding's own class**, cut from the **same analysed
    frame**. A PPE finding about a person must not present a chair as though it
    contributed to the verdict, and an object whose class was never recorded
    cannot be asserted to be a person — so it is left out of the context set
    rather than guessed at. The subject itself is always present: its class
    comes from the finding, which always carries one.

    ### Labels

    `Person #1`, `Person #2` … assigned left to right across the frame, which
    is how somebody looking at the picture would number them. Presentation
    only: the `object_id` travels beside every label so the trace back to the
    platform's own identity never depends on a display string.
    """
    try:
        subjects = dict(getattr(decision, "subjects", {}) or {})
    except Exception:  # noqa: BLE001 - evidence is not the incident
        return _Exhibits()
    if not subjects:
        return _Exhibits()

    subject_id = str(finding.subject.object_id)
    subject_class = str(getattr(finding.subject, "class_id", "") or "")

    relevant = {
        object_id: entry
        for object_id, entry in subjects.items()
        if object_id == subject_id
        or (subject_class and entry.object_class == subject_class)
    }
    if subject_id not in relevant:
        # The subject was not cut from this frame. Nothing here can be said to
        # be evidence *for this finding*, and a gallery of other people would
        # be worse than none.
        return _Exhibits()

    ordered = sorted(relevant.items(), key=lambda item: (item[1].box[0], item[1].box[1]))
    labels = {
        object_id: f"{_class_noun(subject_class)} #{index}"
        for index, (object_id, _entry) in enumerate(ordered, start=1)
    }

    def _describe(object_id: str, entry: Any) -> dict[str, Any]:
        return {
            "object_id": object_id,
            "class": entry.object_class or subject_class,
            "label": labels[object_id],
            "box": [round(float(v), 6) for v in entry.box],
            "is_subject": object_id == subject_id,
            "sent_to_model": bool(entry.sent_to_model),
        }

    # The subject's crop first, so the cap never spends its budget on
    # bystanders and drops the one image the alert is actually about.
    with_pixels = [
        (object_id, entry)
        for object_id, entry in ordered
        if entry.crop_jpeg
    ]
    with_pixels.sort(key=lambda item: item[0] != subject_id)

    crops: list[_CropExhibit] = []
    crop_refs: dict[str, str] = {}
    for object_id, entry in with_pixels[:MAX_CROPS_PER_INCIDENT]:
        ref = f"{evidence_ref}.crop.{object_id}"
        crop_refs[object_id] = ref
        crops.append(
            _CropExhibit(
                evidence_ref=ref,
                object_id=object_id,
                jpeg=bytes(entry.crop_jpeg),
                geometry=json.dumps(
                    {"kind": "decision-crop", **_describe(object_id, entry)}
                ),
            )
        )

    def _entry(object_id: str, entry: Any) -> dict[str, Any]:
        described = _describe(object_id, entry)
        described["crop_ref"] = crop_refs.get(object_id, "")
        return described

    manifest = {
        "kind": "decision-frame",
        "frame": {
            "frame_ref": str(getattr(decision, "frame_ref", "")),
            "width": int(getattr(decision, "width", 0) or 0),
            "height": int(getattr(decision, "height", 0) or 0),
        },
        "subject": _entry(subject_id, relevant[subject_id]),
        "context": [
            _entry(object_id, entry)
            for object_id, entry in ordered
            if object_id != subject_id
        ],
    }
    return _Exhibits(manifest=json.dumps(manifest), crops=tuple(crops))


def _class_noun(class_id: str) -> str:
    """`person` → `Person`. The operator-facing word for a platform class."""
    return (class_id or "object").replace("_", " ").strip().capitalize()


def _evidence_ref_of(finding: Any) -> str:
    """The platform's own evidence handle for this finding, if it issued one.

    Reusing the platform's id rather than minting a new one keeps the stored
    frame joined to the observation that caused the verdict, instead of leaving
    two unrelated identifiers for the same moment.
    """
    for condition in getattr(finding, "conditions", ()):
        ref = getattr(condition, "evidence_ref", None)
        if ref:
            return str(ref)
    return ""


def _observed_at_ns(finding: Any) -> int:
    """When the attribute this verdict rests on was observed.

    The **failed** condition first: that is the observation the alert is about,
    and it is the one whose frame an operator needs to see. A held or
    unresolved condition may carry a different, later instant, and picking that
    one would quietly reintroduce the "later room state" this exists to stop.
    """
    conditions = tuple(getattr(finding, "conditions", ()))
    failed = [c for c in conditions if getattr(c, "outcome", None) is not None
              and str(getattr(c.outcome, "value", c.outcome)) == "failed"]
    for condition in failed or conditions:
        observed = getattr(condition, "observed_at_ns", None)
        if observed:
            return int(observed)
    return 0


def _summary(finding: Any) -> str:
    """The end-user sentence. `Finding.describe()` already assembles it from
    the rule document and ids — nothing here is generated by a model, which is
    what lets a stored finding regenerate the identical sentence years later.
    """
    return str(finding.describe())


def _finding_wire(finding: Any) -> dict[str, Any]:
    """The finding as JSON, frozen onto the incident.

    Written out here rather than borrowed from a serializer, because this is
    the *historical record*: it must keep the reasoning — what each condition
    expected, what was observed, and why anything unresolved was unresolved —
    so the verdict can be re-read without re-running today's rules against it.
    """
    return {
        "finding_id": finding.finding_id,
        "rule_id": finding.rule_id,
        "rule_version": finding.rule_version,
        "ruleset_version": finding.ruleset_version,
        "schema_version": finding.schema_version,
        "state": finding.state.value,
        "severity": finding.severity,
        "object_id": str(finding.subject.object_id),
        "camera_id": str(finding.subject.camera_id),
        "class_id": str(getattr(finding.subject, "class_id", "")),
        "evaluated_at_ns": getattr(finding.evaluated_at, "ns", 0),
        # A compliant verdict reached under 40% coverage is a different claim
        # from one reached under full coverage, and a reviewer cannot tell
        # without this.
        "coverage_fraction": finding.coverage_fraction,
        "conditions": [
            {
                "attribute": str(c.attribute_key),
                "operator": str(getattr(c, "operator", "")),
                "expected": _plain(getattr(c, "expected", None)),
                "observed": _plain(c.observed),
                # `satisfied` is tri-state on the source — True held, False
                # failed, None could not be established — and it is flattened
                # to a word here rather than to a boolean, because `false` and
                # `null` mean entirely different things to whoever reads this
                # record later and JSON makes them easy to confuse.
                "outcome": (
                    "held" if c.satisfied is True
                    else "failed" if c.satisfied is False
                    else "unresolved"
                ),
                "unknown_reason": getattr(c.unknown_reason, "value", None),
                "observed_at_ns": getattr(c.observed_at, "ns", None),
                # The handle, never the imagery: resolving it needs the
                # separate evidence privilege.
                "evidence_ref": c.evidence_ref,
                "message": c.message,
            }
            for c in finding.conditions
        ],
    }


def _plain(value: object) -> Any:
    """JSON-safe, without pretending an unknown shape is a string."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


__all__ = ["CompliancePass", "ComplianceDriver"]
