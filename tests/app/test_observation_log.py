"""The observation log's start-up, retention and memory do not grow with history.

Measured on the live deployment on 2026-09-25: the platform's file log parsed a
camera's whole history on the first write after every restart, under the lock
perception appends through — 9.5 million observations, about 55 minutes with
every camera blind, and every id held in memory for the life of the process.

These pin the replacement: the platform's own conformance kit still passes,
start-up parses only the newest records, a re-appended recent record is still
refused across a restart, memory is capped, and retention streams.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.vision import observation_log as module
from app.vision.observation_log import BoundedFileObservationLog, RecentIds
from vision_os.conformance.synthesis_kits import OBSERVATION_LOG_KIT, _observation
from vision_os.core.model.ids import CameraId
from vision_os.core.model.timebase import Instant

CAMERA = CameraId("org-a:cam-12")


def _history(path: Path, lines: int, *, start_ns: int = 1) -> None:
    """A partition file shaped like the real one: one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(lines):
            handle.write(
                json.dumps(
                    {
                        "spatial": {"bbox": [0.1, 0.2, 0.3, 0.4]},
                        "observation_id": f"old-{index}",
                        "camera_id": str(CAMERA),
                        "t_capture_ns": start_ns + index,
                    }
                )
                + "\n"
            )


@pytest.fixture
def parses(monkeypatch):
    """How many records the module parsed as JSON."""
    counted = {"n": 0}
    real = json.loads

    def counting(raw, *args, **kwargs):
        counted["n"] += 1
        return real(raw, *args, **kwargs)

    monkeypatch.setattr(module.json, "loads", counting)
    return counted


# ── the platform's own contract ──────────────────────────────────────────────


def test_the_platform_conformance_kit_passes(tmp_path: Path) -> None:
    report = OBSERVATION_LOG_KIT.run(BoundedFileObservationLog(tmp_path), fast_only=False)
    assert report.passed, report.failures


def test_the_boot_gate_certifies_this_class_not_the_one_it_replaced(tmp_path: Path) -> None:
    twin, dispose = BoundedFileObservationLog(tmp_path).for_conformance()
    try:
        assert type(twin) is BoundedFileObservationLog
        assert OBSERVATION_LOG_KIT.run(twin, fast_only=True).passed
    finally:
        dispose()


# ── start-up ─────────────────────────────────────────────────────────────────


def test_start_up_counts_history_without_parsing_it(tmp_path: Path, parses) -> None:
    log = BoundedFileObservationLog(tmp_path, start_ids=100)
    _history(log._path(CAMERA), 30_000)

    assert int(log.position(CAMERA)) == 30_000
    # The newest hundred, not thirty thousand.
    assert parses["n"] <= 101


def test_positions_after_a_restart_continue_from_the_file(tmp_path: Path) -> None:
    log = BoundedFileObservationLog(tmp_path)
    _history(log._path(CAMERA), 3)

    appended = log.append(CAMERA, [_observation("new")])
    assert int(appended.position) == 4
    assert [str(o.observation_id) for o in log.read(CAMERA, start=appended.position)] == []
    tail = list(log.tail(CAMERA, start=3))
    assert [str(o.observation_id) for o in tail] == ["kit-obs-new"]


def test_a_record_written_before_a_restart_is_still_refused_after_it(tmp_path: Path) -> None:
    first = BoundedFileObservationLog(tmp_path)
    first.append(CAMERA, [_observation("a"), _observation("b")])

    restarted = BoundedFileObservationLog(tmp_path)
    result = restarted.append(CAMERA, [_observation("b"), _observation("c")])

    assert result.appended == 1
    assert [str(o) for o in result.duplicates] == ["kit-obs-b"]
    assert int(result.position) == 3


# ── memory ───────────────────────────────────────────────────────────────────


def test_remembered_ids_are_capped_while_running(tmp_path: Path) -> None:
    log = BoundedFileObservationLog(tmp_path, recent_ids=10)
    for index in range(25):
        log.append(CAMERA, [_observation(f"n{index}")])

    assert len(log._ids[CAMERA]) == 10
    assert int(log.position(CAMERA)) == 25


def test_recent_ids_forget_the_oldest_first() -> None:
    ids = RecentIds(["a", "b", "c"], capacity=2)
    assert set(ids) == {"b", "c"}
    ids.update(["c", "d"])
    assert set(ids) == {"c", "d"}


# ── retention ────────────────────────────────────────────────────────────────


def test_retention_removes_only_what_is_older_and_keeps_order(tmp_path: Path, parses) -> None:
    log = BoundedFileObservationLog(tmp_path)
    path = log._path(CAMERA)
    _history(path, 10, start_ns=1)
    log.position(CAMERA)  # loaded, as a running process would be
    parses["n"] = 0

    removed = log.truncate(CAMERA, Instant(6))
    # Read by its capture time alone: no record was parsed to decide.
    parsed_by_truncate = parses["n"]

    assert removed == 5
    assert parsed_by_truncate == 0
    kept = [json.loads(line)["t_capture_ns"] for line in path.read_text().splitlines()]
    assert kept == [6, 7, 8, 9, 10]
    # The partition renumbers from the new file, and appends continue.
    assert int(log.position(CAMERA)) == 5
    assert int(log.append(CAMERA, [_observation("after")]).position) == 6


def test_retention_with_nothing_old_leaves_the_file_untouched(tmp_path: Path) -> None:
    log = BoundedFileObservationLog(tmp_path)
    path = log._path(CAMERA)
    _history(path, 4, start_ns=100)
    before = path.read_bytes()

    assert log.truncate(CAMERA, Instant(50)) == 0
    assert path.read_bytes() == before
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")], "temporary left behind"


def test_retention_on_a_camera_with_no_file_is_nothing(tmp_path: Path) -> None:
    assert BoundedFileObservationLog(tmp_path).truncate(CAMERA, Instant(50)) == 0


# ── the application runs this log ────────────────────────────────────────────


def test_the_application_binds_this_log_for_a_durable_deployment(tmp_path: Path) -> None:
    from app.configuration.settings import Settings
    from app.vision.runtime import VisionRuntime

    settings = Settings(
        app_env="test",
        secret_key="test-only-secret-value-not-for-any-deployment",
        database_url_override="sqlite+aiosqlite:///:memory:",
        redis_enabled=False,
        observation_log="file",
        observation_log_path=str(tmp_path / "observations"),
    )
    log = VisionRuntime(settings)._observation_log()
    assert type(log) is BoundedFileObservationLog
