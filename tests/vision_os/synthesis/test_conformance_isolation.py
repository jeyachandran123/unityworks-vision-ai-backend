"""The conformance kit must never write into a deployment's durable log.

### The outage this pins

The gate runs each adapter's kit before the adapter is used, and the kit writes
real records — a store can only be shown to store by storing. Against
`FileObservationLog` it wrote seven `kit-*` partitions straight into the
deployment's observation directory, and that class has no `reset()` for
`_purge_kit_traces` to call, so they stayed.

On the **next** boot the kit ran again over the same directory, found its own
records already there, and failed:

    [L2] semantics/idempotent_by_id
    [L3] semantics/positions_are_monotonic: positions repeated
    [L7] semantics/tail_follows_without_blocking

The gate raised, synthesis never bound, exposure was never built, and the
compliance driver read zero subjects every five seconds. Detection, tracking and
the model kept running and costing money; nothing they produced could leave M7.
Alerts simply stopped, and `/health/ready` still said `vision_os: true`.

It was self-inflicted and permanent — once poisoned, every subsequent boot
failed, and the only remedy was deleting files by hand.

### Why a twin and not a cleanup

Cleanup is the dangerous shape: anything that deletes `kit-*` partitions from a
live log is one bad glob away from deleting a camera's observations — the system
of record — to tidy up a test fixture. A twin cannot make that mistake.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from vision_os.adapters.synthesis.stores import FileObservationLog
from vision_os.conformance.synthesis_kits import OBSERVATION_LOG_KIT
from vision_os.synthesis_bootstrap import _conformance_twin


@pytest.fixture
def durable_root():
    root = Path(tempfile.mkdtemp(prefix="conformance-isolation-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _gate_once(log: FileObservationLog):
    """Exactly what `_gate` does, minus the platform plumbing."""
    probe, dispose = _conformance_twin(log)
    try:
        return OBSERVATION_LOG_KIT.run(probe, fast_only=True), probe is not log
    finally:
        dispose()


class TestRestartIsIdempotent:
    def test_the_kit_passes_on_every_boot_not_just_the_first(self, durable_root):
        """Boot 1 passed before this repair and boot 2 did not. The property
        that matters is that boot N is indistinguishable from boot 1."""
        for boot in range(1, 6):
            report, used_twin = _gate_once(FileObservationLog(durable_root))
            assert report.passed, (
                f"boot {boot} failed conformance: {'; '.join(report.failures)}"
            )
            assert used_twin, "the durable log must be gated on a twin"

    def test_no_kit_partition_is_ever_written_to_the_durable_log(self, durable_root):
        """The mechanism, observed directly. Seven of these files in production
        are what ended observation publication."""
        for _ in range(3):
            _gate_once(FileObservationLog(durable_root))

        strays = sorted(p.name for p in durable_root.glob("kit-*.jsonl"))
        assert strays == [], f"the kit leaked {strays} into the durable log"


class TestProductionDataIsNeverTouched:
    def test_a_real_partition_survives_the_gate_byte_for_byte(self, durable_root):
        """The system of record must come through the gate unaltered. This is
        the guarantee a cleanup-based fix could not make."""
        real = durable_root / "cam-11.jsonl"
        real.write_text('{"observation_id":"real-1"}\n', encoding="utf-8")
        before = real.read_bytes()

        for _ in range(3):
            _gate_once(FileObservationLog(durable_root))

        assert real.exists(), "the gate must never delete a camera's observations"
        assert real.read_bytes() == before

    def test_the_durable_directory_gains_nothing_at_all(self, durable_root):
        real = durable_root / "cam-12.jsonl"
        real.write_text('{"observation_id":"real-2"}\n', encoding="utf-8")

        _gate_once(FileObservationLog(durable_root))

        assert sorted(p.name for p in durable_root.glob("*.jsonl")) == ["cam-12.jsonl"]


class TestTheTwinIsDisposable:
    def test_the_twin_writes_somewhere_else_and_is_removed(self):
        """The kit's records must exist somewhere while it runs — otherwise it
        is not proving the adapter stores — and must not exist afterwards."""
        log = FileObservationLog(Path(tempfile.mkdtemp(prefix="unused-")))
        twin, dispose = _conformance_twin(log)

        assert twin is not log
        OBSERVATION_LOG_KIT.run(twin, fast_only=True)
        twin_root = twin._root  # noqa: SLF001 - asserting the twin's isolation
        assert list(twin_root.glob("kit-*.jsonl")), "the kit must really write"

        dispose()
        assert not twin_root.exists(), "the twin must be removed whole"

    def test_an_adapter_without_the_factory_is_gated_in_place(self):
        """The protocol is optional. An adapter that cannot provide a twin is
        gated exactly as it always was rather than being refused."""

        class _NoFactory:
            pass

        adapter = _NoFactory()
        probe, dispose = _conformance_twin(adapter)
        assert probe is adapter
        dispose()  # must be safe to call

    def test_a_factory_that_raises_falls_back_rather_than_failing_boot(self):
        """Gating the real adapter is worse than gating a twin, and far better
        than refusing to start."""

        class _Broken:
            def for_conformance(self):
                raise RuntimeError("no temp space")

        adapter = _Broken()
        probe, dispose = _conformance_twin(adapter)
        assert probe is adapter
        dispose()
