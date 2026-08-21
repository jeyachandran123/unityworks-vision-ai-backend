"""UnityWorks Vision OS (UWV) — the perception platform of UnityWorks AI.

Converts video streams into structured, explainable, domain-neutral observations.
It performs no business reasoning, holds no vertical knowledge, and generates no
alerts. See ``backend/docs/unityWorksVisionOS/`` for the frozen architecture.

Package layout mirrors the architecture's layer stack (01_LAYERED_ARCHITECTURE):

    core/          contracts only — object model + ports. Stdlib-only, no I/O.
    kernel/        L0 — runtime, config, plugins, events, health, metrics.
    acquisition/   L1 — camera manager, source manager, scheduler, buffer.
    adapters/      the ONLY place external technology may appear.
    conformance/   executable port conformance kits (V3).
    testing/       in-memory fakes for every port.

Implementation status: **Flow 1 (Infrastructure & Acquisition)**.
Flows 2-8 (detection, tracking, crop, understanding, observation builder,
vision state, observation API) are NOT implemented.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0-flow1"
