"""Vision OS core — contracts only.

This package contains the Vision Object Model (02_VISION_OBJECT_MODEL) and the
port protocols (06_PORTS_AND_ADAPTERS). It is deliberately **stdlib-only**: no
third-party import may appear anywhere beneath ``core/``. The architecture
boundary test in ``tests/vision_os/architecture/`` enforces this mechanically.

Core performs no I/O, owns no mutable global state, and knows nothing about any
concrete detector, tracker, codec, database, or vendor (invariant V3).
"""

from __future__ import annotations
