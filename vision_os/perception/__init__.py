"""L2 Perception — establish what things exist.

Single responsibility for the layer: *find things, and follow them. Never say
what they are like, and never say what they mean.*

    Flow 2   detection/   M5 Detection Engine — what is present in this frame
    Flow 3   (absent)     M6 Tracking, M7 Object Registry

Detection is memoryless by construction. Nothing in this layer remembers a
previous frame or holds a persistent identity until Flow 3 introduces the modules
that own those responsibilities.
"""

from __future__ import annotations
