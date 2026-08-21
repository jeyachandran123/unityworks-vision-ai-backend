"""UnityWorks Vision AI — production backend.

Two packages sit beside this one and are **not** part of it:

    vision_os/    the perception platform, migrated verbatim
    compliance/   the rule engine that consumes it

This package depends on both. Neither depends on this one, and a boundary test
asserts that direction — the moment it reverses, the platform has acquired a
business opinion.
"""

__version__ = "1.0.0"
