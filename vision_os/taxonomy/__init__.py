"""The Visual Taxonomy registry — a platform-owned asset, not a flow layer.

Referenced by 03_MODULES M5 as a Detection Engine dependency and by M11 in
Flow 6. It exists so that "a model-native label never escapes an adapter" is
enforceable rather than aspirational (02_VOM section 8).
"""

from __future__ import annotations

from .registry import DEFAULT_TAXONOMY_VERSION, TaxonomyRegistry

__all__ = ["DEFAULT_TAXONOMY_VERSION", "TaxonomyRegistry"]
