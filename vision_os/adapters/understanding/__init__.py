"""Understanding adapters — reference understanders, coercion, and the M10 seam.

**Nothing outside this package and the composition root may name an
understander or a coercion strategy.** The platform holds P15 and P16; which
implementation satisfies each is a configuration fact, exactly as Flow 2 keeps
YOLO invisible and Flow 5 keeps crop strategies invisible.

**Two VLM adapters now ship**, alongside the reference ones. This used to read
"no VLM ships", on the grounds that binding one needs weights, a runtime and a
device — M18's concern and a deployment's choice. That is still true of the
*weights*; what changed is that both adapters here need none of their own. One
speaks to a hosted OpenAI-compatible endpoint (``nvidia_vl``), the other to a
locally-served model (``ollama_vl``), and each carries only a URL, a model name
and a timeout.

What ships, then:

| Adapter | Residency | Needs |
|---|---|---|
| ``NvidiaVisionUnderstander`` | ``remote`` | an API key |
| ``OllamaVisionUnderstander`` | ``local`` | a reachable Ollama |
| ``StaticAttributeHead`` | ``local`` | nothing |
| ``ScriptedUnderstander`` | ``local`` | nothing |
| ``UnavailableUnderstander`` | ``local`` | nothing |

The reference three remain the reason the engine's paths are testable without a
model: between them they exercise every branch it has without making a test
depend on what a model happened to say.

Which one a deployment binds is resolved by
``adapters/configuration/understander_providers.py``, and the platform is never
told the name — ``build_understanding_layer`` receives a bound adapter, exactly
as P15 requires. Resolution lives over there rather than here because §M9
forbids ambient state in this package, and reading an environment is ambient
state; the architecture suite enforces it.
"""

from .coercion import (
    MAX_SCAN_CHARS,
    JsonCoercion,
    KeyValueCoercion,
    PassthroughCoercion,
)
from .nvidia_vl import NvidiaVisionUnderstander
from .ollama_vl import OllamaVisionUnderstander
from .payload import encode_png_base64, extract_json, split_by_schema
from .prompts import PROVIDER_ID, PromptTemplate, StaticPromptProvider
from .understanders import (
    ScriptedAnswer,
    ScriptedUnderstander,
    StaticAttributeHead,
    UnavailableUnderstander,
)

#: Coercion strategies selectable by configuration.
#:
#: A closed table, like the tracker and crop-strategy factories. A deployment
#: names a strategy; it does not import one.
COERCION_FACTORIES = {
    "coercion.json": JsonCoercion,
    "coercion.keyvalue": KeyValueCoercion,
    "coercion.passthrough": PassthroughCoercion,
}

__all__ = [
    "COERCION_FACTORIES",
    "MAX_SCAN_CHARS",
    "PROVIDER_ID",
    "JsonCoercion",
    "KeyValueCoercion",
    "NvidiaVisionUnderstander",
    "OllamaVisionUnderstander",
    "PassthroughCoercion",
    "PromptTemplate",
    "ScriptedAnswer",
    "ScriptedUnderstander",
    "StaticAttributeHead",
    "StaticPromptProvider",
    "UnavailableUnderstander",
    "encode_png_base64",
    "extract_json",
    "split_by_schema",
]
