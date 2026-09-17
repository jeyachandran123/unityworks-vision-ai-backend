---
name: vision-os-boundaries
description: Use when a change touches vision_os/, compliance/, app/vision/, config/policies or config/rules; when deciding where new perception, compliance or product logic belongs; when something in the vision pipeline reports zero; or when a boundary or architecture test fails.
---

# Vision OS Boundaries

## Overview

> If it has a business opinion, it does not belong in the platform.

```
app/  ──►  compliance/  ──►  vision_os/          one way; asserted by tests
 └── app/vision/  composition only: joins app to platform, holds no logic
```

"Head covering not visible" is an observation (platform). "Hygiene violation" is a judgment
(compliance or app). `hairnet` lives in `config/policies/*.json`, never in code.

## Where does this code go?

| It… | Goes in |
|---|---|
| detects, tracks, crops, asks a model what is visible | `vision_os/` (migrated verbatim — edits need explicit user confirmation; the protected-tree hook blocks them) |
| turns observations into a rule outcome | `compliance/` (same protection) |
| names a business attribute, threshold or vertical | `config/policies/*.json`, `config/rules/` |
| wires platform layers together, adapts shapes between them | `app/vision/` — `bridges.py` for shape adaptation, `composition.py` for assembly |
| persists, authorizes, serves, audits | `app/` outside `app/vision/` |

If the honest answer is "`vision_os/` needs to change", stop and ask. It is a design change.

## The tests that hold the line

| Rule | Test |
|---|---|
| Platform never imports app or compliance | `tests/app/test_migration.py::TestApplicationDependencyDirection`, `tests/compliance/test_boundaries.py::TestTheDependencyRunsOneWay` |
| Compliance cannot perceive (no inference/network deps, no platform internals, no clock) | `tests/compliance/test_boundaries.py::TestTheRuleEngineCannotPerceive` |
| No domain vocabulary in platform **identifiers** | `tests/vision_os/architecture/test_boundaries.py::TestSemanticCeiling` |
| `core/` stdlib only; layer law; injected clock; no module-level mutable singletons; bounded queues | `tests/vision_os/architecture/test_boundaries.py` (`TestCoreIsPure`, `TestLayerDependencyLaw`, `TestInjectedClock`, `TestDependencyInjection`, `TestBoundedByConstruction`) |
| `EmbeddingPort`, `IdentityResolverPort` stay unbindable | `…::TestFlowScope` |
| Migrated verbatim, relative imports, no `sys.path` | `tests/app/test_migration.py` |
| One `AttributeRegistry` by identity | `tests/app/test_shared_attribute_registry.py` |

`pytest tests/vision_os/architecture tests/compliance/test_boundaries.py tests/app/test_migration.py tests/app/test_shared_attribute_registry.py`

**Known gap:** `TestSemanticCeiling` inspects identifiers only, against a fixed vocabulary
(`kitchen`, `restaurant`, `alert`, …). A string literal such as `"kitchen"` inside `vision_os/`
passes both vocabulary suites. Review literals by eye: `grep -rn '"kitchen\|"hairnet\|"violation' vision_os/`.

## `app/vision/` rules — each one is a past silent outage

1. **Same `AttributeRegistry` object** into `build_registry_layer` and `build_understanding_layer`.
   Equality is not enough. `assert_shared_attribute_registry()` raises at assembly.
2. **Never assign to a platform private** (`runtime._sink = …`). Use declared seams:
   `AdmittedFrameConsumer`, `on_tracked`, `bus.subscribe`; adapt shapes in `bridges.py`.
3. **Count after the call**, per exception class. Never `except Exception: continue`.
4. **One analysis thread.** Registry, tracker, metrics and `VisionStateManager` are shared and not
   thread-safe. The camera wall's per-camera threads are different — each owns its decoder.
5. **No demands → no model calls**, correctly. Check `register_policy_demands` ran first.

## When a layer reports zero

Suspect wiring before the layer. In order: platform assembled (`/api/v1/status`)? demands registered?
same registry instance? tracking sink bound via `on_tracked`? write-back rejections by class in
`writeback.py` counters? Only then look at the model.

## Four-valued semantics

`PRESENT / ABSENT / NOT_VISIBLE / UNKNOWN` stay distinct to the edge. The fold exists once:
`app/domain/observations.py`. Only `ABSENT` may become a violation. Every declared attribute keeps a
`not_visible` refusal value.
