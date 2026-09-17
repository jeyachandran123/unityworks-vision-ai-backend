# Invariants — Vision AI backend

Read by the `invariant-audit` skill and the architecture-auditor / qa-engineer agents. Each row: the
invariant, the test that must hold it, and a realistic mutation that should turn that test red.
Run tests with the repository's venv: `.venv/Scripts/python.exe -m pytest <node-id>` — check the exit
code; `addopts = -q` plus another `-q` hides the summary line.

| Invariant | Test | A realistic mutation |
|---|---|---|
| Vision OS migrated verbatim; relative imports only | `tests/app/test_migration.py::TestVisionOsIsUnchanged` | Add `from vision_os.core import x` inside `vision_os/` |
| No `sys.path` manipulation | `tests/app/test_migration.py::TestNoSiblingRepositoryDependency::test_no_source_file_mutates_sys_path` | `sys.path.insert(0, "..")` in an `app/` module |
| No sibling-repo dependency | `tests/app/test_migration.py::TestNoSiblingRepositoryDependency::test_the_platform_imports_in_a_subprocess_with_no_extra_path` | Import something from `../` |
| `app → compliance → vision_os` one way | `tests/compliance/test_boundaries.py::TestTheDependencyRunsOneWay`, `tests/app/test_migration.py::TestApplicationDependencyDirection` | `import compliance` in a `vision_os/` module |
| Semantic ceiling (no business words in platform) | `tests/vision_os/architecture/test_boundaries.py::TestSemanticCeiling`, `tests/compliance/test_boundaries.py::TestNoDomainVocabularyInCode` | Identifier `kitchen_alert = True` in `vision_os/perception/__init__.py` (verified red). A literal `"kitchen"` there stays **green** in both suites — known gap: `TestSemanticCeiling` scans identifiers only |
| One `AttributeRegistry`, by identity | `tests/app/test_shared_attribute_registry.py` | Build a second registry in `app/vision/composition.py` |
| Three/four-valued compliance | `tests/compliance/test_dataset_regression.py::TestTheSafetyContract::test_not_visible_never_produces_a_violation`, `tests/app/test_observations.py::test_not_visible_survives_the_fold_unchanged`, `tests/app/test_freshness_regression.py::TestThreeValuedSemantics::test_every_declared_attribute_carries_a_refusal_value` | Map `not_visible` → `none` in `app/domain/observations.py` |
| Deny by default (no empty camera tuple) | `tests/app/test_foundation.py -k "no_access_cannot_become"` | Return `()` instead of raising in `AccessDecision.to_grant()` |

## Tautology check

Backend role → permission mapping lives in `app/authorization/model.py` (`permissions_for`). Test
fixtures `make_user` / `admit` in `tests/app/conftest.py` must hold exactly those permissions.
