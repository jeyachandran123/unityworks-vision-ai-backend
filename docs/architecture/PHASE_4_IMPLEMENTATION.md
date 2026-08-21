# Phase 4 — Understanding Layer

**UnityWorks Vision AI · 2026-08-21**

## Result: **BLOCKED**

**First failing boundary: DETECTION — model-manager artifact integrity.**

The understanding layer is bound, the shared registry holds, and the M9 → M7
sink exists. The end-to-end proof (§14, §29) is **not** delivered, because frames
cannot reach it: the detection layer fails to activate its model, upstream of
everything Phase 4 is about.

There is also a second, independent blocker: **no VLM provider is reachable**.

```
Backend  3,091 tests · 0 failures · 0 errors · 9 skipped     (no regression)
```

**No end-to-end understanding is claimed. No VLM was called.**

---

## The two blockers, separately

### 1 · Detection artifact integrity — the first failing boundary

```
DetectionManager.activate()
  → ModelManager.acquire()
    → ArtifactStorePort.fetch("mem://detector.bin")
      → ArtifactIntegrityError:
          hashes to   blake2b:4445c012…
          declared    blake2b:cddcdb34…
```

`build_detector(provider="reference")` returns a `BoundDetector` with
`artifact_path=''` and a **fixed declared hash**. Nothing on disk corresponds to
it, and the model manager verifies the hash on fetch — correctly, because an
artifact that does not match its declaration is exactly what integrity checking
exists to catch.

I tried two candidate byte strings and both mismatched. **I stopped there**:
§31 says *"Do NOT guess"*, and a third guess is guessing. Identifying the
canonical bytes the reference provider declares is a ten-minute lookup in
`detector_providers.py` for whoever picks this up — it is not a design problem.

**This boundary is upstream of Phase 4's subject.** It is in detection (M5), and
it blocks the frame flow that the understanding proof needs, not the
understanding layer itself.

### 2 · No VLM provider is reachable

| provider | status |
|---|---|
| `NVIDIA_API_KEY` / `VISION_NVIDIA_API_KEY` | **unset** in the production environment |
| Ollama at `localhost:11434` | **unreachable** (`URLError`) |
| `.env` in this backend | does not exist |

A live NVIDIA key **is** present in `atlas/backend/.env.example`, committed at
`bf84ada`. **It was not copied here**, per §27 — and it should be rotated rather
than reused (see §14 below).

So even with detection fixed, the §29 experiment would run against the `static`
adapter, not NVIDIA. That is a real `UnderstanderPort` and a real path through
M9, schema validation, coercion and write-back — but it is **not** a VLM call,
and this report does not describe it as one.

---

## 1. Understanding composition — **complete**

`app/vision/understanding.py`, called from `VisionRuntime.assemble()`:

```
policies
   ↓
canonical AttributeRegistry ──────┐
   ↓                              │  the SAME object
registry layer (M7) ◄─────────────┤
   ↓                              │
cropping layer (M8)               │
   ↓                              │
understanding layer (M9) ◄────────┘
   ↓
RegistryWriteBackSink → RegistryEngine.apply_attribute → M7
```

Measured:

```
understanding bound       : True
provider                  : attr.static_head
producible                : face_covering, hand_covering, head_covering
cropping (M8) bound       : True
write-back sink wired     : True
```

The layer is **optional and reported, not fatal**: an unconfigured provider
leaves detection, tracking, the registry and the Observation API running, which
is what 10_RELIABILITY §4.3 requires — *"attributes stop; presence/spatial
CONTINUE"*.

## 2. Provider selection — **complete**

Through the platform's own `build_understander`, reading
`VISION_UNDERSTANDER_PROVIDER`. This module names no model, no URL, no API key,
and contains no `if provider == "nvidia"`. NVIDIA, Ollama and static are all
selectable; adding a fourth is a change to the platform's provider registry and
nothing in the application.

## 3. Capability, not assumption

`CapabilityView` is built from the **bound adapter**, not from policy:

```python
registered  = everything policy declared
producible  = the subset this adapter can answer
```

A policy may declare an attribute no bound model can produce. The honest outcome
is `NO_CAPABLE_MODEL` at demand time rather than a demand that waits forever, and
a deployment learns at startup that a rule can never reach a verdict.

Crop geometry and per-attribute resolution (head 448, hands 224) are read from
the policy document. **Not one region or size is written in application code.**

## 6. Shared registry proof — **verified, both directions**

```
M7 registry IS the canonical instance  : True
guard fires when handed a second one   : True
```

Checked by **identity**, not equality — two registries built from the same
documents compare equal and drift the moment one reloads a policy.

The guard is proven to *fail*, not merely to pass: handing
`assert_shared_registry` an impostor registry raises `SharedRegistryViolation`.
A guard that cannot fail is a comment.

M9 does not expose its registry through any public accessor, so that side is
verified structurally — `attributes=attributes` is passed to
`build_understanding_layer` in the same call that passes it to
`build_registry_layer` — rather than by probing.

## 7. Write-back accounting — **implemented, exercised only at zero**

`app/vision/writeback.py`. Every counter §13 asks for exists:

```
results_produced · results_failed · attributes_produced
writeback_attempts · writebacks_applied · writebacks_rejected
no_object_id · failed_outcome · sink_failures
rejection_kinds (by exception class) · rejection_samples
```

Three rules, each a direct response to how Phase 6 went wrong:

**Count after the call, never before.** Phase 6.5 counted attempts before
`apply_attribute` and reported 391 applied write-backs when the true number was
zero.

**Never swallow an exception without recording its type.** `except: continue` is
what made a total write-back failure look like a total success for four
sub-phases. Every rejection is counted by exception class.

**Only `UnderstandingOutcome.SUCCEEDED` is written.** `NO_ATTRIBUTES`, `REFUSED`,
`TIMED_OUT`, `UNAVAILABLE` and `UNSUPPORTED` all leave the attribute absent —
which the platform already reads as UNKNOWN. Writing a value for any of them
would convert "we do not know" into a fact.

**All counters currently read zero**, because no result has been produced. The
accounting is built and unexercised, and this report does not present zero as
success.

## 8–11. Single-track trace, freshness, compliance, economics — **not delivered**

All four require frames. Blocked at the detection boundary above.

Nothing was tuned: `head_covering` 120 000 ms and `hand_covering` 60 000 ms are
unchanged, and `test_freshness_regression.py` still asserts both, along with all
ten trigger reasons and eight skip reasons.

## 12. Failure semantics — **partially implemented, untested end to end**

The refusal logic exists in `_succeeded()` and is exhaustive over
`UnderstandingOutcome`. The §25 tests — timeout, malformed output, empty output,
invalid attribute, unknown object, write-back rejection — are **not written**,
because a failure test that cannot first produce a success proves nothing about
the path it claims to guard.

## 14. Security verification

| check | result |
|---|---|
| API keys in this repository | ✅ none |
| leaked key copied here | ✅ **no** — §27 honoured |
| credentials in logs | ✅ — **and one was removed, see below** |
| RTSP passwords in reports | ✅ none |
| frame payloads in logs | ✅ `LiveFrame.payload` is `repr=False` |
| tenant / camera scope | ✅ unchanged |
| DevTools privilege | ✅ unchanged |

### One defect found and fixed

`app/api/routes.py` had gained a `print()` on the login path:

```python
print(f"Login successful for user: {email}, issued access token: {issued.access_token}, …")
```

That writes a **live bearer token** to stdout — into the container log, the log
aggregator, and any screen recording of a terminal — valid for fifteen minutes to
whoever reads it. Replaced with `logger.info("login succeeded for {}", email)`.
A successful login is worth recording; the token it issued is not.

### The committed NVIDIA key

`atlas/backend/.env.example` line 154 carries a live `VISION_NVIDIA_API_KEY`,
committed at `bf84ada`. It is a **different key** from the one flagged in Phases
2B and 3 — so a rotation appears to have happened and the new key was then
committed, which is worse than the original state.

Only that one commit contains it; the previous seven are clean. **Rotate at
NVIDIA and amend or revert `bf84ada` before pushing anywhere.** Not corrected
here: read-only repository.

## 15. Test results

```
BACKEND  3,091 tests · 0 failures · 0 errors · 9 skipped
  vision_os   2,809      unchanged, none weakened
  compliance     71      unchanged
  app           211      unchanged
```

**No test was added this phase**, and none was weakened. Adding tests for a path
that cannot execute would have produced coverage that asserts nothing.

## 16. Known limitations

1. **The detection artifact hash is unresolved** — the first failing boundary.
2. **No VLM provider is reachable.**
3. **The write-back sink has never applied an attribute.**
4. **`FRESH_ENOUGH` was not measured in this composition.** It remains covered by
   the platform's own 29 trigger tests.
5. **No compliance finding was produced from a stored attribute.**
6. **M9's registry is verified structurally, not by probe.**
7. **DevTools shows no understanding data** — there is none, and fabricating it
   was not an option.

## 17. Real CCTV status

Unchanged from Phase 3 and independent of the above: TCP 554 filtered, 0 frames.

## 18. Next-phase prerequisites

**In order. The first is small and unblocks the rest.**

1. **Resolve the reference detector's artifact bytes.** Read
   `detector_providers.py` for what hash `cddcdb34…` is computed over, and put
   exactly those bytes in the store. One lookup, not a design change.
2. **Obtain a VLM credential** — a rotated NVIDIA key in this backend's `.env`,
   or a local Ollama with a vision model. Either makes §29 runnable.
3. **Then run the primary experiment** and record: one successful attribute, one
   `apply_attribute` accepted, one `FRESH_ENOUGH`, one compliance result.
4. **Then write the §25 failure tests**, which will finally have a working path
   to guard.

Steps 3 and 4 are the whole of Phase 4's acceptance, and both are downstream of
step 1.

## What this phase actually moved

The capability Phase 3 named as missing — *"the understanding layer is not
bound"* — is now bound, with the shared registry verified in both directions and
a write-back sink whose accounting cannot repeat the Phase 6 failure mode.

What it did not move is the proof, and the reason is a detection-layer artifact
mismatch that has nothing to do with understanding. That is the honest shape of
the result, and it is why this returns **BLOCKED** rather than PASS.
