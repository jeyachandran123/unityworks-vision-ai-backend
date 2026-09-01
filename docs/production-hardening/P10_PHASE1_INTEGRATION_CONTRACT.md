# P10 Phase 1 — Integration Contract for `meta/llama-3.2-90b-vision-instruct`

**Status:** Phase 1 only. **No production code has been changed.**
**Date:** 2026-08-31
**Scope:** define exactly what it means for the 90B model to be a *supported* NVIDIA
Understanding model, and prove by test that configuration can select it through the
normal adapter path.

Production remains on `meta/llama-3.2-11b-vision-instruct`. Nothing in this phase
switches it.

---

## 0. What "supported" means here

A model is **supported** when every one of the seventeen properties in §1–§17 below is
either (a) already true of the existing NVIDIA adapter path, or (b) named as a defect
with the phase that will fix it. Nothing is deferred silently.

A model being *supported* is **not** a claim that it is *production-ready*. That is
Phase 4's question (freshness) and Phase 9's verdict. This document deliberately says
nothing about whether the 90B should be switched on.

---

## 1. Configuration source

| | |
|---|---|
| Primary | `VISION_NVIDIA_MODEL` |
| Secondary | `NVIDIA_MODEL` |
| Last resort | `nvidia_vl.DEFAULT_MODEL` |

Read in `_build_nvidia` via `_setting(env, "VISION_NVIDIA_MODEL", "NVIDIA_MODEL",
default=DEFAULT_MODEL)`. Layering is `defaults → file → environment`.

`.env` reaches this through `Settings.understander_options()`, which emits **only
non-empty values** — so an unset variable means "the adapter's default applies", never
"the empty string".

**Contract:** selecting the 90B is `VISION_NVIDIA_MODEL=meta/llama-3.2-90b-vision-instruct`
and nothing else. No source edit. `DEFAULT_MODEL` is the fallback layer and must not be
the mechanism by which a deployment chooses a model.

> **Known defect, Phase 5.** The working tree currently has
> `DEFAULT_MODEL = "moonshotai/kimi-k3"`, edited by hand during model trials. It is
> overridden by `.env` today, so it changes no behaviour — but it means the source
> default and the deployment disagree, which is exactly the drift this contract forbids.

---

## 2. Model resolution

```
Settings(.env)
   └─ understander_options()          # only non-empty values
        └─ build_understander(defaults=…)
             └─ _build_nvidia(env=…)
                  └─ NvidiaVisionUnderstander(model=…)
                       └─ self._model  →  payload["model"]
```

The string is passed through verbatim. No normalisation, no alias table, no vendor
prefix handling. `meta/llama-3.2-90b-vision-instruct` therefore requires **no code
change to resolve** — verified by the Phase 1 tests in §18.

---

## 3. Adapter binding

`UNDERSTANDER_FACTORIES` is a **closed table** — `{"nvidia", "ollama", "static"}`. The
90B binds through the existing `"nvidia"` entry.

**No new provider and no new adapter is required, and none may be added.** The brief
forbids it and the architecture does not need it: the endpoint is OpenAI-compatible and
the model differs only by string.

Binding failure is a **composition-time** error (`ProviderConfigurationError`), never a
request-time one — "a deployment that cannot bind an understander must fail while
someone is watching the logs, not on the first crop of a live session."

---

## 4. Request construction

`POST {base_url}/chat/completions`, one crop per call:

```json
{"model": "<configured>",
 "messages": [{"role": "user", "content": [
     {"type": "text", "text": "<rendered prompt>"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,<crop>"}}]}],
 "max_tokens": "<min(request.max_tokens, prompt.max_output_tokens)>",
 "temperature": "<engine config, 0.0>"}
```

The adapter decides **none** of this content. The prompt arrives rendered and
version-pinned by P17; the output schema arrives with the request; the attribute
vocabulary is enforced after return by `AttributeValidator`. There is no domain
vocabulary in the adapter and none may be added.

**Contract:** the 90B receives byte-identical request structure to the 11B. Only the
`model` field differs. Confirmed compatible in Phase 0 smoke testing; Phase 2 verifies
it through the real engine path rather than a standalone script.

---

## 5. Image transport

`encode_png_base64(crop.pixels, w, h, colour_space=…, max_side=self._max_side)`

- BGR24 → RGB, hand-rolled PNG (zlib only, **no Pillow/OpenCV dependency**)
- nearest-neighbour downscale to `max_side`, integer stride
- inlined as a `data:image/png;base64,…` URL — **no external image host**

Two consequences worth stating rather than discovering:

1. **Vision tokens scale with area**, so `max_side` is the dominant cost lever. The
   NVIDIA-hosted example image measured 6,423 prompt tokens; a 448px kitchen crop is far
   smaller.
2. The resampler is a **pure-Python per-pixel loop**. At 448px that is ~200k iterations
   per crop on the calling thread. Negligible beside a 73s inference, but it is CPU spent
   inside the concurrency cap and belongs in Phase 4's throughput arithmetic.

> **Known defect, Phase 5.** `_build_nvidia` **does not forward `max_side`**. The
> constructor default (448) coincides with the configured `VISION_UNDERSTANDER_MAX_SIDE=448`,
> so nothing is visibly wrong today — and any other value is silently ignored. The Ollama
> factory has the same gap.

---

## 6. Timeout behaviour

Two independent budgets, and they compose by `max`, not by `min`:

| Budget | Value | Role |
|---|---|---|
| `UnderstandingSection.timeout_ms` | 2,000 | engine **hint**, passed as `request.timeout` |
| `VISION_UNDERSTANDER_TIMEOUT_S` | 600 | adapter's own socket deadline |

```python
timeout = self._timeout
if request.timeout is not None:
    timeout = max(timeout, request.timeout.millis / 1000.0)
```

**The engine does not enforce its own deadline.** There is no `future.result(timeout=…)`
anywhere in the call path — `timeout_ms` only travels as a hint, and the adapter takes
the more generous of the two. So the effective per-call deadline for the 90B is **600
seconds**, not 2.

**Contract:** this is correct and must not be "fixed". A slow model is not cut off
mid-flight; it is governed by freshness (Phase 4) and by concurrency (§7), which are the
right controls. Lowering the adapter timeout to match the engine hint would convert slow
answers into timeouts, which is strictly worse: a timeout yields no evidence at all,
whereas a late answer at least reaches the staleness rules that can judge it.

---

## 7. Concurrency behaviour

| | |
|---|---|
| `max_concurrency` | 4 (local models) |
| `remote_concurrency` | **2** (this adapter is `data_residency="remote"`) |
| Effective cap | `min(4, 2) = 2` in flight |
| Over-cap behaviour | **shed**, never queued |

```
UnderstanderUnavailableError: "enrichment is shed rather than queued, because a
queued call outlives the frame it describes"
```

**Contract:** the shed-not-queue rule is a safety property and is untouchable. It is
precisely what stops a 73-second model from building an unbounded backlog of answers
about frames that no longer exist. A shed call produces no attribute, which the rules
read as UNKNOWN — honest, and never mistaken for compliance.

Throughput arithmetic is Phase 4's job. Stated here only as the mechanism: 2 in flight
at 73 s ⇒ ~98 calls/hour ceiling, against a policy budget of 400.

---

## 8. Retry behaviour

| | |
|---|---|
| `max_retries` | 1 — *"Retry once with backoff; then fallback model; then fail the request."* |
| `retry_backoff_ms` | 100 |
| `circuit_breaker_threshold` | 3 consecutive failures |
| `circuit_breaker_cooldown_ms` | 30,000 |
| `fallback_depth` | 2, terminating in explicit unavailability |

**Contract:** retries stay bounded at one. The brief forbids aggressive retry loops and
the existing design already agrees — *"a transient blip resolves on the first retry and a
real outage does not resolve on the third."*

This matters specifically for the 90B: at 73 s mean, one retry costs ~146 s of a 2-slot
concurrency budget. Retry policy must not be widened to compensate for a slow model.

---

## 9. Rate-limit behaviour (HTTP 429)

Already classified, from the MiniMax work earlier today:

- `RateLimitedError`, its own type — no layer matches on message text
- counted as `stats.rate_limited`, and recorded in the health window as `"rate_limited"`
- **not** a retirement: a quota resets, so it is never latched
- sustained ⇒ `health()` reports `state: "rate_limited"`, `available: false`

**Contract:** unchanged by this integration, and reused as-is.

---

## 10. HTTP 504 behaviour — **the open question of Phase 3**

Currently **unclassified**. Falls to:

```python
raise RuntimeError(f"HTTP {exc.code} from the model service: {detail}")
```

which the engine turns into a refusal. Measured once against the 90B: **HTTP 504 after
302 seconds**.

The safety-relevant part is already correct — a 504 becomes `structured={}`, `refused=True`,
and therefore **never a PPE value**. What is missing is *classification*: a gateway
timeout is a transient upstream condition, distinct from both a quota (429) and a
permanent retirement (404/410), and an operator cannot currently tell them apart without
reading log text.

**Phase 3 must decide, with evidence, whether 504/502/503/408 are:**

1. retried once under the existing `max_retries=1` (they are, today, as generic failures), and
2. surfaced in `health()` as a named degraded state rather than folded into `failing`.

**Constraint carried into Phase 3:** whatever classification is chosen, a 504 must remain
a refusal. Nothing about naming it may create a path where it becomes an attribute.

---

## 11. Health reporting

| State | Trigger | `available` |
|---|---|---|
| `ok` | recent success ≥ 50%, or < 10 samples | `true` |
| `rate_limited` | sustained failure, majority 429 | `false` |
| `failing` | sustained failure, other causes | `false` |
| `model_retired` | latched 404/410 | `false` |

Window: last 32 calls, minimum 10 before judging.

**Contract, and the brief's hard rule:** *a model that answers zero requests must never
report itself as healthy.* With the current window, 0/10 ⇒ `failing`, `available: false`.
This holds for the 90B without modification — verified in Phase 3's regression tests.

`health()["available"]` is **report-only**; it gates no pipeline stage.

---

## 12. Parser behaviour

`extract_json(text)` → `dict | None`, then `split_by_schema(decoded, output_schema)`.

Shapes handled today, all measured:

| Shape | Source | Handled |
|---|---|---|
| `{"k": "v"}` | 90B | ✅ |
| ` ```json {…}``` ` | MiniMax M3 | ✅ |
| ` ``` {…}``` ` | — | ✅ |
| `"k": "v"` braceless, quoted | 11B | ✅ |
| prose around an object | — | ✅ |
| `"k": v` braceless, **unquoted** | 11B | ❌ unparseable |
| `**K:** v` markdown | 11B | ❌ unparseable |

**Contract:** the parser must remain **model-agnostic**. Phase 2 forbids making it
model-specific unless absolutely necessary, and nothing observed so far requires it —
the 90B returns clean braced JSON and needs no new branch. Any hardening for the 11B's
unquoted/markdown shapes must be shape-driven, never keyed on model name.

`None` ⇒ `stats.unparseable`, `structured={}`, health window `"failed"`. **No partial
dict is ever returned**, so a half-read answer cannot become an attribute.

---

## 13. Refusal semantics

Port obligation **U2 — never fabricate.** Every failure path returns
`structured={}`, `refused=True`, `raw_output=b""`:

- crop encoding failure
- socket / connection timeout
- 429 (`RateLimitedError`)
- 404 / 410 (`ModelRetiredError`, latched)
- any other HTTP status, including 504
- empty response body
- unparseable body *(returns `refused=False` but `structured={}` — see below)*

> **Precision worth keeping.** An unparseable reply is **not** a refusal: it returns
> `refused=False` with `structured={}`. Downstream both yield no attribute, so safety is
> identical — but they are different facts, and conflating them is exactly the error
> corrected in `FINAL_LIVE_PPE_DETECTION_REPORT.md` §24a, where "answered" was counted as
> success. Phase 6 must count **parsed**, not `not refused`.

A refusal reaches compliance as an **absent attribute** → `UnknownReason.ATTRIBUTE_ABSENT`
→ UNKNOWN. Never COMPLIANT, never VIOLATION.

---

## 14. Metrics

Emitted by the engine around every invocation, all label-partitioned by camera and model:

`understanding.results{outcome}`, `.refusals{model}`, `.unsupported`, `.timeouts`,
`.retries`, `.fallbacks`, `.circuit_open`, `.failures`, `.adapter_errors`,
`.concurrency_rejected{model}`, `.in_flight`, `.cache_hits/misses`, `.latency_ms`,
`.cost_units`, `.attributes_produced`.

Adapter-local, via `stats.to_wire()`: `requests, succeeded, failed, refused, timed_out,
unparseable, rate_limited, prompt_tokens, eval_tokens, p50/p95_latency_ms`.

**Contract:** `refusals{model=…}` and `concurrency_rejected{model=…}` are already
model-labelled, so 90B and 11B traffic separate in the metrics without any change. This is
the mechanism Phase 6 uses to attribute live qualification numbers.

> `vision_os.cropping.budget_spent` is emitted as `.increment(0)` and can never move.
> Pre-existing, cosmetic, suppresses nothing. Out of scope; recorded so it is not
> mistaken for a 90B effect.

---

## 15. Logging

Adapter logs nothing itself. Composition logs the binding
(`"understanding bound — provider={} producible={} ({})"`) and reachability
(`_report_reachability`), which names the model and endpoint on failure.

**Contract:** `health()["reason"]` and every log line carry the model name and the
upstream detail, and **never the API key**. Already asserted by
`test_the_reason_never_carries_the_key`. Two live keys were pasted into an operator
conversation today; neither reached source, config, tests or reports, and both should be
rotated.

---

## 16. Evidence attribution

```python
ModelMeta(model_id=ModelId(self._model),                      # full string, exact
          model_version=self._model.rsplit("-", 1)[-1],       # derived
          artifact_hash=f"nvidia:{self._model}",              # full string
          adapter_id="understander.nvidia_vl",
          deterministic=False)
```

**Provenance is preserved exactly** — `model_id` and `artifact_hash` both carry the full
configured string, so a 90B observation is distinguishable from an 11B one at the
observation level, permanently, with no extra work.

> **Weakness, recorded not fixed.** `model_version` is derived by `rsplit("-", 1)` and
> yields **`"instruct"`** for `meta/llama-3.2-90b-vision-instruct` — and `"instruct"` for
> the 11B too. It is not a version and cannot distinguish the two models. It is cosmetic
> and nothing depends on it for provenance; `model_id` does that job. Changing the
> derivation would alter the shape of previously recorded evidence, so it stays.
> **Phase 6 must attribute by `model_id`, never by `model_version`.**

`deterministic=False` is honest: a hosted model behind a load balancer is not
reproducible even at temperature 0.

**Non-negotiable, restated:** new 90B evidence goes in new files. Historical 11B
evidence — `tests/compliance/kitchen01_model_answers.json`, the 13 `experiments/vlm_prompt/runs/*.json`
— stays attributed to `meta/llama-3.2-11b-vision-instruct`. These were relabelled once
already today by a repo-wide find-and-replace and had to be restored from `58989cd`.

---

## 17. What this contract does **not** cover

Deliberately out of scope for Phase 1, each with its owning phase:

| Question | Phase |
|---|---|
| Does the 90B work through the real engine path? | 2 |
| How should 504/502/503/408 be classified? | 3 |
| **Can the 90B meet attribute freshness at 73 s?** | **4** |
| `max_side` forwarding fix, `DEFAULT_MODEL` drift | 5 |
| Live qualification | 6 |
| Safety regression | 7 |
| Full regression + lint | 8 |
| Production verdict | 9 |

**Phase 4 is the phase that decides this integration.** Phase 0 measured
`observed_at = request.t_capture` (frame capture time), against `face_covering` and
`hand_covering` validity of 60 s and a measured 73 s mean latency. That arithmetic is
carried into Phase 4 unresolved and must not be pre-empted here.

---

## 18. Phase 1 verification

Tests added: `tests/vision_os/understanding/test_ninety_b_configuration.py` — **19 tests,
all passing.** No production code changed.

They assert the contract's central claim — that the 90B resolves through the ordinary
NVIDIA path by configuration alone:

| Group | Asserts |
|---|---|
| Resolution | `VISION_NVIDIA_MODEL` and `NVIDIA_MODEL` both select it; primary wins; whitespace tolerated |
| Layering | `.env` → `Settings` → factory → adapter, end to end; environment beats defaults; empty never overrides |
| No source dependence | selecting it requires **no** `DEFAULT_MODEL` edit; a changed default cannot override an explicit setting |
| No hard-coding | the string appears in **no** production module under `app/`, `vision_os/`, `compliance/` |
| Binding | binds the existing `nvidia` factory; adds no entry to `UNDERSTANDER_FACTORIES`; satisfies the port kit |
| Attribution | `model_id` / `artifact_hash` carry the exact string; 90B and 11B are distinguishable; `model_version` is pinned as the known-degenerate value so the weakness cannot regress unnoticed |
| Residency | still declared `remote`, so a residency-policy site still refuses it |

---

## 19. Phase 1 verdict

**The 90B is selectable as a supported NVIDIA Understanding model by configuration alone,
with no new provider, no new adapter and no source edit.** §1–§16 are satisfied by the
existing architecture as written.

Three defects are carried forward, none blocking Phase 2:

1. `max_side` not forwarded by `_build_nvidia` → **Phase 5**
2. `DEFAULT_MODEL` drifted to `moonshotai/kimi-k3` in the working tree → **Phase 5**
3. HTTP 504 unclassified → **Phase 3**

One question is carried forward that **can** block the integration outright:

4. **73 s mean latency against 60 s attribute validity** → **Phase 4**

Proceed to Phase 2.
