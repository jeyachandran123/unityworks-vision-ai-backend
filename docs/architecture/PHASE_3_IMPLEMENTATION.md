# Phase 3 — Live CCTV Runtime

**UnityWorks Vision AI · 2026-08-21**

## Result: **SOFTWARE PASS — REAL CCTV BLOCKED**

The runtime is complete and tested. **TCP 554 at the restaurant is still
filtered**, re-measured today, so no frame has ever arrived from a real camera
and `streaming` is `false` everywhere it is reported.

```
Backend    3,091 tests · 0 failures · 0 errors · 9 skipped · 143.8 s   (+69)
Frontend      92 tests · 0 failures                                    (+13)
typecheck clean · lint clean · build 771 ms
```

---

## 19. Real CCTV connectivity — measured today, layer by layer

| # | layer | result | detail |
|---|---|---|---|
| 1 | **DNS** | ✅ PASS | `gayatri.freemyip.com` → `203.118.57.154` |
| 2 | **TCP 554** | ❌ **TIMEOUT** | 6.0 s via hostname **and** via raw IP |
| 3 | RTSP handshake | — not reached | no TCP session to speak RTSP over |
| 4 | Authentication | — not reached | |
| 5 | Channel path | — not reached | |
| 6 | Decoder | — not reached | |
| 7 | **Frames received** | **0** | |

Same host, other ports: **80 OPEN (0.0 s) · 443 OPEN (0.0 s) · 9001 OPEN (0.1 s)**.

The DVR is online and answering. Only 554 hangs — *filtered*, not refused, which
is a firewall or missing port-forward signature rather than DVR
misconfiguration. Identical to the 2026-08-19 measurement.

**No workaround was implemented.** No 9001 tunnel, no ONVIF, no SmartPSS
automation, no HTTP snapshot polling. RTSP has not been shown unavailable; it has
been shown unreachable from here, and those need different fixes.

**`streaming` is `false`, and no camera result is claimed.**

---

## 1. Live source architecture

```
FrameSource (abstract)
├── SyntheticFrameSource   generated, paced      REPLAY
├── ReplayFrameSource      recorded file, paced  REPLAY
└── LiveRtspSource         RTSP over TCP         LIVE
```

One base class owns the state machine, the transition log and the counters, so
no source can change state without being observed and none can invent its own
reporting.

**States:** `CREATED → CONNECTING → RUNNING → STOPPING → STOPPED`, with
`RECONNECTING` between running and connecting, and `ERROR` terminal. Every
transition records a timestamp and a reason; the log is bounded at 200 entries so
a camera reconnecting all night cannot outgrow the frames it failed to deliver.

**A replay is never labelled live.** `kind` is `REPLAY` or `LIVE`, it appears in
every status payload, and a frontend test asserts a replay row never renders as
live.

## 2. Session architecture

```
FrameSource ──► sampler ──► bounded queue ──► session loop ──► handler
   (replay or live — identical from the sampler onward)
```

`VisionSession` owns the loop. `LiveSession` and `ReplaySession` differ only in
what they *declare* — `seekable`, `bounded` — not in how a frame is processed.

**Producer and consumer are separate tasks.** A slow model call must not stall
the decoder, because a stalled decoder becomes a dropped TCP connection and then
a camera outage — a processing problem promoted into a connection problem.

`pause()` is meaningful for live: the stream stays connected (reconnecting would
cost more than the pause) and arriving frames are dropped by the bounded queue,
so resuming does not deliver a backlog of stale video.

## 3. Source / decoder boundaries

Decoding is PyAV, resolved lazily inside each source. The module imports on a
host with no codec; a replay without PyAV raises a **capability gap** rather than
yielding nothing, because "no decoder" and "no frames" are different facts and
only one is a configuration error.

`LiveRtspSource` accepts an injected `opener`, which is how connect, reconnect,
authentication failure, decoder failure and credential redaction are all tested
without a network.

## 4. Backpressure policy — **drop-oldest**

When the source outruns the pipeline, the queue evicts the **oldest** frame and
keeps the newest.

For live safety monitoring the choice is not close. An operator asks "is that
person wearing a hairnet *now*"; a frame from twelve seconds ago answers a
question nobody is asking. And blocking the producer to preserve old frames makes
the stream fall further behind with every missed deadline, so the displayed world
drifts from reality without ever saying so.

Three properties hold while it drops, each asserted by a test:

1. **Timestamps are never reordered** — eviction is from the front.
2. **Frames are never fabricated or duplicated.**
3. **Every drop is counted**, and `sampled_out` is counted separately from
   `queue_full`. A single "dropped" number would make a healthy 25 fps camera
   look identical to an overloaded one.

## 5. Queue and buffer limits

| bound | value | why |
|---|---|---|
| frames queued per camera | **8** (`CCTV_QUEUE_CAPACITY`) | ~2 s of slack at 4 fps — enough to ride out a GC pause or a slow model call, short enough that a frame reaching the pipeline is still worth looking at |
| transition log | 200 entries, trimmed to 100 | bounded over months |
| reconnect delay | 1 s → 60 s | capped |
| reconnect attempts | 0 = indefinite, delay still capped | one attempt a minute, not a spin loop |

`LiveFrameQueue(0)` raises. **There is no value that means unlimited**, and a test
asserts it.

## 6. Timestamp model

Every frame carries `captured_at_ns` **and** `received_at_ns`.

Freshness ages against **capture** time. Stamping arrival would make every
observation look newer than it is, and a stale answer would present as a current
one. `observed_at` is not reset because the source became live.

The sampler also judges on capture time: a burst of buffered frames after a
network hiccup would otherwise all pass the gate at once, spending a second of
model budget on a second of already-stale video.

A test drives a source whose simulated capture clock advances 40 ms per frame
while real elapsed time advances by microseconds, and asserts the two spans
differ — they are independent clocks.

## 7. Camera identity

`camera_id` comes from configuration and **is never inferred from a frame**. The
session refuses to construct if its source names a different camera, because a
mismatch silently merges two kitchens downstream — `camera_id` is the partition
key for tracking, the registry and the observation log.

`epoch` increments on every reconnect and every replay loop, so a sequence never
appears to continue across a gap it did not survive and tracking cannot associate
across the seam.

## 8. Credential architecture

```
camera row      credential_ref = "env:CCTV_PASSWORD"     ← a reference
                       ↓
SecretProvider  env: | file: | literal:                  ← application boundary
                       ↓
dial_uri(password)  used once, never stored
redacted_uri()      rtsp://***:***@host:554/...          ← everything else
```

`app/vision/secrets.py` is the minimum `SecretProviderPort` implementation, and
it lives in the application deliberately: **Vision OS receives resolved
configuration and never learns how a secret is fetched.**

`_redact()` scrubs the live password out of every exception message, because
decoder libraries habitually quote the URL they failed to open. Tested by raising
an `OSError` containing the dial URL and asserting the secret reaches neither
`last_error` nor the wire payload.

A `literal:` reference contains its own secret, so error messages show only the
scheme. `file:` strips the trailing newline `echo` adds — otherwise a correct
password fails authentication in a way that looks exactly like a wrong one.

**A missing credential is never dialled.** Retrying a blank password counts
toward DVR account lockout.

## 9. Reconnection

Exponential backoff 1 s → 60 s, capped, with the exponent clamped at 32 — `2.0 **
9998` raised `OverflowError` in Phase 7A and turned a patient reconnect into a
crash.

`RECONNECTING` is its own state, distinct from `ERROR`: one is expected to
recover and should not page anyone, the other needs a human.

**Bad credentials and unknown stream paths are not retried at all.** A test
asserts exactly one attempt.

## 10. Shutdown

`stop_all()` cancels both tasks per session, closes and clears the queue, closes
the source and drops the resolved password. Sessions stop concurrently — a dozen
cameras each waiting out a backoff would otherwise make shutdown take minutes.

Tested: after shutdown, producer and consumer are `None`, queue depth is 0, and
the source state is terminal. Also tested under sustained backpressure, and for
idempotence.

## 11. Vision OS integration

The runtime accepts an `on_frame` handler and drives it per admitted frame. The
composition root sets it; absent one, the runtime still proves source, session,
sampling and backpressure behaviour — which is what the development path
exercises today.

**No second pipeline exists.** Replay and live enter the same session, the same
queue, the same sampler.

## 12–13. VLM integration and M9 → M7 — **not completed**

Reported plainly because §15 asked for it and it is not done.

The understanding layer is **still not bound** in the application composition
root. `VisionRuntime.assemble()` builds the platform and the registry layer with
the shared `AttributeRegistry`, and `understanding` remains `None`.

What **is** preserved and verified:

- `assert_shared_attribute_registry()` runs at assembly and checks by **object
  identity**, not equality.
- `test_shared_attribute_registry.py` (Phase 1) still passes: M7 holds the
  canonical instance, a second registry raises, and the neutrality gate refuses
  verdict-shaped keys.
- All 2,809 platform tests pass, including the 29 trigger tests covering
  `FRESH_ENOUGH`, `ATTRIBUTE_STALE` and the rest.

What is **not** demonstrated: an end-to-end M9 → M7 write-back on a live frame.
That needs the understanding layer bound to a real crop path, and binding it
without a source to feed it would have produced a composition nothing exercises.
It is the first item of Phase 4.

## 14. Freshness — preserved, not re-proven end to end

`validity_ms` is unchanged: `head_covering` 120 000, `hand_covering` 60 000. No
window was tuned; `test_freshness_regression.py` asserts both values and all ten
trigger reasons and eight skip reasons still exist.

## 15. Compliance semantics

Unchanged and still enforced. The frontend's `resolveState` keeps `none` →
ABSENT, `not_visible` → NOT_VISIBLE, missing → UNKNOWN, and only ABSENT may
become a violation. 24 semantics tests plus the fixture smoke test.

## 16. WebSocket live state

The handshake is unchanged: connect → `authenticate` frame → `ready`. The token
is still never in the URL.

**`streaming` is now derived from the runtime**:

```python
session.streaming = state is RUNNING and source.frames_produced > 0 and source.is_producing
```

An open socket has never been sufficient. `ready` carries `streaming`, the
visible sessions and the runtime summary; the heartbeat re-reports `streaming`
and per-camera health, so a stream starting or stopping between frames reaches
the client without a new event type.

**No fabricated events.** No detection, frame or compliance event is emitted,
because none is backed by real platform data yet.

Session summaries over the socket carry no URI and no credential reference —
those are DevTools, behind its own permission.

## 17. Frontend changes

Minimal, and only to consume real state:

| surface | change |
|---|---|
| `/api/v1/status` types | gained `cameras` and `live_runtime` |
| Dashboard | "Cameras online" is real; `—` when none is configured |
| Live Monitoring | a real camera wall from real health — **and no `<img>` anywhere**, asserted by a test |
| DevTools → Sources | was a placeholder, now reads `GET /devtools/live` |
| test harness | new `cameras` / `runtime` / `live` stubs |

The Sources screen shows queue depth against capacity, drops split into
sampled-out and queue-full, reconnects, epoch and the **redacted** URI.

`cameras` left `not_yet_reported` — real camera health is reported now, so
naming it as unreported would itself be the inaccuracy.

## 18. Test results

```
BACKEND   3,091 · 0 failures · 0 errors · 9 skipped · 143.8 s
  vision_os   2,809      unchanged, none weakened
  compliance     71      unchanged
  app           211      +69 this phase

FRONTEND     92 · 0 failures · 5 files
  live.test.tsx  13      new
```

New backend coverage: bounded queue and drop-oldest ordering · sampler ·
source lifecycle and transitions · camera health derivation · secret provider ·
credential redaction · reconnect and non-retry · session processing · **the
backpressure contract** · streaming state · **multi-session isolation** ·
autostart safety.

The backpressure test asserts memory stays bounded, drops increase, sequence
never goes backwards, no frame is processed twice, and gaps prove old frames were
dropped rather than queued — not merely that the queue eventually empties.

Multi-session runs three concurrent sessions and asserts no crossover of frames,
queues, tenants or camera ids, plus that an empty camera scope grants nothing.

**Two Phase 1/2 tests changed**, both because the behaviour they asserted is now
correctly obsolete: `cameras` left `not_yet_reported` on the backend and in the
frontend stub. No test was weakened.

## 20. Performance observations

| measure | value |
|---|---|
| queue capacity | 8 frames/camera |
| frame at 1080p BGR24 | ~6.2 MB → **~50 MB/camera** bounded |
| analysis rate | 4 fps default, independent of camera fps |
| sampling reduction at 25 fps | 84% of frames never reach the pipeline |
| DevTools chunk | 28.2 kB (7.9 kB gzip), still lazy |
| build | 771 ms |

**Not measured:** CPU under real decode, VLM concurrency, per-camera limits at
scale. All need a real stream, and estimating them would be inventing numbers.
The bound that matters — memory in the frame path — is enforced structurally
rather than estimated.

## 21. Security verification

| check | result |
|---|---|
| credentials in source | ✅ none — references only |
| credentials in logs | ✅ `_redact()` on every emitted string |
| credentials in exceptions | ✅ tested against a URL-quoting decoder error |
| credentials in API responses | ✅ redacted URI only |
| credentials in metrics | ✅ labelled by `camera`, never URI |
| tenant scoping | ✅ sessions filtered by tenant; other tenants get `NOT_FOUND`, not `FORBIDDEN` |
| camera scoping | ✅ empty tuple grants nothing |
| `VIEW_LIVE` for the socket | ✅ 4403 without it |
| evidence separate | ✅ unchanged |
| DevTools separate | ✅ unchanged |
| no autostart | ✅ three deliberate acts required |

## 22. Known limitations

1. **No real camera has ever connected.** TCP 554 filtered.
2. **The understanding layer is not bound** — §12–13 above. First item of Phase 4.
3. **The Dahua path is unverified.** `/cam/realmonitor?channel=1&subtype=1` is
   the documented default for this family and no handshake has confirmed it.
4. **No frames endpoint**, so Frame-by-Frame stays an honest placeholder naming
   the route it needs. A fabricated endpoint would have been worse.
5. **Evidence still in memory.** Nothing survives a restart.
6. **Single-node only.** Sessions are process state.
7. **Replay decoding needs PyAV.** Reported as a capability gap.
8. **No compliance events over the socket.** Nothing produces them yet.

## 23. Phase 4 prerequisites

**External, blocking live:** forward TCP 554 at the restaurant, then re-run the
ladder. First genuinely new information will be at layer 4 (authentication) and
layer 5 (whether the Dahua path is right for this XVR).

**Backend, in order:**

1. **Bind the understanding layer** — the crop path, provider selection, quality
   gate, and the M9 → M7 write-back end to end. Everything else waits on this.
2. Durable evidence store with a retention sweeper.
3. Camera configuration in the database, replacing `CCTV_CHANNELS`.
4. Incident persistence, with the finding frozen at creation.
5. A frames endpoint, once there are frames.

**The invariant to carry forward:** `streaming` is derived from a genuine frame
having arrived, in exactly one place. Every surface reads it and none asserts it.
The day a camera does connect, that is what will make the green light mean
something.
