# Phase 7B — Real CCTV Connection

**Date:** 2026-08-19
**Status: BLOCKED at network reachability. RTSP port 554 is not reachable, and the failure is two
layers below authentication — the password could not have changed this outcome.**

**No frames were received. Milestone B is not claimed.**

---

## 1. Phase status

| | |
|---|---|
| completed before | Vision OS pipeline, verification, compliance, M9→M7 write-back, freshness (6.9), Phase 7A source adapter |
| this phase | 7B — connect one real Dahua camera |
| result | **failure isolated to TCP 554** |
| next | unblock port 554, then re-run this phase unchanged |

## 2. Device

| | |
|---|---|
| site / model | Gayatri Restaurant · DH-XVR5116HS-I3 |
| host | `gayatri.freemyip.com` → **203.118.57.154** |
| channel under test | 1 |
| stream subtype | 1 (sub) — never reached |
| URL pattern | `rtsp://admin:***@gayatri.freemyip.com:554/cam/realmonitor?channel=1&subtype=1` |

### ⚠️ Host spelling

This brief spells the host **`gayatri.freeymip.com`** — that does **not** resolve. The working
spelling is **`gayatri.freemyip.com`** (the provider is *freemyip.com*). Phase 7A used the correct
one. Worth correcting wherever it is written down, because it will otherwise surface later as a
confusing DNS failure.

## 3. Diagnostic ladder (§11 order)

| # | layer | result | detail |
|---|---|---|---|
| 1 | **DNS** | ✅ **PASS** | `gayatri.freemyip.com` → 203.118.57.154 |
| 2 | **TCP 554** | ❌ **FAIL** | `TimeoutError` after 6 s — via hostname **and** via raw IP |
| 3 | RTSP handshake | — not reached | no TCP session to speak RTSP over |
| 4 | Authentication | — not reached | |
| 5 | Stream path | — not reached | |
| 6 | Channel number | — not reached | |
| 7 | Decoder | — not reached | |
| 8 | Stream subtype | — not reached | |

### Port scan of the same host

| port | purpose | result |
|---|---|---|
| 80 | HTTP | **OPEN** |
| 443 | HTTPS | **OPEN** |
| **554** | **RTSP** | **TIMEOUT** |
| 9000 | UDP-paired TCP | ConnectionRefused |
| 9001 | Dahua TCP | **OPEN** |

## 4. What this means

**The DVR is online and reachable.** Ports 80, 443 and 9001 answer immediately — that is the
device responding, so this is not a dead host, a wrong IP, or a down internet connection.

**Only 554 hangs.** The distinction matters: a *closed* port refuses immediately (as 9000 does,
in 2.2 s); a *filtered* port drops the packet and the client waits for timeout (as 554 does, at
the full 6 s, twice). That signature points at a firewall or missing port-forward rule, not at
DVR configuration.

Most likely, in order:

1. **Port 554 is not forwarded** on the restaurant router, while 80/443/9001 are. This is the
   common case — Dahua's own mobile apps use 9001, so people forward that and never notice 554.
2. **RTSP is disabled** on the DVR itself (Dahua: *Network → Port → RTSP*).
3. **ISP blocking** of 554 inbound.

## 5. Credentials

**Not available to this process.** `CCTV_USERNAME` and `CCTV_PASSWORD` are unset in the
environment and absent from every `.env` file — the brief's placeholder
`<real password from secure environment>` was never substituted with an actual value.

**This did not affect the result.** The connection fails at TCP, two layers below authentication.
Had the password been present, the outcome would have been byte-identical.

Per §1 and §10, nothing was hardcoded, printed, logged or committed, and the redaction machinery
from Phase 7A remains in place and asserted by tests.

## 6. Success criteria

| criterion | result |
|---|---|
| DNS resolves | ✅ |
| RTSP port 554 reachable | ❌ |
| Authentication succeeds | — not reached |
| Dahua path confirmed | — not reached |
| Channel 1 opens | — not reached |
| `LiveRtspSource` reaches RUNNING | — not reached |
| Real frames received | ❌ **0 frames** |
| Frame dimensions / timestamps / sequence | — not reached |
| **No credentials exposed** | ✅ |
| **File mode unchanged** | ✅ 160 harness tests passing |
| **No Vision OS core modification** | ✅ |

## 7. What was *not* done, deliberately

Per §14, no fallback protocol was implemented. **No ONVIF, no HTTP snapshot polling, no tunnelling
over 9001.** RTSP has not been shown to be unavailable — it has been shown to be *unreachable from
here*, which is a different fact with a different fix.

`LiveRtspSource` was not exercised against the device, because doing so would only have produced
the same timeout wrapped in a `CameraState.ERROR`. That is worth confirming *after* the port opens,
not as a substitute for it.

## 8. Smallest corrective action

**Forward TCP 554 to the DVR on the restaurant router**, then confirm from outside the LAN:

```
# should connect rather than hang
Test-NetConnection gayatri.freemyip.com -Port 554
```

Also worth checking on the DVR: *Network → Port → RTSP Port* is 554 and RTSP is enabled.

If 554 cannot be opened for policy reasons, say so — the DVR's own port 9001 is reachable, and
that changes the design conversation. But that is a decision to take deliberately, not a
workaround to reach for because a port was closed.

## 9. Re-running this phase

Once 554 is open, nothing here needs rewriting. Set:

```
CCTV_HOST=gayatri.freemyip.com
CCTV_CHANNELS=1
CCTV_USERNAME=admin
CCTV_PASSWORD=<from secure environment>
VIDEO_SOURCE=rtsp
```

and re-run the same ladder. The first genuinely new information will be at layer 4
(authentication) and layer 5 (whether `/cam/realmonitor?channel=1&subtype=1` is the right path for
this XVR).

## 10. Phase 7C requirement — unchanged and still accurate

The session boundary identified in Phase 7A is where live frames must connect:

`Session.__init__` takes a **pre-decoded frame list** (`list[DecodedFrame]`), and
`assembly.build_stack(frames=...)` passes it to `ReplayFileSource`. A live camera has no such
list — it has a stream. Phase 7C is the work of feeding `LiveRtspSource` frames into that seam
continuously without creating a second session pipeline.

I would not design that against a stream I have never opened. Resolution, frame rate and timestamp
behaviour of this specific DVR all shape it, and all three are currently unknown.
