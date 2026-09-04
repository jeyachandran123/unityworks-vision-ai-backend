# Live Camera / RTSP Input Recovery & End-to-End Perception Validation

Forensic trace of the live camera input path for cam-11…14.

| | |
|---|---|
| Repository | `unityworks-vision-ai-backend` |
| Branch | `feat/unityworks-vision-os-prod-hardining` |
| HEAD | `6992489` |
| Date | 2026-09-03, 15:10–15:45 IST |
| Code changed this phase | **none** |
| Tests | **3,982 passing / 1 pre-existing failure — unchanged** |

Every claim below is labelled **PROVEN**, **LIKELY** or **UNPROVEN**.

---

## 1. Executive verdict

## **OUTCOME B — EXTERNAL BLOCKER PROVEN**

The repository, the runtime and the credentials are all correct and working. The
backend is doing everything it should: it loads all four cameras, schedules them,
creates sessions, resolves credentials, builds correct RTSP URLs and **is
continuously dialling the DVR right now**. The connection never completes.

> **PROVEN — the exact boundary:** TCP SYN packets to `203.118.57.154:554` are
> **silently dropped**. Not refused, not reset — dropped. The host answers
> instantly on 80 and 443 and never on 554, 8554, 37777 or 37778.

> **PROVEN — and this changes the diagnosis from the previous phase:** the host at
> `gayatri.freemyip.com` is **not the DVR**. Its web interface identifies itself
> as **`EdgeOS`** — a Ubiquiti EdgeRouter. We are reaching the *router's* WAN
> interface. The DVR sits on the LAN behind it, and RTSP must be port-forwarded
> to be reachable. It is not.

The previous phase reported "the DVR RTSP port is unreachable". That was
correct but incomplete, and the difference matters: the remedy is **a port-forward
rule on the EdgeRouter**, not a DVR service restart.

### Failure classification

Every category from the brief, decided on evidence:

| | Category | Verdict |
|---|---|---|
| A | Camera configuration | ✅ **ELIMINATED** — all four rows correct (§3) |
| B | Database state | ✅ **ELIMINATED** — enabled + analysis_enabled + fps 3.0 |
| C | Runtime scheduling | ✅ **ELIMINATED** — `FEATURE_LIVE_CCTV=True`, all four scheduled |
| D | Session startup | ✅ **ELIMINATED** — sessions exist and are dialling (§4) |
| E | RTSP URL / path | ✅ **ELIMINATED** — standard Dahua path, well-formed |
| F | Authentication | ✅ **ELIMINATED** — credential resolves (9 chars); never presented, TCP never opens |
| G | Transport (TCP/UDP) | ✅ **ELIMINATED** — TCP forced, correct for NAT |
| H | Wrong RTSP port | ✅ **ELIMINATED** — 554, 8554, 37777, 37778 all dropped identically |
| I | DVR RTSP service disabled | ⚠️ **UNPROVEN and untestable** — the DVR is not reachable at all, so its service state cannot be observed from here |
| **J** | **Firewall / router / NAT** | 🔴 **PROVEN — this is the failure** |
| K | Backend capture implementation | ✅ **ELIMINATED** — PyAV path correct; independent test reproduces the same TCP failure |
| L | Codec incompatibility | ✅ **ELIMINATED** — no media ever arrives to decode |
| M | Frame capture after connect | ✅ **ELIMINATED** — no connection is ever established |

---

## 2. Exact failure boundary

```
Camera / DVR              ← unreachable, behind the router
        ↓
RTSP connection           ← 🔴 STOPS HERE. TCP SYN_SENT, no SYN-ACK, ever
        ↓
Video capture session     ← created, running, retrying
        ↓
Frames arriving           ← none
        … everything downstream unexercised
```

The boundary is **before RTSP negotiation** and **before authentication**. The
DVR never sees a packet from this machine.

---

## 3. Camera 11–14 configuration matrix

**PROVEN** — read directly from the database. No secret printed.

| Camera | Name | Org / Restaurant | enabled | analysis_enabled | analysis_fps | Redacted URI |
|---|---|---|---|---|---|---|
| cam-11 | Channel 11 | org-unityworks / gayatri-main | ✅ true | ✅ true | 3.0 | `rtsp://***:***@gayatri.freemyip.com:554/cam/realmonitor?channel=11&subtype=0` |
| cam-12 | Channel 12 | org-unityworks / gayatri-main | ✅ true | ✅ true | 3.0 | `…channel=12&subtype=0` |
| cam-13 | Channel 13 | org-unityworks / gayatri-main | ✅ true | ✅ true | 3.0 | `…channel=13&subtype=0` |
| cam-14 | Channel 14 | org-unityworks / gayatri-main | ✅ true | ✅ true | 3.0 | `…channel=14&subtype=0` |

Common: `host=gayatri.freemyip.com`, `rtsp_port=554`, `stream_type=main`
(`subtype=0`), `username=admin`, `credential_ref=env:CCTV_PASSWORD` (a pointer,
never a value). Path template is the stock Dahua
`/cam/realmonitor?channel={channel}&subtype={subtype}`.

Stage 1 answers:

1. **All four enabled?** ✅ Yes.
2. **All four analysis-enabled?** ✅ Yes.
3. **All four selected by runtime startup?** ✅ Yes — `FEATURE_LIVE_CCTV=True`, and
   `_start_cameras_from_database` filters on `row.host and row.analysis_enabled`,
   which all four satisfy.
4. **Does each produce a perception session?** ✅ Yes — sessions exist and are
   dialling (§4).
5. **Exact condition preventing startup?** **None inside the application.** The
   sessions start correctly; the transport fails.

---

## 4. Runtime startup matrix

**PROVEN by runtime observation**, not by reading source. The worker's outbound
sockets were sampled 41 times over 45 seconds:

```
samples taken: 41
--- non-loopback remote endpoints on PID 24840 ---
  203.118.57.154:554   SynSent   seen in 80 samples
```

Roughly two simultaneous `SYN_SENT` sockets to the DVR's RTSP port at any
instant, sustained across the whole window. **The backend is dialling
continuously and the reconnect loop is alive and working.**

> A correction to the previous phase: it reported "zero sockets to the DVR". That
> was a single point-in-time sample that fell between connection attempts. Sampling
> over time shows the opposite — the backend never stopped trying.

| Camera | 1. loaded | 2. should run | 3. session created | 4. connection attempted | 5. connected | 6. stream received | 7. ≥1 frame |
|---|---|---|---|---|---|---|---|
| cam-11 | ✅ | ✅ | ✅ | ✅ **continuously** | ❌ | ❌ | ❌ |
| cam-12 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| cam-13 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| cam-14 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

No diagnostic logging needed to be added: socket-level observation of the exact
worker PID was sufficient and is stronger evidence than a log line.

---

## 5. RTSP / network evidence

### Network layer — **PROVEN**

```
DNS   gayatri.freemyip.com -> 203.118.57.154        ✅ resolves, single A record

  port  result                  latency  note
    80  OPEN                      0.04s  HTTP
   443  OPEN                      0.04s  HTTPS
   554  TIMEOUT (dropped)         8.01s  RTSP  <-- CONFIGURED
  8554  TIMEOUT (dropped)         8.00s  RTSP alternate
 37777  TIMEOUT (dropped)         8.01s  Dahua native TCP
 37778  TIMEOUT (dropped)         8.00s  Dahua native control
```

Repeated at 12 s timeout, three consecutive attempts on 554 — all timed out.

**The distinction that matters:** a *stopped service on a reachable host* returns
TCP RST → `ConnectionRefusedError`, immediately. What we get is **silence**. Every
port except 80 and 443 behaves identically. That is a packet filter, not a dead
service.

### What is actually answering — **PROVEN, and the key discovery**

```
GET http://203.118.57.154/    -> 301 -> https://203.118.57.154:443/
GET https://203.118.57.154/   -> 200 OK   Server: Server
                                 <title>EdgeOS</title>
```

**`EdgeOS` is Ubiquiti EdgeRouter firmware.** The public endpoint is the router,
not the DVR. So:

* the DDNS record is healthy and current — **not** a stale-DNS problem;
* the site's internet connection is up;
* the router forwards **80 and 443** to itself, and nothing else inward;
* the DVR is on the LAN, invisible from here.

> **Incidental security finding (LIKELY, worth acting on):** the EdgeRouter's
> management interface is exposed to the public internet on 80/443. That is a
> router-administration concern outside this repository, but it is worth closing
> regardless of the RTSP issue.

### RTSP service layer — **UNPROVEN, and untestable from here**

Whether the DVR's own RTSP service is running, on which port, and with which
paths, **cannot be determined** while the router drops the packets. No claim is
made. No credentials or paths were brute-forced.

---

## 6. Independent stream capture evidence

**PROVEN.** Run with the backend's own stack (PyAV 18.1.0, the same
`_open_with_pyav` options: `rtsp_transport=tcp`, 10 s timeout), against the exact
configured URL and the real resolved credential:

```
credential resolved : YES (9 characters)      ← category F eliminated
dialling            : rtsp://***:***@gayatri.freemyip.com:554/cam/realmonitor?channel=11&subtype=0

RESULT after 10.0s
  exception : ExitError  (FFmpeg interrupt callback — the configured timeout firing)
  CLASS     : 1 — CANNOT CONNECT TO RTSP HOST (no TCP session established)
```

Against the six required outcome classes:

| # | Outcome | Result |
|---|---|---|
| **1** | **Cannot connect to RTSP host** | 🔴 **THIS ONE** |
| 2 | TCP connects, RTSP negotiation fails | not reached |
| 3 | Authentication fails | not reached — credential never presented |
| 4 | Negotiation succeeds, no media | not reached |
| 5 | Media arrives, codec cannot decode | not reached |
| 6 | Frames decode successfully | not reached |

This is **not** "codec unsupported" and **not** "camera unreachable due to bad
credentials". It is a TCP-layer failure, and it independently reproduces what the
backend experiences.

---

## 7. Backend capture-layer evidence

**PROVEN — no defect found.** The capture layer was audited because Stage 5
requires it, and it is correct:

* `_produce` runs a bounded exponential-backoff reconnect loop; failures are
  recorded and the epoch increments so tracking never associates across a gap.
* `_open_with_pyav` forces `rtsp_transport=tcp` — the right choice behind NAT —
  and passes **both** `timeout` and `stimeout`, because FFmpeg renamed the option
  and the older spelling is silently ignored on libavformat 62. The module
  documents having measured this.
* Auth and not-found errors are classified and **not** retried, so a rejected
  password cannot drive DVR account lockout.
* Frame timestamps come from stream PTS, not arrival time.
* `thread_type="NONE"`, `thread_count=1` per container — deliberate, so sixteen
  cameras do not multiply into hundreds of decode threads.

**A false alarm I raised and then disproved.** My ad-hoc capture script printed
the credential, because PyAV quotes the failing URL in its exception and my
script's naive `.replace(password, "***")` missed the URL-encoded form. I checked
whether production had the same hole. **It does not** — `LiveRtspSource._redact`
scrubs *both* the raw value and `quote(password, safe="")`, and every error string
that reaches a log or a status endpoint goes through it. The leak was in my
throwaway script, not in the backend. Recorded here so the non-finding is not
mistaken for a finding.

---

## 8. Exact root cause

> **PROVEN:** The Ubiquiti EdgeRouter at `203.118.57.154` is **not forwarding
> inbound TCP 554 to the DVR on the LAN**. SYN packets are silently dropped.
>
> Ports 80 and 443 terminate on the router's own management UI. No inbound port
> reaches the DVR.

**LIKELY causes**, in order — all outside this repository, none distinguishable
from here:

1. The RTSP port-forward / destination-NAT rule was removed, disabled, or its
   target LAN IP changed (a DVR DHCP lease change would do this silently).
2. A firewall rule ahead of the forward is dropping WAN→LAN 554.
3. The ISP began blocking or CGNAT-ing inbound 554 (80/443 still mapped).

It worked earlier the same day: `data/observations/cam-*.jsonl` were last
appended at **10:51:24**, which requires a live RTSP session. Something changed
between 10:51 and 15:10.

---

## 9. Exact repair

**None. No code was changed, and none should be.**

The stop condition applies: the failure is outside the repository. Every
application-side hypothesis was tested and eliminated (§1). Changing backend code
now would be compensating for a network fault, which the brief forbids and which
would leave a real defect masked.

### Minimum action required outside the repository

On the EdgeRouter at `gayatri.freemyip.com`:

1. Confirm the DVR's current LAN IP (check for a DHCP lease change; give it a
   static mapping).
2. Restore or correct the port-forward: **WAN TCP 554 → `<DVR LAN IP>:554`**.
3. Confirm no firewall rule ahead of it drops WAN→LAN 554.
4. Separately, and regardless: consider removing public exposure of the router's
   own management UI on 80/443.

### Exact verification command, to run after the external fix

```bash
cd unityworks-vision-ai-backend

# 1. Does TCP now open?  Expect "OPEN", not TimeoutError.
./.venv/Scripts/python.exe -c "
import socket,time
s=socket.socket(); s.settimeout(8); t=time.time()
try: s.connect(('gayatri.freemyip.com',554)); print(f'554 OPEN ({time.time()-t:.2f}s)')
except Exception as e: print('554', type(e).__name__)
finally: s.close()"

# 2. Do real frames decode?  Expect "CLASS: 6 - FRAMES DECODE SUCCESSFULLY".
#    (capture_test.py from this phase, or equivalently:)
#    av.open(dial_uri, options={'rtsp_transport':'tcp','timeout':'10000000'})

# 3. Does the backend pick them up?  Expect Established, not SynSent.
powershell -NoProfile -Command "Get-NetTCPConnection | Where-Object { \$_.RemotePort -eq 554 } | Select-Object OwningProcess,State"

# 4. Are observations growing?  Expect mtimes advancing.
ls -l data/observations/cam-*.jsonl
```

---

## 10. What was deliberately not changed

* Detector, VLM, `nvidia_vl.py`, confidence thresholds, policy severities,
  tracker association thresholds — untouched.
* Incident deduplication — untouched; nothing suppressed.
* The five perception repairs — untouched and intact.
* No test weakened or deleted.
* **No process started, and none killed.** Every PID was recorded and observed
  only. Worker `24840` is still the process it was at the start (started
  14:38:15) and is still serving.
* The seven inert `kit-*.jsonl` files — left in place.
* **Readiness semantics — reviewed but not modified.** See §17.

---

## 11–16. Live validation — **NOT REACHED**

| # | Item | Status |
|---|---|---|
| 11 | End-to-end real-frame validation | ⛔ **UNPROVEN** — no frames |
| 12 | Tracking continuity evidence | ⛔ **UNPROVEN** — no frames |
| 13 | Registry identity evidence | ⛔ **UNPROVEN** — no frames |
| 14 | Observation publication | ✅ **PROVEN bound** in the prior phase (synthesis + exposure bind against the real log, writing and deleting nothing). Not exercised with data. |
| 15 | Compliance / incident evidence | ⛔ **UNPROVEN** — 1,990 incidents, latest `05:06:41Z`, ratio still 1.0000 because nothing new has been measured |
| 16 | Alert visibility | ⛔ **UNPROVEN** — no new findings to surface |

**No violation was manufactured, and no metric was estimated.** The ratio
`incidents / distinct object_ids` remains 1.0000 solely because no new incident
exists. That is not evidence the repair failed; it is evidence nothing has been
measured.

---

## 17. Health / readiness semantics

**Reviewed as instructed. A real weakness confirmed — change escalated, not made.**

`app/api/routes.py:51`:

```python
return {"ready": ready, "database": db_ok, "cache": cache_ok,
        "vision_os": vision.assembled}
```

`vision_os` reports **assembly only**. Right now it is `true` while:

* zero cameras are connected,
* zero frames have been processed for four and a half hours,
* zero observations have been published.

That is exactly the misleading green boolean the brief warns about, and it is why
this failure needed socket forensics to find rather than being visible on a
dashboard.

**Recommended — narrowly scoped, additive, backward compatible:**

```python
composition = getattr(vision, "_composition", None)
publishing  = composition is not None and composition.synthesis is not None \
                                      and composition.exposure  is not None
camera_input = any(s.stats.frames_received > 0 for s in live.sessions)

return {
    "ready": ready,                    # unchanged: still db + cache only
    "database": db_ok,
    "cache": cache_ok,
    "vision_os": vision.assembled,     # unchanged, backward compatible
    "perception_publishing": publishing,   # NEW
    "camera_input": camera_input,          # NEW
}
```

Booleans only, matching the endpoint's existing "never detail" contract — which
matters because this route is **unauthenticated**, so no camera names or counts
may appear. `ready` gating is unchanged, so no orchestrator behaviour changes, and
a temporarily offline camera does not make the service unready.

**Not applied**, for three reasons: it touches a public unauthenticated API; the
external blocker means `camera_input` could only be observed as `false` and never
proven to flip `true`; and this phase's stop condition directs me to stop at the
external boundary. **Escalated for your approval.**

---

## 18. Test counts

```
before this phase : 3,982 passed, 1 failed
after  this phase : 3,982 passed, 1 failed
delta             : 0   (no code changed, no test added)
```

The single failure is unchanged and pre-existing:
`test_ninety_b_configuration.py::test_no_production_module_names_the_model`
(`nvidia_vl.py:73`). Not fixed, not counted as a regression.

**No regression test was added.** The discovered failure is a router
port-forward, which has no in-repository behaviour to pin. Writing a test that
asserts "TCP 554 is reachable" would be a network monitor, not a unit test, and
would fail in CI for reasons unrelated to the code. §9 gives the verification
commands instead.

---

## 19. Remaining unknowns

| Question | Status |
|---|---|
| Is the DVR powered on and healthy? | **UNPROVEN** — invisible behind the router |
| Is the DVR's RTSP service enabled, and on which port? | **UNPROVEN** — untestable from here |
| Are the credentials accepted by the DVR? | **UNPROVEN** — resolve correctly, never presented |
| Does channel/subtype mapping match this DVR? | **UNPROVEN** — standard Dahua, plausible, unverified |
| Is the stream codec decodable by PyAV? | **UNPROVEN** — no media has arrived |
| Does identity continuity hold on real footage? | **UNPROVEN** — the central open question |
| Does `created=False` now occur? | **UNPROVEN** — the measurement that matters |
| Why did RTSP stop between 10:51 and 15:10? | **UNPROVEN** — likely a router or ISP change |

---

## 20. Escalated decisions

1. **Restore the WAN→LAN TCP 554 port-forward on the EdgeRouter.** The single
   blocking action. Everything else waits on it.
2. **Close public access to the router's management UI** on 80/443 — incidental,
   security-relevant, unrelated to RTSP.
3. **Approve the readiness change in §17**, or decline it deliberately.
4. **Restart the backend after RTSP returns**, so cameras start at the new 3 fps.
   Consider dropping `--reload`: every file save tears down all four sessions.
5. **Decide the validation window.** Roughly an hour of footage with people in
   frame, then re-measure `incidents / distinct object_ids` — 1.0000 before;
   anything above 1.0 means deduplication is finally being exercised.

---

## Final answers

| Question | Answer |
|---|---|
| Root cause cam-11…14 not streaming | **PROVEN** — EdgeRouter at `203.118.57.154` silently drops inbound TCP 554; the DVR is unreachable behind it |
| Did the backend attempt sessions for all four? | **PROVEN — yes**, all four, continuously (sustained `SYN_SENT`) |
| Failure before or after RTSP connection? | **Before.** Before negotiation and before authentication |
| Did any independent capture succeed? | **No** — class 1, cannot connect; reproduced with the backend's own PyAV stack |
| What repair was made? | **None.** External boundary; the brief forbids compensating in code |
| Do real frames reach the detector? | **No** |
| Is track continuity observed? | **Not measurable** |
| Is fragmentation reduced? | **Not measurable on live data** — proven in tests only |
| Are observations growing? | **No** — frozen at 10:51:24 |
| Is compliance receiving subjects? | **No** — zero |
| Was a real incident generated? | **No** |
| Tests before / after | **3,982 / 3,982**, 1 pre-existing failure, unchanged |
