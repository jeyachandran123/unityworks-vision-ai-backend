# Cross-Repository CCTV & Perception Forensic Audit

Read-only investigation across every repository in the `atlas/` workspace.
**No code changed. No process started, stopped or killed. No database row modified.
Nothing deleted.**

| | |
|---|---|
| Date | 2026-09-03, ~15:45–16:15 IST |
| Workspace | `c:\Users\Jayachandran\ProjectsAndDocs\atlas` |
| Repositories found | **6**, not 3 |
| Evidence labels | **PROVEN** · **DISPROVEN** · **INFERRED** · **UNKNOWN** |

---

## 1. Executive summary

The audit found **two independent failures separated in time**, which is why the
"video was streaming but no alerts arrived" report looked contradictory. It is
not a contradiction — it is two different faults, and the first one explains the
observation the second one cannot.

| # | Failure | Window | Status |
|---|---|---|---|
| **1** | Synthesis unbound by the conformance-kit-poisoned observation log → **no observations, no incidents, but frames still flowing** | from ~10:52 IST until the 14:38 IST reboot | **repaired** (prior phase) |
| **2** | RTSP transport to the DVR unreachable → **no frames at all** | onset UNKNOWN, confirmed present from 15:10 IST | **open, external** |

**The key structural fact (PROVEN):** the Live Wall and the analysis pipeline
both consume the same `LiveRtspSource` against the same DVR. They cannot diverge
on transport — if RTSP works, both work; if it fails, both fail. But they *can*
diverge on **publication**: the wall needs only frames, while alerts additionally
need synthesis to be bound.

That is precisely the shape of the reported symptom:

> **Failure 1 produces exactly "live video on screen, and no alerts".**

So the most probable reading of "CCTV was streaming after 1:30 PM" is that the
Live Wall was genuinely showing live video from a worker whose synthesis had been
unbound since 10:52 — real frames, no observations, no incidents. **INFERRED, not
proven** — the wall keeps no durable record, and stream viewing is not audited, so
the exact moment RTSP stopped cannot be recovered.

**Answering the success criteria:**

1. **Where does the working CCTV stream come from?** `rtsp://…@gayatri.freemyip.com:554/cam/realmonitor?channel=N&subtype=0` — **the only CCTV source configured anywhere in any repository. PROVEN.**
2. **Which service consumes it?** Only `unityworks-vision-ai-backend`. **PROVEN.**
3. **How could video stream while alerts stopped?** Failure 1 above. **INFERRED (mechanism PROVEN, timing UNKNOWN).**
4. **Are cameras 11–14 the expected physical cameras?** Numerically aligned end-to-end and independently validated on 2026-08-24. **PROVEN for the mapping; UNPROVEN that channel 11 is the physical kitchen the operator means.**
5. **Is UnityWorks using the correct source?** **Yes. PROVEN** — byte-identical to the console's independently validated URL.
6. **Does Atlas use a different source?** **No. DISPROVEN** — same path template, and it defaults to *file* playback.
7. **First blocked layer?** **Layer 0 — camera input. PROVEN.**
8. **Is the remaining problem ML/perception?** **No. PROVEN** — the system is blocked before ML receives a single frame.

---

## 2. Repository inventory

The workspace holds **six** git repositories, of which only two are the live
UnityWorks system. Three belong to other products.

| Directory | Git remote | Branch | HEAD | Role |
|---|---|---|---|---|
| `unityworks-vision-ai-backend/` | `unityworks-vision-ai-backend.git` | `feat/unityworks-vision-os-prod-hardining` | `6992489` (2026-09-03) | 🟢 **LIVE** — perception backend |
| `unityworks-vision-ai-frontend/` | `-unityworks-vision-ai-frontend.git` | `feat/unityworks-vision-os-prod-hardening` | `ae0d177` (2026-09-03) | 🟢 **LIVE** — Live Wall UI |
| `vision_os_validation_console/` | `UnityWorks-vision-os-devtools.git` | `feat/…-prod-hardening` | `7dc04eb` (2026-08-27) | ⚪ devtools harness — **not running** |
| `backend/` | **`Atlas.git`** | `feat/meta-cognitive-architecture` | `bc46f7c` (2026-09-01) | ⚪ **different product** |
| `frontend/` | **`Atlas-frontend.git`** | `feat/meta-cognitive-architecture` | `a6bafbc` (2026-07-30) | ⚪ **different product**, 5 weeks stale |
| `vision_os_demo/` | `UnityWorks-vision-os-demo-app.git` | `main` | `57b9c92` (2026-08-17) | ⚪ demo, not running |

> **Correction to the brief (PROVEN).** The brief named three repositories and
> called the console "atlas/vision_validation_console". The directory is
> `vision_os_validation_console/` and its remote is
> **UnityWorks-vision-os-devtools**, not Atlas. The directories actually named
> `backend/` and `frontend/` are the **Atlas** product — a different codebase with
> a different remote. They are not part of the UnityWorks CCTV path.

### CCTV-touching components

| Repo | File | Responsibility | Input | Output | Touches video? |
|---|---|---|---|---|---|
| uw-backend | `app/vision/sources/rtsp.py` | `LiveRtspSource` — the **only** RTSP client | DVR RTSP | `LiveFrame` | ✅ direct |
| uw-backend | `app/vision/wall.py` | MJPEG wall, one `LiveRtspSource` per camera | DVR RTSP | JPEG frames | ✅ direct |
| uw-backend | `app/vision/manager.py`, `session.py` | analysis sessions, `FrameSampler` | `LiveFrame` | perception | ✅ indirect |
| uw-backend | `app/api/wall.py` | ticketed MJPEG endpoints | HTTP | `multipart/x-mixed-replace` | ✅ serves |
| uw-backend | `app/domain/cameras.py` | camera rows → `RtspCameraConfig` | Postgres | config | ⚪ config only |
| uw-frontend | `src/features/camera-wall.tsx` | Live Wall tiles | wall API | `<img>` | ✅ displays |
| uw-frontend | `src/shared/api/wall.ts` | wall API client | HTTP | typed state | ⚪ no URL knowledge |
| console | `harness/vosvc_harness/sources/cctv.py` | its own RTSP client | DVR RTSP | frames | ✅ direct, **idle** |
| console | `scripts/channel_scan.py`, `channel_decode.py` | channel probes | DVR RTSP | JSON reports | ✅ direct, **idle** |
| console | `harness/vosvc_harness/config.py` | `VIDEO_SOURCE` — **defaults to `file`** | env | config | ⚪ config only |
| Atlas `backend/`, `frontend/` | — | different product | — | — | ❌ not in path |

---

## 3. End-to-end data flow — the real one

```
┌──────────────────────────────────────────────────────────────────┐
│ DVR (Dahua, 16ch, H.265)   — on the LAN, address UNKNOWN         │
└───────────────┬──────────────────────────────────────────────────┘
                │  RTSP 554
┌───────────────▼──────────────────────────────────────────────────┐
│ Ubiquiti EdgeRouter  ·  gayatri.freemyip.com → 203.118.57.154     │
│   80/443 → router's own EdgeOS admin UI        (OPEN, 0.04 s)     │
│   554 / 8554 / 37777 / 37778 → SILENTLY DROPPED  🔴 BLOCKED       │
└───────────────┬──────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────┐
│ unityworks-vision-ai-backend   (uvicorn :8010, PID 24840)        │
│                                                                   │
│   LiveRtspSource ──┬─► CameraWall ──► /wall/…/stream.mjpg ──┐    │
│    (SYN_SENT)      │      (needs frames only)                │    │
│                    └─► VisionSession → FrameSampler          │    │
│                            → detection → tracking → registry │    │
│                            → cropping → VLM → synthesis      │    │
│                            → exposure → compliance → incident│    │
└──────────────────────────────────────────────────────────────┼────┘
                                                               │
┌──────────────────────────────────────────────────────────────▼────┐
│ unityworks-vision-ai-frontend  (vite dev)                         │
│   GET  /wall/cameras                → list + state                │
│   POST /wall/cameras/{id}/ticket    → short-lived signed ticket    │
│   <img src="…/stream.mjpg?ticket="> → MJPEG                        │
└───────────────────────────────────────────────────────────────────┘

Validation console:  same RTSP construction, VIDEO_SOURCE defaults to "file",
                     NOT RUNNING.   Atlas backend/frontend: unrelated product.
```

**The load-bearing observation (PROVEN):** wall and analysis share
`LiveRtspSource`. `app/vision/wall.py:300` constructs one per camera; the
analysis path constructs its own. **Same client, same URL, same transport.**

---

## 4. Camera / channel mapping

| Logical | DB `camera_key` | DB `name` | DB `channel` | RTSP path | Wall `channel` | Console scan | DVR ch |
|---|---|---|---|---|---|---|---|
| Camera 11 | `cam-11` | Channel 11 | 11 | `…channel=11&subtype=0` | 11 | ch 11 PASS | 11 |
| Camera 12 | `cam-12` | Channel 12 | 12 | `…channel=12&subtype=0` | 12 | ch 12 PASS | 12 |
| Camera 13 | `cam-13` | Channel 13 | 13 | `…channel=13&subtype=0` | 13 | ch 13 PASS | 13 |
| Camera 14 | `cam-14` | Channel 14 | 14 | `…channel=14&subtype=0` | 14 | ch 14 PASS | 14 |

**PROVEN:** the number is carried unchanged from database row → `RtspCameraConfig.channel`
→ Dahua path → wall API `channel` field. No repository renumbers, offsets or
remaps. The frontend never learns a URL at all — it receives an id, a channel and
a state.

**PROVEN by independent probe (2026-08-24):** all 16 channels answered
`rtsp_code: 200` and decoded real frames. Channels 11 and 13 at 1920×1080/15 fps,
channels 12 and 14 at 960×576/25 fps, all `looks_blank: false`, codec H265/hevc.

**UNKNOWN:** whether DVR channel 11 is physically the kitchen the operator
believes it is. Nothing in any repository can establish that; it needs someone to
look at a monitor. The wall API comment — *"The DVR channel this camera is wired
to. Verified, not assumed."* — records a human verification, not a machine one.

---

## 5. Configuration comparison

| Aspect | uw-backend | validation console | Atlas `backend/` |
|---|---|---|---|
| Source of truth | Postgres `cameras` table | environment variables | n/a |
| Host | `gayatri.freemyip.com` (DB) | `CCTV_HOST` env (unset) | n/a |
| Port | `554` (DB) | `rtsp_port` (default 554) | n/a |
| Path template | `/cam/realmonitor?channel={channel}&subtype={subtype}` | **identical** | n/a |
| Stream profile | `main` → `subtype=0` | `subtype` from channel spec | n/a |
| Username | `admin` (DB) | `CameraCredentials` | n/a |
| Credential | `env:CCTV_PASSWORD` (a **pointer**) | env | n/a |
| Default mode | live RTSP | **`VIDEO_SOURCE=file`** | n/a |
| Client | PyAV 18.1.0, `rtsp_transport=tcp` | its own `_open_rtsp` | n/a |
| Running? | ✅ yes | ❌ no | ❌ no |

### Is `203.118.57.154:554` the only configured source anywhere?

**PROVEN — yes.** A ripgrep sweep of the whole workspace (excluding
`node_modules`) for `203.118.57.154` returns **11 files, all documentation or
dataset reports** — never live configuration. `rtsp://` literals appear only in
docs, datasets and the console's scan output. The single live source is the
Postgres `cameras` table.

**DISPROVEN: hypotheses 1, 2, 3, 6, 8, 9, 14** — there is no second CCTV
integration, no alternative DVR, no relay, no proxy, no HLS/WebRTC layer and no
second pipeline anywhere in the workspace. No secret is printed in this report.

---

## 6. Atlas / validation console analysis

| Question | Answer | Label |
|---|---|---|
| Connects directly to CCTV? | Yes — its own `sources/cctv.py` | **PROVEN** |
| Displays live video? | Its harness serves sessions; not running | **PROVEN idle** |
| Consumes frames from another service? | No | **PROVEN** |
| Has camera validation tooling? | Yes — `channel_scan.py`, `channel_decode.py`, `rtsp_ladder.py`, `live_pipeline_probe.py` | **PROVEN** |
| Proxy / relay? | No | **PROVEN** |
| Knows the DVR LAN IP? | **No** — only `CCTV_HOST`, unset | **PROVEN** |
| Uses ONVIF? | No | **PROVEN** |
| Uses FFmpeg/PyAV? | Yes | **PROVEN** |
| Contains production camera URLs? | Only in scan output, and identical | **PROVEN** |
| Config absent from UnityWorks? | `VIDEO_SOURCE`, `VOSVC_MEDIA_ROOT`, `VOSVC_VISION_OS_ROOT` — harness-only | **PROVEN** |

**The console does not explain the streaming.** It is not running, it defaults to
file playback, and its media store (`vision_os_validation_console/media/`) holds
nine files, newest **2026-08-18**. The workspace `media/` folder likewise ends at
2026-08-18. **PROVEN: no video file anywhere in the workspace was written today.**

Its one genuinely valuable contribution is the 2026-08-24 evidence that this exact
URL, port and path decode real frames on all 16 channels.

> Note: the console imports Vision OS from `VOSVC_VISION_OS_ROOT`, defaulting to
> the **Atlas `backend/`** directory — a stale coupling to a different product. Not
> a live fault (nothing runs), but worth knowing before anyone starts it.

---

## 7. Frontend Live Wall analysis

**PROVEN** request → response → state chain:

```
useQuery ──► GET  /api/v1/wall/cameras            → WallCamera[]
                     { camera_id, channel, state, width, height,
                       frames_decoded, seconds_since_frame, last_error }
         ──► POST /api/v1/wall/cameras/{id}/ticket → short-lived signed ticket
         ──► <img src="{stream_path}?ticket=…">    → multipart/x-mixed-replace (MJPEG)
```

* **What makes the UI show CONNECTING:** the tile's phase is `connecting` until a
  frame actually decodes. `camera-wall.tsx:37` is explicit — *"a ticket that
  returned 200 proves only that a ticket was issued"*; only a decoded frame
  promotes a tile to `live`.
* **What makes it show "resolution unknown":** `camera.width` is falsy
  (`camera-wall.tsx:359`) — the server has never measured a frame for this camera.
* **Can the backend stream while the UI shows CONNECTING?** **No. DISPROVEN
  (hypothesis 13).** The state is server-derived from genuine frame arrival —
  the client cannot set it. Four tiles reading CONNECTING with unknown resolution
  means the **server has decoded zero frames** since the 14:38 boot.
* **Stale state / API mismatch?** **DISPROVEN.** The wall types match the backend
  payload, and the query re-fetches.
* **Is the browser opening video directly?** No — it never learns a host,
  credential or RTSP URL. Everything is proxied through the backend.

---

## 8. Backend camera pipeline

For each of cam-11…14 (**PROVEN**, from the prior phase's socket forensics and
re-confirmed here):

| Step | Result |
|---|---|
| 1. Loaded from database | ✅ yes |
| 2. `enabled` | ✅ true |
| 3. `analysis_enabled` | ✅ true |
| 4. Session created | ✅ yes (`FEATURE_LIVE_CCTV=True`) |
| 5. Frame source created | ✅ yes |
| 6. Connection attempted | ✅ **continuously** — `203.118.57.154:554 SynSent` in 80 of 41 samples |
| 7. Connection established | ❌ **no** |
| 8–18. frames → detection → tracking → registry → observation → synthesis → exposure → compliance → incident → alert | ⛔ never reached |

`analysis_fps` = 3.0 on all four. Composed tracker = `tracker.sort`. Synthesis and
exposure **bind successfully** against the real production log (prior phase) —
so layers 5–8 are healthy and merely starved.

---

## 9. Runtime & deployment topology

**PROVEN**, all on one Windows host, no containers for the application:

| Component | Where | PID | Started |
|---|---|---|---|
| uw-backend `uvicorn :8010 --reload` (launcher) | native | 30000 | 12:09:48 |
| ↳ reload **supervisor** | native | 11364 | 12:09:48 |
| ↳ **live worker** | native | **24840** | **14:38:15** |
| uw-frontend `npm run dev` + vite | native | 33332 / 33068 | 12:09:49 |
| vite :5277 | native | 20296 / 29904 | 02-09 |
| vite :5278 | native | 17920 / 14472 | 02-09 |
| pgAdmin | native | 36272 | 11:09:48 |
| stale scratchpad `x.py` | native | 19168 / 28840 | 09:49:38 |

Docker holds **no application container** — only `ai-coding-assistant` services
plus shared infrastructure: `aic_postgres` (5432) and `aic_redis` (6379), which the
backend connects to. **Port 8000 is `aic_api`, unrelated to Vision OS.**

**PROVEN: nothing from the validation console, Atlas `backend/` or Atlas
`frontend/` is running.** There is exactly one backend and one frontend, and both
correspond to the repository versions inspected. Two extra idle vite servers
(:5277, :5278) are leftovers from 02-09 — **INFERRED** to be stale dev servers, not
a second product.

---

## 10. Timeline (all IST)

| Time | Event | Label |
|---|---|---|
| 2026-08-24 08:30 | Console scans all 16 channels — **all PASS, frames decode** | **PROVEN** |
| 2026-09-03 09:49 | Stale scratchpad process starts | PROVEN |
| **10:02** | `camera.disabled` ×2 audited | **PROVEN** |
| **10:17** | Last `auth.login` | **PROVEN** |
| **10:35** | **Last evidence record captured** — last proof of a real frame | **PROVEN** |
| **10:36** | Last incident created | **PROVEN** |
| **10:51** | Last observation-log append (`cam-11…14.jsonl`) | **PROVEN** |
| **10:52** | Boot writes 7 `kit-*.jsonl` → **synthesis unbound** | **PROVEN** |
| 12:09 | Backend process tree restarted (pre-repair code) | **PROVEN** |
| **12:37** | Last audited activity — `evidence.read` on a stored crop | **PROVEN** |
| **~13:30** | **User reports CCTV video streaming** | **user report** |
| 14:12 | `analysis_enabled` work committed (`6992489`) | PROVEN |
| 14:14–14:22 | Perception repairs edited on disk | PROVEN |
| **14:38** | Worker respawns with repaired code; **kit files NOT rewritten** | **PROVEN** |
| 15:10–15:45 | RTSP 554 confirmed unreachable; wall shows CONNECTING | **PROVEN** |

**The critical gap:** between 10:52 and 14:38 the backend ran with synthesis
unbound. In that window frames could flow to the wall while producing no
observation, no finding and no incident.

---

## 11. The post-1:30 PM streaming contradiction

Each hypothesis, decided on evidence:

| # | Hypothesis | Verdict |
|---|---|---|
| A | Atlas uses another working CCTV source | **DISPROVEN** — same path template, defaults to file, not running |
| B | Another application is streaming the cameras | **DISPROVEN inside the workspace**; **UNKNOWN outside it** (see below) |
| C | The DVR public IP changed | **DISPROVEN** — DDNS resolves to 203.118.57.154 and that host answers |
| D | Vision OS has stale camera configuration | **DISPROVEN** — byte-identical to the console's validated URL |
| E | The frontend points at a different service | **DISPROVEN** — one backend, `/api/v1/wall/*` |
| F | Streaming via an internal LAN path | **UNKNOWN** — plausible; not observable from this host |
| G | The reported IP is only the router | **PROVEN** — EdgeOS; but it is also the only configured path |
| **H** | **The video was cached / replayed / previously captured** | **PARTLY DISPROVEN** — no video file written today, no media newer than 2026-08-18. But at **12:37 the user was reading stored evidence crops**, which is real DVR imagery rendered in the UI |
| I | Multiple DVR/NVR systems exist | **DISPROVEN** in configuration; **UNKNOWN** on the physical site |
| J | Cameras 11–14 are not the ones that were streaming | **UNKNOWN** — numbering is consistent everywhere; physical identity unverifiable from here |
| **K** | **Another explanation** | 🟢 **the leading one — see below** |

### The leading explanation (INFERRED, mechanism PROVEN)

> Between **10:52 and 14:38** the backend was running with **synthesis unbound**.
> Frames reaching `LiveRtspSource` still fed the Live Wall, which needs nothing
> but frames — so tiles would show **live video** — while every observation was
> discarded before publication, so **no alert could ever appear**.
>
> That is exactly "CCTV streaming, no alerts". RTSP then stopped at some point
> before 15:10, and the 14:38 reboot — the first with repaired code — found the
> transport already dead, which is why all four tiles now read CONNECTING.

**Why it cannot be fully proven:** the wall keeps no durable record, stream
viewing is **not audited** (the audit table has no wall or stream action), and the
worker that ran during that window has been replaced. The mechanism is proven;
the timing is not.

**One alternative remains open and is outside this workspace:** the operator may
have viewed the cameras through the DVR's own interface, a vendor mobile app
(Dahua P2P/cloud relay needs **no** port forward), or from the LAN. Any of those
would show live video with no involvement from any repository here. **UNKNOWN,
and only the operator can say which screen they were looking at.**

---

## 12. Layer-by-layer pipeline health

| Layer | Status | Evidence |
|---|---|---|
| 0 · Camera input | 🔴 **BLOCKED** | `SYN_SENT` to 554; 3× 12 s timeouts; 554/8554/37777/37778 all silently dropped while 80/443 answer in 0.04 s | 
| 1 · Frame capture | 🔴 **BLOCKED** | consequence of 0; wall reports zero frames, "resolution unknown" |
| 2 · Detection | ⚪ **UNKNOWN** | starved — cannot be exercised |
| 3 · Tracking | ⚪ **UNKNOWN** | starved. `tracker.sort` composed ✅ |
| 4 · Registry / identity | ⚪ **UNKNOWN** | starved |
| 5 · Observation publication | 🟢 **HEALTHY** | synthesis **binds** against the real log; 0 files written, 0 removed |
| 6 · Synthesis | 🟢 **HEALTHY** | `synthesis bound: True` |
| 7 · Exposure | 🟢 **HEALTHY** | `exposure bound: True` |
| 8 · Compliance | 🟡 **DEGRADED** | driver alive on its 5 s interval, receiving **0 subjects** |
| 9 · Incident | ⚪ **UNKNOWN** | 1,990 stored; none since 10:36 |
| 10 · Alert | ⚪ **UNKNOWN** | notifications frozen since 10:51 |
| 11 · Frontend display | 🟢 **HEALTHY** | correctly reports CONNECTING — the UI is telling the truth |

**How far did the last real data travel?** All the way. On 2026-09-03 up to
**10:35 IST** the pipeline completed end to end: frames → detection → tracking →
registry → VLM → observation → compliance → **incident + evidence + notification**.
The system is not broken end to end; it is **starved at layer 0**, and it was
briefly **truncated at layer 5** between 10:52 and 14:38.

---

## 13. Root cause classification

Two distinct causes. **Not collapsed.**

### Cause 1 — External network / router — **PROVEN, OPEN**
*Category 1.* The EdgeRouter at `203.118.57.154` does not forward inbound TCP 554
to the DVR. SYN packets are dropped, not refused. Everything except 80/443 behaves
identically. The DVR is unreachable from this host; its own state is unobservable.

### Cause 2 — Backend pipeline failure — **PROVEN, REPAIRED**
*Category 6.* Conformance-kit artefacts poisoned the durable observation log, so
every boot from the second onward refused the adapter, leaving synthesis unbound
and publication dead **while frames continued to flow**. Repaired in the prior
phase; the 14:38 boot wrote no kit files, proving the fix live.

### Explicitly NOT causes — **DISPROVEN**
* Category 3 camera configuration drift — configuration is correct and validated.
* Category 4 cross-repository mismatch — one source, identical construction.
* Category 5 wrong channel mapping — consistent everywhere.
* Category 7 frontend integration — the UI is reporting accurately.
* Category 8 Atlas integration difference — Atlas is a different product, not in the path.
* Category 9 deployment/version mismatch — running code matches the repository.
* Category 10 perception architecture — repaired, and blocked before it receives a frame.

**Category 11 (multiple simultaneous causes) is the correct overall
classification**: causes 1 and 2 are independent, overlapped in time, and produced
one confusing symptom.

---

## 14. Unknowns requiring access I do not have

| Unknown | What would settle it |
|---|---|
| Is the DVR powered on and its RTSP service enabled? | LAN access, or the router's admin UI |
| The DVR's current LAN IP (did DHCP move it?) | Router DHCP lease table |
| Exactly when RTSP stopped | Router/ISP logs |
| Which screen showed video after 13:30 | The operator's own account |
| Whether channel 11 is physically the intended kitchen | Someone looking at a monitor |
| Whether another on-site app streams the DVR | Site network inspection |

---

## 15. Recommended next action

**One action unblocks everything, and it is outside this repository.**

1. **On the EdgeRouter — restore WAN TCP 554 → `<DVR LAN IP>:554`.** Check first
   whether the DVR's LAN address moved via DHCP; a silent lease change breaks a
   port-forward exactly like this. Give the DVR a static mapping.
2. **Ask the operator one question:** *which screen was showing video after
   1:30 PM — the UnityWorks Live Wall, the DVR's own web page, or a phone app?*
   That single answer decides between the leading explanation and hypothesis F/B,
   and no amount of code reading can substitute for it.
3. **Then verify, in order:**
   ```bash
   # TCP opens?
   python -c "import socket;s=socket.socket();s.settimeout(8);s.connect(('gayatri.freemyip.com',554));print('OPEN')"
   # Backend picks it up?  Expect Established, not SynSent.
   powershell -Command "Get-NetTCPConnection | ? { $_.RemotePort -eq 554 } | Select OwningProcess,State"
   # Frames arriving?  Wall tiles turn live and report a resolution.
   # Observations growing?
   ls -l unityworks-vision-ai-backend/data/observations/cam-*.jsonl
   ```
4. **Only then** measure `incidents / distinct object_ids` over an hour of real
   footage. It was **1.0000**; anything above 1.0 means deduplication is finally
   being exercised — the outstanding proof of the perception repair.

### Optional, non-blocking

* **Close public exposure of the router's admin UI** on 80/443.
* **Consider auditing wall stream access.** This audit could not date live viewing
  because it leaves no record — the single largest gap in reconstructing today.
* **Add `perception_publishing` and `camera_input` booleans to `/health/ready`**
  (drafted in the prior report, still unapplied). Failure 2 was invisible on the
  dashboard for hours precisely because readiness reports assembly only.

---

## 16. Risk assessment for proposed changes

| Change | Risk | Reversible | Note |
|---|---|---|---|
| Restore router port-forward | **Low** | Yes | Restores intended state; verify the LAN IP first |
| Close router admin exposure | **Low** | Yes | Do not lock yourself out — keep LAN access |
| Audit wall stream access | **Low** | Yes | Additive; adds rows to `audit_events` — check volume, MJPEG is long-lived |
| Readiness booleans | **Low–medium** | Yes | Touches a public **unauthenticated** route; booleans only, `ready` gating unchanged |
| Restart the backend | **Medium** | Yes | Only after RTSP returns. Consider dropping `--reload`: every file save tears down all four sessions |
| Any perception change | **Not recommended** | — | No evidence supports one. The architecture is repaired and starved, not faulty |

---

## Closing statement

**The system is not blocked by machine learning.** Detection, tracking, the
registry, the VLM, synthesis, exposure and compliance are all either proven
healthy or merely starved. The pipeline demonstrably worked end to end this
morning until 10:35, and the perception repairs are confirmed loaded and binding.

What stands between the current state and live alerts is **one port-forward rule
on a router** — plus one question only the operator can answer about which screen
was showing video after 1:30 PM.
