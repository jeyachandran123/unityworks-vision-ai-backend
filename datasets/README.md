# `datasets/` — what lives here, and what deliberately does not

**The production CCTV corpus is not in this directory and must never be committed
to Git.** It lives in an external store selected by `$VISION_OS_DATA_ROOT`.

What is here is the small, reviewable record that makes an experiment citable:
manifests, digests, provenance, session metadata and reports. None of it contains
a recoverable image of any person.

```
repository (this tree)                 external store ($VISION_OS_DATA_ROOT)
├── datasets/manifests/   ──cites──▶   ├── raw/
│     candidates/p9-live/v1.json       ├── candidates/
│     traces/p9-traces/v1.json         ├── annotations/
├── datasets/p9-v1, p9-v2              ├── benchmarks/
│     (P9 annotation manifests)        └── traces/
└── tools/p9_dataset/
```

## Getting set up

```bash
# 1. Choose a root OUTSIDE the repository. Any path; nothing is hard-coded.
export VISION_OS_DATA_ROOT=/srv/vision-os-data      # Linux
setx  VISION_OS_DATA_ROOT C:\vision-os-data          # Windows

# 2. Create the layers.
python -m tools.p9_dataset.store init

# 3. Check where you are pointed and that it is safe.
python -m tools.p9_dataset.store status

# 4. Verify a dataset against its manifest before using it in an experiment.
python -m tools.p9_dataset.store verify --layer candidates --dataset p9-live --version v1
```

If `$VISION_OS_DATA_ROOT` is unset, the store falls back to a **sibling of the
repository** (`../vision-os-data`). There is deliberately no default *inside* the
repository: a store under the working tree would put CCTV back where `git add .`
can reach it, and `DatasetStore.resolve()` refuses such a root outright.

## The five layers

| Layer | Holds | Pixels? | May a model write it? |
|---|---|---|---|
| `raw/` | original camera captures | yes | n/a — capture only |
| `candidates/` | frames the validated sampler selected | yes | yes, sampler output |
| `annotations/` | **human ground truth only** | no | **never** |
| `benchmarks/` | immutable evaluation sets from human labels | no | **never** |
| `traces/` | hashes, boxes, event decisions, session metadata | **no** | yes |

The `annotations/` row is the one the whole programme rests on. A `derived`
artifact in that layer is a machine label wearing ground truth's clothes, and
`store.build()` refuses to describe one.

## Retention matrix

Engineering policy is implemented. Anything marked **DECISION** needs a human
owner — this project does not invent legal or privacy requirements.

| Layer | Proposed retention | Rebuildable from | Status |
|---|---|---|---|
| `raw/` | shortest of all layers; it is the most sensitive and the largest | nothing — it is the origin | **DECISION**: retention period, and whether raw is kept at all once candidates exist |
| `candidates/` | until its dataset version is superseded or annotated | raw + a policy version, exactly | implemented: versions are immutable, so superseding is explicit |
| `annotations/` | long-lived, versioned, never overwritten | nothing — human effort | implemented: P9 manifests are immutable and digest-verified |
| `benchmarks/` | permanent for any published result | annotations + a split policy | implemented |
| `traces/` | longest; no person is identifiable from a trace | re-recordable, but only of a moment that has passed | implemented |

Two properties make this practical rather than aspirational:

* **Candidates are reconstructible.** Given `raw` and a frozen policy version,
  the sampler reproduces the same selection deterministically. So candidates can
  expire before raw does without losing the ability to rebuild them.
* **Traces are cheap and anonymous.** 96 KB per 4-minute 4-camera window against
  ~90 MB for the same window as frames. Sampling-policy work no longer needs the
  image corpus at all.

**Nothing here deletes anything.** `store.py` copies and verifies; it has no
delete path. Removal is a deliberate human act under a retention policy, because
*"do not silently discard inconvenient data"* is a standing rule and a storage
tool is precisely where that rule gets broken.

## Privacy and access boundary

### Implemented controls

| Control | Mechanism |
|---|---|
| CCTV cannot reach GitHub by accident | `.gitignore` covers every pixel format under `datasets/`; verified by test |
| New exposure is detected | `python -m tools.p9_dataset.guard` fails on any newly tracked or stageable image |
| The store cannot be placed inside the repo | `DatasetStore.resolve()` raises; verified by test |
| Corpus integrity is checkable | SHA-256 per artifact; dataset digest over sorted `(path, digest)` pairs |
| Versions are immutable | `ingest` refuses to overwrite a non-empty version |
| Copies are verified | every ingested file is re-hashed and compared before the copy is accepted |
| No credential ever recorded | collector reuses the production `RtspCameraConfig`; manifests store `credential_ref`, never a password |
| Traces carry no pixels | observation keys are exactly `{i, t, hash, boxes}`; verified by test |

### DECISIONS still required from a human owner

These are policy, not engineering, and this project does not fabricate them:

1. **Where the store may physically live** — local disk, an encrypted volume, a
   private object bucket. The abstraction supports all three; the choice is not
   ours.
2. **Who may read `raw/` and `candidates/`.** There is currently **no access
   control**: the store is an ordinary directory with filesystem permissions.
3. **Retention period for raw frames**, and whether raw is discarded once
   candidates and manifests exist.
4. **Whether benchmark sets may contain identifiable people.** They will, unless
   a decision is taken to crop or blur — and cropping changes what the benchmark
   measures, so this is a methodological decision as much as a privacy one.
5. **Whether the corpus may leave the premises at all** (backup, cloud, sharing).
6. **What happens to `datasets/kitchen-01/` and `datasets/vision-phase5/`**,
   which contain 153 CCTV-derived images already committed to Git history — see
   the P9.7 report.

## Rules that are not negotiable

1. A model output can never become ground truth.
2. Do not fabricate annotations.
3. Do not silently discard data.
4. Do not duplicate frames to manufacture camera balance.
5. Do not modify P9-v1 or P9-v2.
6. Do not rewrite historical evidence to make an architecture look cleaner.
7. Do not claim temporal coverage from metadata-only traces.
8. Do not claim representativeness the collected pixels do not support.
9. Do not put production CCTV into GitHub.
10. When evidence is insufficient, report `UNMEASURABLE_WITH_CURRENT_EVIDENCE`.

## Reference

* Storage, manifests, integrity — `tools/p9_dataset/store.py`
* Repository guard — `tools/p9_dataset/guard.py`
* Full rationale and audit — [`docs/production-hardening/P9_7_DATASET_STORAGE_REPORT.md`](../../docs/production-hardening/P9_7_DATASET_STORAGE_REPORT.md)
