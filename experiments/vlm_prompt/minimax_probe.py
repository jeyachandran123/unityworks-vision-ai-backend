"""Collect NEW MiniMax M3 evidence, kept apart from the Llama runs.

    python -m experiments.vlm_prompt.minimax_probe [budget_seconds]

§19 of the live-hardening brief: historical records produced by
`meta/llama-3.2-11b-vision-instruct` stay attributed to Llama, and a new model
gets a new evidence path. This writes `runs/minimax_m3_<date>.json` and touches
no existing run.

It exists because `run.py` cannot be used for this. That script walks the corpus
once and records whatever came back; against an account rate-limited to a few
percent it would record 40 refusals and 3 answers, which measures the quota
rather than the model. This one retries each subject under a wall-clock budget
and — importantly — **reports the attempt cost alongside every answer**, so the
two facts stay separable: how often MiniMax is reachable, and whether it is
right when it is.

Nothing here is ground truth. `truth` is the human annotation from
datasets/kitchen-01; `predicted` is a model output and is only ever compared
against the human label, never substituted for it.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .crops import build
from .run import ask, load_env, pose_states, render
from .variants import all_variants

RUNS = Path(__file__).resolve().parent / "runs"
MODEL = "minimaxai/minimax-m3"

#: Seconds to wait after a 429 before trying the same subject again. Rising,
#: because a quota that refused twice in a row is not about to yield to a third
#: immediate attempt.
BACKOFF = (2.0, 5.0, 12.0, 25.0)


def build_adapter(model: str):
    from app.configuration.settings import Settings
    from vision_os.adapters.understanding.nvidia_vl import NvidiaVisionUnderstander
    from vision_os.core.model.ids import AttributeKey

    settings = Settings()
    key = settings.vision_understander_api_key
    key = key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
    return NvidiaVisionUnderstander(
        producible=(AttributeKey("head_covering"), AttributeKey("face_covering")),
        api_key=key,
        model=model,
        base_url=settings.vision_nvidia_base_url or "https://integrate.api.nvidia.com/v1",
    )


def main(budget_s: float = 900.0) -> int:
    load_env()
    adapter = build_adapter(MODEL)
    variant = all_variants()["A"]
    poses = pose_states()
    subjects = build()

    started = time.perf_counter()
    cases: list[dict] = []
    attempts_total = 0
    rate_limited_total = 0

    for subject in subjects:
        if time.perf_counter() - started > budget_s:
            break
        prompt_text = render(variant, subject, poses.get(subject.key, "located"))
        attempts = 0
        answered = False
        for delay in (0.0, *BACKOFF):
            if delay:
                time.sleep(delay)
            if time.perf_counter() - started > budget_s:
                break
            attempts += 1
            attempts_total += 1
            raw, meta = ask(adapter, prompt_text, subject, variant.max_output_tokens)
            if not meta["refused"]:
                answered = True
                predicted, detail = variant.parse(raw, {})
                cases.append({
                    "frame": subject.frame,
                    "subject": subject.subject,
                    "truth": subject.truth,
                    "crop_sha256": subject.crop_sha256,
                    "attempts": attempts,
                    "predicted": predicted,
                    "detail": detail,
                    "raw": raw[:600],
                    "latency_ms": meta["latency_ms"],
                })
                break
            if "429" in meta["refusal_reason"]:
                rate_limited_total += 1
            else:
                cases.append({
                    "frame": subject.frame, "subject": subject.subject,
                    "truth": subject.truth, "crop_sha256": subject.crop_sha256,
                    "attempts": attempts, "predicted": None, "detail": {},
                    "raw": "", "latency_ms": meta["latency_ms"],
                    "error": meta["refusal_reason"][:200],
                })
                answered = True
                break
        if not answered:
            cases.append({
                "frame": subject.frame, "subject": subject.subject,
                "truth": subject.truth, "crop_sha256": subject.crop_sha256,
                "attempts": attempts, "predicted": None, "detail": {},
                "raw": "", "latency_ms": 0.0, "error": "exhausted retries (429)",
            })
        print(f"  {subject.key:22} attempts={attempts} "
              f"predicted={cases[-1].get('predicted')}", flush=True)

    answered_cases = [c for c in cases if c.get("predicted") is not None]
    payload = {
        "_comment": [
            "NEW MiniMax M3 evidence. Never mixed with the Llama runs in this",
            "directory, which were produced on 2026-08-27 by",
            "meta/llama-3.2-11b-vision-instruct and remain attributed to it.",
            "",
            "`attempts` is per subject and is the cost of getting an answer at",
            "all. Accuracy below is computed ONLY over subjects that answered,",
            "so it describes the model and not the quota; `answer_rate` is the",
            "other half and neither number means anything without the other.",
            "",
            "`truth` is the human annotation from datasets/kitchen-01. Model",
            "output is never ground truth.",
        ],
        "model": MODEL,
        "variant": "A",
        "prompt_source": "config/policies/kitchen-safety.example.json",
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "budget_s": budget_s,
        "subjects_attempted": len(cases),
        "subjects_in_corpus": len(subjects),
        "requests_total": attempts_total,
        "rate_limited_total": rate_limited_total,
        "answer_rate": round(len(answered_cases) / len(cases), 4) if cases else 0.0,
        "request_success_rate": (
            round(len(answered_cases) / attempts_total, 4) if attempts_total else 0.0
        ),
        "agreement_when_answered": (
            round(sum(c["predicted"] == c["truth"] for c in answered_cases)
                  / len(answered_cases), 4) if answered_cases else None
        ),
        "cases": cases,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"minimax_m3_{datetime.now(UTC):%Y%m%d}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"  subjects attempted   : {len(cases)}/{len(subjects)}")
    print(f"  requests sent        : {attempts_total}  (429s: {rate_limited_total})")
    print(f"  answered             : {len(answered_cases)}")
    print(f"  request success rate : {payload['request_success_rate']:.1%}")
    print(f"  agreement when answered: {payload['agreement_when_answered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 900.0))
