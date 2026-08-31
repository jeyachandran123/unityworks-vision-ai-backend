"""Run one variant over the whole corpus and record everything it did.

Uses the **production adapter**, so encoding, `max_side`, temperature, transport,
retry and refusal handling are the shipped ones. Only the instruction text
differs. The raw model text is kept verbatim on every case, because a run whose
answers cannot be re-read later is not evidence.

    python -m experiments.vlm_prompt.run A
    python -m experiments.vlm_prompt.run B C D

Runs are written to `experiments/vlm_prompt/runs/<variant>.json` and are never
edited afterwards. Re-running overwrites deliberately and stamps a new timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .crops import ATTRIBUTE, Subject, build
from .variants import all_variants

RUNS = Path(__file__).resolve().parent / "runs"
ROOT = Path(__file__).resolve().parents[2]

#: The pose producer's recorded verdict per subject — a genuine inference-time
#: signal (P33), used only to render Variant D's metadata header and to compute
#: the P8-gated view when scoring. Never used to choose or alter an answer.
POSE = ROOT / "tests" / "compliance" / "kitchen01_pose_verdicts.json"

_POSE_WORDS = {
    "located": "region_probably_visible",
    "low_confidence": "region_visibility_uncertain",
    "not_located": "region_probably_not_visible",
}


def load_env() -> None:
    """Read `.env` without printing anything from it."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def build_adapter():
    """The production understander, built exactly as the composition root does."""
    from vision_os.adapters.configuration import build_understander
    from vision_os.core.model.ids import AttributeKey

    adapter, note = build_understander(
        producible=(AttributeKey(ATTRIBUTE), AttributeKey("face_covering")),
        env_file=ROOT / ".env",
    )
    if adapter is None:
        raise SystemExit(f"no understander could be bound: {note}")
    return adapter, note


def pose_states() -> dict[str, str]:
    data = json.loads(POSE.read_text(encoding="utf-8"))
    return {f"{c['frame']}/{c['subject']}": c["state"] for c in data["cases"]}


def render(variant, subject: Subject, pose_state: str) -> str:
    if not variant.templated:
        return variant.prompt
    return variant.prompt.format(
        region_pct=45,
        width=subject.width,
        height=subject.height,
        subject_ref=f"{subject.frame}/{subject.subject}",
        observability_signal=_POSE_WORDS.get(pose_state, "unavailable"),
    )


def ask(adapter, prompt_text: str, subject: Subject, max_tokens: int):
    """One call through the production adapter. Returns (raw_text, meta)."""
    from PIL import Image

    from vision_os.core.model.ids import AttributeKey, CropId, PromptId, RequestId
    from vision_os.core.ports.understanding import (
        CropView,
        OutputSchema,
        RenderedPrompt,
        UnderstandingPortRequest,
    )

    image = Image.open(subject.crop_path).convert("RGB")
    rendered = RenderedPrompt(
        prompt_id=PromptId("experiment.vlm_prompt"),
        version="1.0.0",
        text=prompt_text,
        # Non-strict: the experiment reads the raw text itself, so the adapter's
        # schema split must not silently drop keys a variant declares.
        output_schema=OutputSchema(fields=(AttributeKey(ATTRIBUTE),), strict=False),
        content_hash=hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
        max_output_tokens=max_tokens,
    )
    request = UnderstandingPortRequest(
        request_id=RequestId(f"exp-{subject.frame}-{subject.subject}"),
        crops=(
            CropView(
                crop_id=CropId(subject.crop_sha256),
                pixels=memoryview(image.tobytes()),
                width=image.width,
                height=image.height,
                colour_space="rgb24",
            ),
        ),
        prompt=rendered,
        output_schema=rendered.output_schema,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    started = time.perf_counter()
    response = adapter.understand(request)
    elapsed = (time.perf_counter() - started) * 1000.0
    raw = (response.raw_output or b"").decode("utf-8", errors="replace")
    return raw, {
        "latency_ms": round(elapsed, 1),
        "refused": bool(response.refused),
        "refusal_reason": response.refusal_reason or "",
    }


def run_variant(code: str, subjects: list[Subject], adapter, note: str) -> dict:
    variant = all_variants()[code]
    poses = pose_states()
    cases = []
    print(f"\n=== Variant {code} — {variant.title} ===", flush=True)

    for index, subject in enumerate(subjects, 1):
        pose_state = poses.get(subject.key, "located")
        prompt_text = render(variant, subject, pose_state)
        raw, meta = ask(adapter, prompt_text, subject, variant.max_output_tokens)
        predicted, detail = variant.parse(raw, {})
        cases.append(
            {
                "frame": subject.frame,
                "subject": subject.subject,
                "truth": subject.truth,
                "note": subject.note,
                "crop_sha256": subject.crop_sha256,
                "pose_state": pose_state,
                "predicted": predicted,
                "detail": detail,
                "raw": raw[:600],
                **meta,
            }
        )
        mark = "." if predicted else "?"
        print(mark, end="" if index % 43 else "\n", flush=True)

    stats = getattr(adapter, "stats", None)
    latencies = sorted(c["latency_ms"] for c in cases)
    return {
        "_comment": [
            "One variant's answers over the whole kitchen-01 corpus.",
            "EVIDENCE. Never edited to make a result look better; re-run to change.",
            "`raw` is the model's verbatim text, truncated to 600 chars for size.",
        ],
        "variant": code,
        "title": variant.title,
        "hypothesis": variant.hypothesis,
        "notes": list(variant.notes),
        "prompt": variant.prompt,
        "prompt_sha256": hashlib.sha256(variant.prompt.encode()).hexdigest()[:16],
        "max_output_tokens": variant.max_output_tokens,
        "templated": variant.templated,
        "model": os.environ.get("VISION_NVIDIA_MODEL", ""),
        "binding": note,
        "temperature": 0.0,
        "max_side": int(os.environ.get("VISION_UNDERSTANDER_MAX_SIDE") or 448),
        "dataset": "kitchen-01",
        "attribute": ATTRIBUTE,
        "corpus_digest": hashlib.sha256(
            "".join(s.crop_sha256 for s in subjects).encode()
        ).hexdigest()[:16],
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "prompt_tokens_total": getattr(stats, "prompt_tokens", 0),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", choices=sorted(all_variants()))
    parser.add_argument(
        "--tag",
        default="",
        help=(
            "Suffix for the run file, e.g. --tag r2. Used to collect REPEATS of "
            "the same variant. The provider is not deterministic at temperature "
            "0 — measured at 79%% run-to-run agreement on this corpus — so a "
            "single run of a variant cannot be distinguished from noise."
        ),
    )
    args = parser.parse_args()

    load_env()
    subjects = build()
    adapter, note = build_adapter()
    print(f"bound: {note}")
    print(f"corpus: {len(subjects)} subjects, 448x448 crops")

    RUNS.mkdir(parents=True, exist_ok=True)
    for code in args.variants:
        result = run_variant(code, subjects, adapter, note)
        suffix = f"_{args.tag}" if args.tag else ""
        path = RUNS / f"variant_{code}{suffix}.json"
        path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        parsed = sum(1 for c in result["cases"] if c["predicted"])
        print(
            f"\n  -> {path.name}: {parsed}/{len(result['cases'])} parsed, "
            f"p50 {result['latency_p50_ms']} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
