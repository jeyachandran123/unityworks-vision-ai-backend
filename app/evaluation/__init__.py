"""Reading the evaluation artifacts this repository already produces.

A **consumer** of `tools/vision_eval/`, `experiments/vlm_prompt/` and the
committed dataset reports. It computes no metric of its own, changes no
evaluation logic, and writes to nothing under `datasets/` or `experiments/`.

Every number that leaves this package carries where it came from, what it means,
and what it may not be compared with.
"""
