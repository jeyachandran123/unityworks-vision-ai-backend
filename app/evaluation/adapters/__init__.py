"""One adapter per artifact family. Adding a source is adding a module here.

Each adapter answers the same question — "what did this file actually measure,
and what may it be compared with" — over a format it alone understands. The
differences between `tools/vision_eval`'s report shape and
`experiments/vlm_prompt`'s scoring shape are absorbed here so that neither tool
has to emit a new common format and the frontend never parses a repository file.
"""
