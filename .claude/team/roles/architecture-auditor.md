# architecture-auditor — Vision AI backend brief

The map is `.claude/team/invariants.md` (six invariants plus the one-way arrow and the semantic
ceiling). Skill: `vision-os-boundaries`.

- Known gap to re-confirm: `TestSemanticCeiling` inspects identifiers only; a string literal such as
  `"kitchen"` inside `vision_os/` passes both vocabulary suites.
- Run the ORIGINAL venv's python from inside the scratch copy:
  `<repo>/.venv/Scripts/python.exe -m pytest <node-id>` — imports resolve to the copy.
- `vision_os/`, `compliance/`, `tools/` are protected in the real repository; mutate them only in the
  scratch copy, with Bash.
