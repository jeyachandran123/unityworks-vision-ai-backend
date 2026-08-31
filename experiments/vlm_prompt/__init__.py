"""Controlled VLM prompt / evidence-presentation experiment.

Isolated from production by construction: nothing here is imported by `app/`,
`compliance/` or `vision_os/`, and no variant prompt is ever written into
`config/policies/`. The production path is read, never modified.
"""
