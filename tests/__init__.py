"""Makes ``tests`` a package, which is load-bearing after the Phase 1 rename.

`tests/vision_os/` and the platform package `vision_os/` now share a name. With
`tests` left as a plain directory, pytest's `prepend` import mode puts `tests/`
on `sys.path` — and `import vision_os` then resolves to the **test directory**,
so the platform fails to import with a misleading
`No module named 'vision_os.conformance.kit'`.

Under the old `app.vision_os` name the two could not collide. The rename created
the collision; this file resolves it by making pytest walk up to the repository
root and import these modules as `tests.vision_os.*`, leaving `vision_os` to mean
the platform and nothing else.

Renaming the directory would work too, and was rejected: `tests/vision_os` is
where every engineer on this programme expects the platform suite to live, and a
one-line packaging fact is cheaper than retraining that expectation.
"""
