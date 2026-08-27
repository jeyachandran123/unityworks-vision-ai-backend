"""Where each subject's head actually is, relative to its detector box.

Recorded by visual inspection of `review/headloc/*.jpg`, which draw the person
box with decile gridlines so a head's vertical extent can be read off as a
fraction of box height. Never derived from a model.

Values are ``(top, bottom)`` as fractions of the **person box height**, matching
the `evidence_region` convention. ``None`` means the head is not inside the
detector's person box at all — either above it, beside it, or absent.

This is the ground truth for *localization*, and it is deliberately separate
from `head_covering` in `labels.py`, which records *readability*. The two are
different questions and Phase 4.3 exists because the system conflated them:

  - f01500 s2 — head located, but turned away: locatable, not readable
  - f00540 s1 — head located low in the box, looking down: locatable, not readable
  - f00660 s1 — head above the box entirely: not locatable

A system that cannot tell these apart reports a violation against a worker whose
head it never saw.
"""

# (frame_index, subject_index): (top, bottom) as fractions of box height, or None
HEAD_BANDS = {
    (60, 0): (0.00, 0.15),
    (60, 1): (0.00, 0.16),
    (180, 0): (0.00, 0.13),
    (180, 1): None,            # bent fully over; head not identifiable in the box
    (300, 0): (0.00, 0.13),
    (300, 1): (0.00, 0.16),
    (420, 0): (0.00, 0.15),
    (420, 1): (0.00, 0.15),
    (420, 2): (0.00, 0.15),
    (420, 3): (0.05, 0.45),    # bent toward the lens; head large and high-left
    (540, 0): (0.00, 0.13),
    (540, 1): (0.40, 0.75),    # bent fully over: head LOW in the box, not at the top
    (660, 0): (0.00, 0.15),
    (660, 1): None,            # head above the box top
    (660, 2): (0.08, 0.42),
    (660, 3): (0.00, 0.20),
    (780, 0): (0.05, 0.45),
    (780, 1): (0.00, 0.13),
    (780, 2): None,
    (780, 3): None,            # head above the box top
    (780, 4): (0.00, 0.18),
    (900, 0): (0.25, 0.70),    # close to the lens: head in the MIDDLE of the box
    (900, 1): (0.00, 0.15),
    (900, 2): (0.00, 0.15),
    (900, 3): None,            # only partly in the box
    (1020, 0): (0.00, 0.15),
    (1020, 1): (0.00, 0.18),
    (1020, 2): (0.00, 0.35),
    (1020, 3): None,
    (1140, 0): (0.00, 0.15),
    (1140, 1): (0.00, 0.15),
    (1140, 2): None,
    (1260, 0): (0.00, 0.14),
    (1260, 1): None,
    (1380, 0): (0.00, 0.14),
    (1500, 0): (0.00, 0.14),
    (1500, 1): (0.00, 0.16),
    (1500, 2): (0.02, 0.15),   # head IS in the box and covered; it is turned away
    (1620, 0): (0.00, 0.15),
    (1620, 1): None,
    (1740, 0): (0.00, 0.15),
    (1740, 1): None,
    (1740, 2): None,
}
