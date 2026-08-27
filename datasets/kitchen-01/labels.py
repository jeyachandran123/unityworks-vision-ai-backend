"""Human annotations for kitchen-01, recorded by visual inspection.

Every state below was decided by a person looking at an enlarged crop of the
frame in ``review/``. None of it was read from YOLO, from the VLM, or from a
compliance result — an evaluation built from model output measures a system
against itself.

Box geometry comes from detector proposals that were then **visually confirmed
to be real people**. That is a deliberate, and limiting, choice: it means a
person the detector never proposed was never annotated, so this dataset cannot
measure detection recall. See ``box_source`` in the manifest.

Legend — ``P`` present, ``A`` absent, ``N`` not visible, ``U`` unknown.

The head column is read from the top band: a hairnet, a cap or a bandana is
``P``; bare hair is ``A``; a head turned away, cropped off, or too small to
resolve is ``N``. The hand column is read the same way and is ``N`` far more
often, because this camera looks down at torsos.
"""

# (frame_index, subject_index): (head_covering, hand_covering, note)
LABELS = {
    (60, 0): ("P", "A", "blue hairnet; both hands bare on the mixing bowl"),
    (60, 1): ("P", "N", "blue cap; forearm raised, hand behind the head"),
    (180, 0): ("P", "N", "blue hairnet; hands in front, out of view"),
    (180, 1): ("N", "N", "bent over, head region dark and unresolvable"),
    (300, 0): ("P", "N", "blue hairnet clear; hands below the counter line"),
    (300, 1): ("P", "N", "blue cap; arm motion-blurred, hand behind the head"),
    (420, 0): ("P", "N", "blue hairnet"),
    (420, 1): ("P", "N", "blue cap; hands at hips, roughly 15px, not resolvable"),
    (420, 2): ("P", "N", "blue cap; hands together at chest but too small to read"),
    (420, 3): ("P", "N", "dark bandana covering the head; hands out of view"),
    (540, 0): ("P", "N", "blue hairnet"),
    (540, 1): ("N", "N", "bent fully over; head is an unresolvable dark mass"),
    (660, 0): ("P", "N", "blue hairnet; pale object near the hand, not identifiable"),
    (660, 1): ("N", "N", "head cropped off at the frame edge"),
    (660, 2): ("P", "N", "dark cap, clearly worn; hands below the counter"),
    (660, 3): ("P", "N", "blue hairnet visible despite blur"),
    (780, 0): ("P", "N", "dark bandana; extended arm bare but the hand is blurred"),
    (780, 1): ("P", "N", "blue hairnet; facing away"),
    (780, 2): ("N", "N", "facing away and bent; head region ambiguous"),
    (780, 3): ("N", "N", "head cropped off at the top of the box"),
    (780, 4): ("P", "N", "blue hairnet"),
    (900, 0): ("P", "N", "dark knitted cap, close to the lens"),
    (900, 1): ("P", "N", "pale cap; distant"),
    (900, 2): ("P", "N", "blue hairnet; bent over a pot"),
    (900, 3): ("N", "N", "head only partly in the box"),
    (1020, 0): ("P", "N", "blue hairnet"),
    (1020, 1): ("P", "N", "blue hairnet; bare forearm, hand not resolvable"),
    (1020, 2): ("P", "N", "dark cap; hand near the counter but heavily blurred"),
    (1020, 3): ("N", "N", "head turned away and out of the box"),
    (1140, 0): ("P", "N", "blue hairnet"),
    (1140, 1): ("P", "N", "blue hairnet; distant"),
    (1140, 2): ("N", "N", "bent over; head outside the box"),
    (1260, 0): ("P", "N", "blue hairnet"),
    (1260, 1): ("N", "N", "bent over, head not in view"),
    (1380, 0): ("P", "N", "blue hairnet"),
    (1500, 0): ("P", "N", "blue hairnet; hand in the bowl"),
    (1500, 1): ("P", "A", "blue hairnet; both hands bare, holding an item at the waist"),
    (1500, 2): ("N", "A", "head turned away; both hands bare and clearly visible"),
    (1620, 0): ("P", "N", "blue hairnet"),
    (1620, 1): ("N", "N", "bent over, head out of frame"),
    (1740, 0): ("P", "N", "blue hairnet"),
    (1740, 1): ("N", "N", "head not in the box"),
    (1740, 2): ("N", "N", "distant and truncated"),
}
