"""A brand carries two facts, not one, and that is the whole point of the file.

Dahua's main stream is subtype `0`; Hikvision's is `1`. A path template on its
own would make whoever adds a recorder supply that number from memory, and a
wrong guess does not fail at save time — it fails much later as a decoder error
on a camera that appears to have been configured correctly.

Keeping the pair together in data means the form can ask "what brand is it",
which is a question an installer can answer by reading the box.
"""

from __future__ import annotations

import pytest

from app.domain.recorder_brands import CUSTOM, Brand, BrandError, load_brands


def test_every_shipped_brand_declares_both_a_path_and_its_stream_numbers() -> None:
    brands = load_brands()
    assert brands, "the brands file must not be empty; the dropdown would have nothing in it"

    for key, brand in brands.items():
        assert "{channel}" in brand.path_template, key
        assert brand.main != brand.sub, f"{key} cannot use one number for both streams"
        assert brand.default_port > 0, key
        assert brand.label, key


def test_dahua_and_hikvision_disagree_about_which_number_means_main() -> None:
    """The regression this whole file exists to prevent.

    Before this, one hardcoded Dahua template served every device, so a
    Hikvision recorder could not be onboarded at all.
    """
    brands = load_brands()

    assert brands["dahua"].path(channel=3, stream_type="main") == (
        "/cam/realmonitor?channel=3&subtype=0"
    )
    assert brands["dahua"].path(channel=3, stream_type="sub") == (
        "/cam/realmonitor?channel=3&subtype=1"
    )
    # Hikvision concatenates the channel and the stream into one number, and
    # its main stream is 1 rather than 0. Both differences are real.
    assert brands["hikvision"].path(channel=3, stream_type="main") == "/Streaming/Channels/301"
    assert brands["hikvision"].path(channel=3, stream_type="sub") == "/Streaming/Channels/302"


def test_an_unknown_stream_type_is_refused_rather_than_guessed() -> None:
    """Defaulting to main would silently pull full-resolution video on every
    camera of a sixteen-channel recorder, which is a bandwidth decision nobody
    made."""
    with pytest.raises(BrandError, match="stream_type"):
        load_brands()["dahua"].path(channel=1, stream_type="highest")


def test_a_channel_below_one_is_refused() -> None:
    """Channel numbering starts at 1 on every device here. Zero would render a
    URL the recorder rejects, reported as a connection failure."""
    with pytest.raises(BrandError, match="channel"):
        load_brands()["dahua"].path(channel=0, stream_type="main")


def test_a_custom_brand_must_place_the_channel_somewhere() -> None:
    """A template without `{channel}` builds an identical URL for all sixteen
    channels. That connects, and shows the wrong camera — which is worse than
    failing, because nothing reports it."""
    with pytest.raises(BrandError, match=r"\{channel\}"):
        Brand(
            key=CUSTOM,
            label="Custom",
            path_template="/stream",
            main=0,
            sub=1,
            default_port=554,
        )


def test_a_custom_brand_is_otherwise_accepted() -> None:
    """The escape hatch has to actually work: a device matching nothing in the
    list is exactly when somebody needs this to be possible."""
    brand = Brand(
        key=CUSTOM,
        label="Custom",
        path_template="/live/ch{channel}/stream{subtype}",
        main=1,
        sub=2,
        default_port=8554,
    )

    assert brand.path(channel=4, stream_type="sub") == "/live/ch4/stream2"


def test_the_brands_are_loaded_once_and_not_mutable_by_a_caller() -> None:
    """A caller that could edit the returned mapping would change how every
    later recorder builds its URL, from anywhere in the process."""
    first = load_brands()
    assert load_brands() is first

    with pytest.raises((TypeError, AttributeError)):
        first["dahua"] = first["hikvision"]  # type: ignore[index]
