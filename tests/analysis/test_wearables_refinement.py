"""Unit tests for refined wearable and earbud categorization."""

from proctoring.analysis.wearables import (
    WearableCategory,
    WearableDetection,
    WearableDetector,
    EventType,
)


def test_wearable_classification_categories():
    detector = WearableDetector(auto_load=False)
    ear_regions = [(100, 100, 160, 200)]  # width 60, height 100; y from 100 to 200

    # 1. Over-ear headphones
    headphone_det = WearableDetection(
        target="headphones",
        prompt="over-ear headphones",
        event_type=EventType.HEADPHONES_DETECTED,
        confidence=0.75,
        bbox=(80, 80, 220, 220),
    )
    cat, reason = detector._classify_category(headphone_det, ear_regions)
    assert cat == WearableCategory.OVER_EAR_HEADPHONES

    # 2. Earring (other ear object) at lobule (y > 175)
    earring_det = WearableDetection(
        target="earbuds",
        prompt="small white earbud",
        event_type=EventType.EARBUDS_SUSPECTED,
        confidence=0.35,
        bbox=(125, 185, 140, 198),  # bottom 15% of ear
    )
    cat, reason = detector._classify_category(earring_det, ear_regions)
    assert cat == WearableCategory.OTHER_EAR_OBJECT
    assert "earring" in reason.lower()

    # 3. AirPod in concha with hand corroboration
    airpod_det = WearableDetection(
        target="earbuds",
        prompt="wireless earbud in ear",
        event_type=EventType.EARBUDS_SUSPECTED,
        confidence=0.48,
        bbox=(115, 130, 145, 165),  # concha center
    )
    cat, reason = detector._classify_category(airpod_det, ear_regions, hand_corroborated=True)
    assert cat == WearableCategory.EARBUD_AIRPOD
    assert "concha" in reason.lower() or "airpod" in reason.lower()
