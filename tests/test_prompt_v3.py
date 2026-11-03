"""
Tests for prompt v3 and its test variants.

v3 answers the second-round reviews of v2:
- Direction is stated finely enough that every hour reads differently
  (v2's eight 45-degree sectors gave 9, 10 and 11 AM the same words).
- "Ahead" and "behind" are defined relative to the camera.
- The render's baked-in lighting is replaced, not layered over.
- A very low sun gives long but soft, low-contrast shadows and a darker image.
- Shadow length comes from the unrounded elevation.
- Windows keep their original lit or unlit state by default (no flicker).

Variants, for benchmarking against each other:
- v2:          the previous prompt, unchanged, as the baseline
- v3:          new wording, even sun motion (15 degrees per hour)
- v3-true-sun: new wording, true solar azimuth
- v3-glow:     new wording, even warm window glow when the sun is low
"""

import math
import re

import pytest

import prompts
import prompts_v3 as v3

HOURS = list(range(8, 21))


def _slots(variant):
    return {s.hour: s for s in v3.generate_slots(variant)}


def _prompt(hour, variant="v3"):
    return v3.build(_slots(variant)[hour], variant)


def _direction_line(p):
    return next(line for line in p.splitlines() if line.startswith("- Sunlight comes"))


# ──────────────────────────────────────────────
# Variant registry
# ──────────────────────────────────────────────

class TestVariants:

    def test_variant_names(self):
        assert set(v3.VARIANTS) == {"v2", "v3", "v3-true-sun", "v3-glow"}

    def test_default_is_v3(self):
        assert v3.DEFAULT_VARIANT == "v3"

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError):
            v3.generate_slots("v9")
        with pytest.raises(ValueError):
            v3.build(_slots("v3")[8], "v9")

    @pytest.mark.parametrize("hour", HOURS)
    def test_v2_is_the_unchanged_baseline(self, hour):
        legacy = {s.hour: s for s in prompts.generate_time_slots()}[hour]
        assert _prompt(hour, "v2") == prompts.build_prompt(legacy)

    @pytest.mark.parametrize("variant", ["v3", "v3-true-sun", "v3-glow"])
    def test_thirteen_unique_prompts(self, variant):
        ps = [_prompt(h, variant) for h in HOURS]
        assert len(ps) == 13 and len(set(ps)) == 13


# ──────────────────────────────────────────────
# Direction: camera-relative, clock face, distinct per hour
# ──────────────────────────────────────────────

class TestDirection:

    def test_terms_are_defined(self):
        p = _prompt(10)
        assert "AHEAD means into the scene" in p
        assert "BEHIND means behind the camera" in p
        assert "12 is straight ahead" in p
        assert "6 is directly behind the camera" in p

    @pytest.mark.parametrize("variant", ["v3", "v3-true-sun", "v3-glow"])
    def test_every_hour_gets_its_own_direction(self, variant):
        lines = [_direction_line(_prompt(h, variant)) for h in HOURS]
        assert len(set(lines)) == 13, lines

    @pytest.mark.parametrize("rel, clock", [
        (0, "12:00"), (15, "12:30"), (90, "3:00"), (105, "3:30"),
        (180, "6:00"), (270, "9:00"), (285, "9:30"), (345, "11:30"), (80.3, "2:30"),
    ])
    def test_clock_reading(self, rel, clock):
        assert v3.clock_reading(rel) == clock

    @pytest.mark.parametrize("hour, light, shadow", [
        (8, "3:30", "9:30"), (13, "6:00", "12:00"), (20, "9:30", "3:30"),
    ])
    def test_even_sun_clock_positions(self, hour, light, shadow):
        line = _direction_line(_prompt(hour))
        assert f"about {light}" in line
        assert f"toward about {shadow}" in _prompt(hour)

    def test_solar_noon_words(self):
        p = _prompt(13)
        assert "from directly behind the camera" in p
        assert "Shadows fall straight away from the camera, into the scene" in p

    def test_8am_words(self):
        p = _prompt(8)
        assert "from the RIGHT side of the frame, slightly behind the camera" in p
        assert "toward the LEFT side of the frame, angled slightly away from the camera" in p

    @pytest.mark.parametrize("variant", ["v3", "v3-true-sun"])
    @pytest.mark.parametrize("hour", HOURS)
    def test_shadow_clock_is_opposite_light_clock(self, variant, hour):
        s = _slots(variant)[hour]
        light = v3.clock_reading(s.sun_azimuth)
        shadow = v3.clock_reading(s.sun_azimuth + 180)
        lh, lm = map(int, light.split(":"))
        sh, sm = map(int, shadow.split(":"))
        diff = ((sh % 12) * 60 + sm - (lh % 12) * 60 - lm) % 720
        assert diff == 360, f"{hour}: light {light}, shadow {shadow}"


# ──────────────────────────────────────────────
# True sun motion (variant)
# ──────────────────────────────────────────────

class TestTrueSun:

    def test_8am_true_azimuth(self):
        assert v3.true_azimuth(8) == pytest.approx(80.3, abs=0.1)

    def test_solar_noon_is_due_south(self):
        assert v3.true_azimuth(13) == pytest.approx(180.0, abs=0.01)

    @pytest.mark.parametrize("dh", [1, 2, 3, 4, 5])
    def test_symmetric_about_noon(self, dh):
        assert v3.true_azimuth(13 - dh) + v3.true_azimuth(13 + dh) == pytest.approx(360, abs=0.01)

    def test_moves_faster_near_noon(self):
        near = v3.true_azimuth(13) - v3.true_azimuth(12)
        far = v3.true_azimuth(9) - v3.true_azimuth(8)
        assert near > far

    def test_true_sun_variant_uses_true_azimuth(self):
        s = _slots("v3-true-sun")
        assert s[8].sun_azimuth == pytest.approx(80.3, abs=0.1)
        assert "from the RIGHT side of the frame, slightly in front of the camera" in _prompt(8, "v3-true-sun")

    def test_even_sun_variant_keeps_15_degree_steps(self):
        s = _slots("v3")
        assert [s[h].sun_azimuth for h in HOURS] == [180 + 15 * (h - 13) for h in HOURS]


# ──────────────────────────────────────────────
# Shadow length from the unrounded sun
# ──────────────────────────────────────────────

class TestShadowLength:

    def test_8pm_ratio_uses_exact_elevation(self):
        assert v3.shadow_ratio(20) == pytest.approx(13.5, abs=0.05)

    @pytest.mark.parametrize("hour", HOURS)
    def test_ratio_is_cot_of_exact_elevation(self, hour):
        el = v3.exact_elevation(hour)
        assert v3.shadow_ratio(hour) == pytest.approx(1 / math.tan(math.radians(el)))

    def test_displayed_elevation_is_rounded_exact(self):
        assert _slots("v3")[20].sun_elevation == round(v3.exact_elevation(20), 1)


# ──────────────────────────────────────────────
# Baked lighting, low sun, windows
# ──────────────────────────────────────────────

class TestLight:

    @pytest.mark.parametrize("hour", HOURS)
    def test_replaces_baked_in_lighting(self, hour):
        p = _prompt(hour)
        assert "The original render has its own lighting" in p
        assert "contact shadows" in p

    @pytest.mark.parametrize("hour", HOURS)
    def test_low_sun_gives_soft_shadows_and_darker_image(self, hour):
        p = _prompt(hour)
        low = _slots("v3")[hour].sun_elevation < 8
        assert ("soft-edged" in p) == low, hour
        assert ("low-contrast" in p) == low, hour
        assert ("darker than midday" in p) == low, hour

    @pytest.mark.parametrize("hour", HOURS)
    def test_default_keeps_window_states(self, hour):
        p = _prompt(hour)
        assert "keep each window lit or unlit exactly as in the original" in p
        assert "interior glow" not in p

    @pytest.mark.parametrize("hour", HOURS)
    def test_glow_variant_lights_windows_only_when_sun_is_low(self, hour):
        p = _prompt(hour, "v3-glow")
        low = _slots("v3-glow")[hour].sun_elevation < v3.GLOW_BELOW_ELEVATION
        assert ("same soft, warm interior glow" in p) == low, hour
        assert ("keep each window lit or unlit exactly as in the original" in p) == (not low), hour

    def test_glow_threshold_covers_7_and_8_pm_only(self):
        s = _slots("v3-glow")
        low = [h for h in HOURS if s[h].sun_elevation < v3.GLOW_BELOW_ELEVATION]
        assert low == [19, 20]


# ──────────────────────────────────────────────
# Preservation wording from the reviews
# ──────────────────────────────────────────────

class TestPreservation:

    def test_trunks_and_canopy_fixed(self):
        p = _prompt(10)
        assert "tree trunk and canopy outline" in p
        assert "do not add trees" in p.lower()

    def test_no_reframing(self):
        assert "Do not crop, zoom, pan, or recenter" in _prompt(10)

    def test_white_balance_fixed(self):
        assert "White balance" in _prompt(10)

    def test_reflectivity_preserved(self):
        assert "reflectivity" in _prompt(10)


# ──────────────────────────────────────────────
# Hygiene, carried over from v2
# ──────────────────────────────────────────────

class TestHygiene:

    @pytest.mark.parametrize("variant", ["v3", "v3-true-sun", "v3-glow"])
    @pytest.mark.parametrize("hour", HOURS)
    def test_rules(self, variant, hour):
        p = _prompt(hour, variant)
        low = p.lower()
        for word in ("east", "west", "north", "south"):
            assert not re.search(rf"\b{word}", low), word
        assert "—" not in p
        assert "Do not draw the sun itself" in p
        assert not re.search(r"below (the )?horizon", low)
        assert _slots(variant)[hour].label in p
        assert str(_slots(variant)[hour].color_temp_k) in p
        assert len(p.split()) <= 380, len(p.split())

    @pytest.mark.parametrize("hour", HOURS)
    def test_high_sun_never_long_low_sun_never_short(self, hour):
        el = _slots("v3")[hour].sun_elevation
        low = _prompt(hour).lower()
        if el > 40:
            assert not re.search(r"\blong\b", low)
        if el < 20:
            assert not re.search(r"\bshort\b", low)
