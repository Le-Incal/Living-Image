"""
Consistency tests for the relighting prompt.

Every statement the prompt makes about the sun, shadows, and color must
agree with every other statement in the same prompt, and with the solar
model that produced it. A prompt that contradicts itself asks the model
to break a tie, and different models break it differently.

Frame convention: the camera faces north, so east is frame RIGHT,
west is frame LEFT, south is BEHIND the camera, north is AHEAD (into
the scene). The prompt never names compass directions, because the
real orientation of the building is unknown.
"""

import math
import re

import pytest

import prompts
from prompts import build_prompt, generate_time_slots

SLOTS = generate_time_slots()
IDS = [s.label for s in SLOTS]


def _slot(hour):
    return next(s for s in SLOTS if s.hour == hour)


# ──────────────────────────────────────────────
# One sun: sun side and shadow side agree
# ──────────────────────────────────────────────

class TestSunAndShadowAgree:

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_shadow_side_is_opposite_light_side(self, slot):
        light = prompts._light_side(slot.sun_azimuth)
        shadow = prompts._shadow_side(slot.sun_azimuth)
        assert shadow == prompts.OPPOSITE_SIDE[light], (
            f"{slot.label}: light from {light} but shadows fall {shadow}"
        )

    def test_8am_light_from_right_shadows_fall_left(self):
        s = _slot(8)
        assert prompts._light_side(s.sun_azimuth) == "RIGHT"
        assert prompts._shadow_side(s.sun_azimuth) == "LEFT"

    def test_8pm_light_from_left_shadows_fall_right(self):
        s = _slot(20)
        assert prompts._light_side(s.sun_azimuth) == "LEFT"
        assert prompts._shadow_side(s.sun_azimuth) == "RIGHT"

    def test_solar_noon_light_from_behind_shadows_fall_ahead(self):
        s = _slot(prompts.SOLAR_NOON_HOUR)
        assert prompts._light_side(s.sun_azimuth) == "BEHIND"
        assert prompts._shadow_side(s.sun_azimuth) == "AHEAD"

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_prompt_states_the_computed_sides(self, slot):
        p = build_prompt(slot)
        light_phrase = prompts.LIGHT_PHRASES[prompts._light_side(slot.sun_azimuth)]
        shadow_phrase = prompts.SHADOW_PHRASES[prompts._shadow_side(slot.sun_azimuth)]
        assert light_phrase in p, f"{slot.label}: light phrase missing"
        assert shadow_phrase in p, f"{slot.label}: shadow phrase missing"

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_prompt_uses_no_compass_words(self, slot):
        """The building's orientation is unknown; frame language only."""
        p = build_prompt(slot).lower()
        for word in ("east", "west", "north", "south"):
            assert not re.search(rf"\b{word}", p), f"{slot.label}: contains '{word}'"


# ──────────────────────────────────────────────
# One solar noon
# ──────────────────────────────────────────────

class TestSingleSolarNoon:

    def test_elevation_peaks_at_solar_noon(self):
        peak = max(SLOTS, key=lambda s: s.sun_elevation)
        assert peak.hour == prompts.SOLAR_NOON_HOUR

    def test_azimuth_is_due_south_at_solar_noon(self):
        assert _slot(prompts.SOLAR_NOON_HOUR).sun_azimuth == 180

    def test_color_temperature_peaks_at_solar_noon(self):
        peak = max(SLOTS, key=lambda s: s.color_temp_k)
        assert peak.hour == prompts.SOLAR_NOON_HOUR

    def test_noon_label_does_not_claim_highest_sun_unless_it_is(self):
        """12 PM text must not call the sun 'highest' if the peak is 1 PM."""
        if prompts.SOLAR_NOON_HOUR != 12:
            assert "highest" not in build_prompt(_slot(12)).lower()


# ──────────────────────────────────────────────
# Solar model matches its stated location and season
# ──────────────────────────────────────────────

class TestSolarModelMatchesClaim:

    def test_peak_elevation_matches_40N_summer_solstice(self):
        """At 40°N on the June solstice the noon sun stands about 73.4°."""
        expected = 90 - prompts.LATITUDE + 23.44
        peak = max(s.sun_elevation for s in SLOTS)
        assert abs(peak - expected) < 1.0, f"peak {peak} vs expected {expected:.1f}"

    def test_8pm_sun_is_low_but_above_horizon(self):
        el = _slot(20).sun_elevation
        assert 0 < el < 8, f"8 PM elevation {el}"

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_prompt_never_puts_a_risen_sun_below_the_horizon(self, slot):
        if slot.sun_elevation > 0:
            assert not re.search(r"below (the )?horizon", build_prompt(slot).lower())


# ──────────────────────────────────────────────
# Shadow length agrees with elevation
# ──────────────────────────────────────────────

class TestShadowLength:

    @pytest.mark.parametrize("el", [4.2, 10, 25.9, 45, 73.4])
    def test_ratio_is_cotangent_of_elevation(self, el):
        expected = 1 / math.tan(math.radians(el))
        assert abs(prompts._shadow_length_ratio(el) - expected) < 0.05

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_high_sun_prompt_never_says_long(self, slot):
        if slot.sun_elevation > 40:
            assert not re.search(r"\blong\b", build_prompt(slot).lower()), slot.label

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_low_sun_prompt_never_says_short_or_minimal(self, slot):
        if slot.sun_elevation < 20:
            p = build_prompt(slot).lower()
            assert not re.search(r"\bshort\b", p), slot.label
            assert "minimal shadow" not in p, slot.label

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_prompt_states_shadow_length_as_multiple_of_height(self, slot):
        assert "times the height" in build_prompt(slot), slot.label


# ──────────────────────────────────────────────
# Color temperature follows the sun, not the clock
# ──────────────────────────────────────────────

class TestColorFollowsElevation:

    def test_lower_sun_is_never_cooler(self):
        ordered = sorted(SLOTS, key=lambda s: s.sun_elevation)
        for lo, hi in zip(ordered, ordered[1:]):
            assert lo.color_temp_k <= hi.color_temp_k, (
                f"{lo.label} ({lo.sun_elevation}°) is cooler than "
                f"{hi.label} ({hi.sun_elevation}°)"
            )


# ──────────────────────────────────────────────
# Prompt hygiene
# ──────────────────────────────────────────────

class TestPromptHygiene:

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_sun_is_never_placed_in_the_image(self, slot):
        """Telling the model the sun is 'in the scene' invites a painted sun disc."""
        p = build_prompt(slot).lower()
        assert "sun position in the scene" not in p
        assert "sun on the" not in p

    def test_vegetation_rule_stated_once(self):
        p = build_prompt(SLOTS[0]).lower()
        assert p.count("trees") == 1

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_no_em_dashes(self, slot):
        assert "—" not in build_prompt(slot)

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_prompt_is_concise(self, slot):
        words = len(build_prompt(slot).split())
        assert words <= 260, f"{slot.label}: {words} words"

    @pytest.mark.parametrize("slot", SLOTS, ids=IDS)
    def test_dusk_note_only_when_sun_is_low(self, slot):
        has_note = "darker than midday" in build_prompt(slot).lower()
        assert has_note == (slot.sun_elevation < 8), slot.label
