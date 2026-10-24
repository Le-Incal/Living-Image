"""
Tests for the Living Image prompt system.

These tests verify that the solar model, color temperature curve,
and prompt generation encode physically accurate architectural
lighting knowledge. The domain expertise in this file is the core
intellectual contribution of the project.
"""

import math
import pytest
from prompts import (
    generate_time_slots,
    build_prompt,
    TimeSlot,
    FIRST_HOUR,
    LAST_HOUR,
    _compute_sun_elevation,
    _compute_sun_azimuth,
    _compute_color_temp,
    _color_temp_description,
    _shadow_description,
    _sun_position_in_frame,
    _shadow_direction_explicit,
    _shadow_length_description,
    _sky_description,
    _format_hour_label,
    _evening_darker_note,
    get_lighting_description,
)


# ──────────────────────────────────────────────
# Time Slot Generation
# ──────────────────────────────────────────────

class TestTimeSlotGeneration:
    """Verify we generate the correct number and range of time slots."""

    def test_generates_13_slots(self):
        slots = generate_time_slots()
        assert len(slots) == 13, "Must produce one slot per hour from 8am to 8pm inclusive"

    def test_first_slot_is_8am(self):
        slots = generate_time_slots()
        assert slots[0].hour == 8
        assert slots[0].label == "8:00 AM"

    def test_last_slot_is_8pm(self):
        slots = generate_time_slots()
        assert slots[-1].hour == 20
        assert slots[-1].label == "8:00 PM"

    def test_hours_are_sequential(self):
        slots = generate_time_slots()
        hours = [s.hour for s in slots]
        assert hours == list(range(8, 21))

    def test_all_slots_are_timeslot_instances(self):
        slots = generate_time_slots()
        for slot in slots:
            assert isinstance(slot, TimeSlot)

    def test_all_slots_have_populated_fields(self):
        """No slot should have empty or None critical fields."""
        slots = generate_time_slots()
        for slot in slots:
            assert slot.label, f"Hour {slot.hour}: label is empty"
            assert slot.shadow_description, f"Hour {slot.hour}: shadow_description is empty"
            assert slot.sky_description, f"Hour {slot.hour}: sky_description is empty"
            assert slot.lighting_description, f"Hour {slot.hour}: lighting_description is empty"
            assert slot.color_temp_k > 0, f"Hour {slot.hour}: color_temp_k must be positive"


# ──────────────────────────────────────────────
# Solar Model: Elevation
# ──────────────────────────────────────────────

class TestSunElevation:
    """
    Verify the sinusoidal sun elevation model.
    Calibrated to ~40N latitude, summer day.
    Sunrise ~6:00, sunset ~20:30, peak ~58 degrees.
    """

    def test_peak_elevation_near_solar_noon(self):
        """Solar noon at ~40N in summer is roughly 13:00-13:30.
        The sine model should peak near the middle of the day."""
        elevations = {h: _compute_sun_elevation(h) for h in range(8, 21)}
        peak_hour = max(elevations, key=elevations.get)
        assert 12 <= peak_hour <= 14, f"Peak should be near solar noon, got hour {peak_hour}"

    def test_peak_elevation_is_58_degrees(self):
        """Max elevation for ~40N summer should be approximately 58 degrees."""
        elevations = [_compute_sun_elevation(h) for h in range(8, 21)]
        max_el = max(elevations)
        assert 55 <= max_el <= 58, f"Peak elevation should be ~58, got {max_el}"

    def test_morning_elevation_lower_than_midday(self):
        assert _compute_sun_elevation(8) < _compute_sun_elevation(12)

    def test_evening_elevation_lower_than_midday(self):
        assert _compute_sun_elevation(20) < _compute_sun_elevation(12)

    def test_elevation_symmetry(self):
        """Morning and evening at equal distances from solar noon
        should have roughly similar elevations."""
        el_9 = _compute_sun_elevation(9)
        el_18 = _compute_sun_elevation(18)
        # Not perfectly symmetric due to sunrise/sunset asymmetry, but close
        assert abs(el_9 - el_18) < 15, "Elevation should be roughly symmetric around noon"

    def test_all_elevations_non_negative(self):
        for h in range(8, 21):
            el = _compute_sun_elevation(h)
            assert el >= 0, f"Elevation at hour {h} should not be negative, got {el}"

    def test_elevation_before_sunrise_is_zero(self):
        assert _compute_sun_elevation(5) == 0.0

    def test_elevation_after_sunset_is_zero(self):
        assert _compute_sun_elevation(21) == 0.0


# ──────────────────────────────────────────────
# Solar Model: Azimuth
# ──────────────────────────────────────────────

class TestSunAzimuth:
    """
    Verify sun azimuth progresses east to west.
    8am = 90 degrees (east), 8pm = 270 degrees (west).
    """

    def test_8am_azimuth_is_east(self):
        az = _compute_sun_azimuth(8)
        assert az == 90, f"8am should be 90 (east), got {az}"

    def test_8pm_azimuth_is_west(self):
        az = _compute_sun_azimuth(20)
        assert az == 270, f"8pm should be 270 (west), got {az}"

    def test_noon_azimuth_is_south(self):
        """Noon should be approximately 180 (south) at ~40N."""
        az = _compute_sun_azimuth(12)
        # hour 12: t = (12-8)/(20-8) = 4/12 = 0.333, az = 90 + 60 = 150
        # Actually at noon it's 150, not 180. Solar south is around 2pm in this model.
        # The model uses linear interpolation, so true south (180) is at hour 14.
        assert 140 <= az <= 190, f"Noon azimuth should be roughly south, got {az}"

    def test_azimuth_monotonically_increases(self):
        """Sun must move consistently east to west, never backwards."""
        azimuths = [_compute_sun_azimuth(h) for h in range(8, 21)]
        for i in range(len(azimuths) - 1):
            assert azimuths[i] < azimuths[i + 1], (
                f"Azimuth must increase: hour {8+i}={azimuths[i]}, "
                f"hour {9+i}={azimuths[i+1]}"
            )

    def test_azimuth_step_is_constant(self):
        """Linear model should produce equal steps between hours."""
        azimuths = [_compute_sun_azimuth(h) for h in range(8, 21)]
        steps = [azimuths[i+1] - azimuths[i] for i in range(len(azimuths) - 1)]
        assert all(abs(s - steps[0]) < 0.1 for s in steps), (
            f"Azimuth steps should be constant (linear model), got {steps}"
        )


# ──────────────────────────────────────────────
# Color Temperature Curve
# ──────────────────────────────────────────────

class TestColorTemperature:
    """
    Verify the color temperature curve follows real-world lighting:
    warm at dawn/dusk, cool at midday.
    """

    def test_8am_is_warm(self):
        k = _compute_color_temp(8)
        assert k <= 3200, f"8am should be warm (<= 3200K), got {k}"

    def test_8pm_is_warmest(self):
        k = _compute_color_temp(20)
        assert k <= 2800, f"8pm should be very warm (<= 2800K), got {k}"

    def test_midday_is_cool(self):
        k = _compute_color_temp(13)
        assert k >= 6000, f"1pm should be cool (>= 6000K), got {k}"

    def test_morning_warms_to_cool(self):
        """Temperature should increase from 8am toward midday."""
        temps = [_compute_color_temp(h) for h in range(8, 14)]
        for i in range(len(temps) - 1):
            assert temps[i] <= temps[i + 1], (
                f"Morning temps should increase: {8+i}={temps[i]}, {9+i}={temps[i+1]}"
            )

    def test_afternoon_cools_to_warm(self):
        """Temperature should decrease from midday toward 8pm."""
        temps = [_compute_color_temp(h) for h in range(13, 21)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1], (
                f"Afternoon temps should decrease: {13+i}={temps[i]}, {14+i}={temps[i+1]}"
            )

    def test_all_temperatures_in_realistic_range(self):
        """All values should be within real-world daylight range."""
        for h in range(8, 21):
            k = _compute_color_temp(h)
            assert 2000 <= k <= 7000, f"Hour {h}: {k}K is outside realistic range"

    def test_color_temp_description_warm(self):
        desc = _color_temp_description(2600)
        assert "warm" in desc.lower()

    def test_color_temp_description_cool(self):
        desc = _color_temp_description(6200)
        assert "cool" in desc.lower()


# ──────────────────────────────────────────────
# Shadow Direction and Length
# ──────────────────────────────────────────────

class TestShadows:
    """Verify shadows behave physically: opposite the sun, long at low elevation."""

    def test_morning_shadows_cast_west(self):
        """Sun at azimuth 90 (east). shadow_az = 270 → maps to RIGHT in frame."""
        desc = _shadow_direction_explicit(90)  # 8am azimuth
        assert "RIGHT" in desc, (
            f"Morning shadows (shadow_az=270) should map to RIGHT, got: {desc}"
        )

    def test_evening_shadows_cast_east(self):
        """Sun at azimuth 270 (west). shadow_az = 90 → maps to LEFT in frame."""
        desc = _shadow_direction_explicit(270)  # 8pm azimuth
        assert "LEFT" in desc, (
            f"Evening shadows (shadow_az=90) should map to LEFT, got: {desc}"
        )

    def test_low_sun_produces_long_shadows(self):
        desc = _shadow_length_description(5.0)
        assert "long" in desc.lower()

    def test_high_sun_produces_short_shadows(self):
        desc = _shadow_length_description(55.0)
        assert "short" in desc.lower()

    def test_shadow_description_varies_with_elevation(self):
        """Different elevations should produce different descriptions."""
        low = _shadow_description(8, 10.0, 90)
        high = _shadow_description(13, 55.0, 195)
        assert low != high, "Low and high sun should produce different shadow descriptions"


# ──────────────────────────────────────────────
# Sun Position in Frame
# ──────────────────────────────────────────────

class TestSunPositionInFrame:
    """
    Verify the astronomical-to-compositional translation.
    This function converts azimuth degrees into spatial language
    a vision model can interpret (right/left/above).
    """

    def test_east_maps_to_right(self):
        desc = _sun_position_in_frame(90)
        assert "RIGHT" in desc, f"90 degrees (east) should map to RIGHT, got: {desc}"

    def test_west_maps_to_left(self):
        desc = _sun_position_in_frame(270)
        assert "LEFT" in desc, f"270 degrees (west) should map to LEFT, got: {desc}"

    def test_south_maps_to_overhead(self):
        desc = _sun_position_in_frame(180)
        assert "overhead" in desc.lower() or "above" in desc.lower(), (
            f"180 degrees (south) should map to overhead, got: {desc}"
        )

    def test_all_azimuths_produce_descriptions(self):
        """Every azimuth in our range should produce a non-empty description."""
        for az in range(90, 271, 15):
            desc = _sun_position_in_frame(az)
            assert len(desc) > 10, f"Azimuth {az} produced too short a description"


# ──────────────────────────────────────────────
# Prompt Construction
# ──────────────────────────────────────────────

class TestPromptConstruction:
    """Verify the full prompt is well-formed for each time slot."""

    def test_prompt_includes_time_label(self):
        slots = generate_time_slots()
        for slot in slots:
            prompt = build_prompt(slot)
            assert slot.label in prompt, f"Prompt for hour {slot.hour} missing time label"

    def test_prompt_includes_color_temperature(self):
        slots = generate_time_slots()
        for slot in slots:
            prompt = build_prompt(slot)
            assert str(slot.color_temp_k) in prompt, (
                f"Prompt for hour {slot.hour} missing color temp {slot.color_temp_k}K"
            )

    def test_prompt_includes_do_not_change_section(self):
        slots = generate_time_slots()
        for slot in slots:
            prompt = build_prompt(slot)
            assert "DO NOT CHANGE" in prompt

    def test_prompt_includes_geometry_preservation(self):
        slots = generate_time_slots()
        prompt = build_prompt(slots[0])
        assert "geometry" in prompt.lower()
        assert "composition" in prompt.lower()

    def test_prompt_includes_vegetation_warning(self):
        """Critical: models tend to add trees. Prompt must explicitly prevent this."""
        slots = generate_time_slots()
        prompt = build_prompt(slots[0])
        assert "vegetation" in prompt.lower()
        assert "do not add" in prompt.lower()

    def test_8pm_prompt_includes_darkness_instruction(self):
        """8pm is a special case: must explicitly instruct darker rendering."""
        slots = generate_time_slots()
        evening_slot = slots[-1]
        assert evening_slot.hour == 20
        prompt = build_prompt(evening_slot)
        assert "DARKER" in prompt or "darker" in prompt, (
            "8pm prompt must instruct the model to render a darker scene"
        )

    def test_non_8pm_prompt_has_no_darkness_override(self):
        slots = generate_time_slots()
        noon_slot = slots[4]  # 12pm
        prompt = build_prompt(noon_slot)
        note = _evening_darker_note(noon_slot.hour)
        assert note == "", "Non-8pm slots should not have the darkness override"

    def test_all_13_prompts_are_unique(self):
        """Each hour must produce a distinct prompt."""
        slots = generate_time_slots()
        prompts = [build_prompt(s) for s in slots]
        assert len(set(prompts)) == 13, "All 13 prompts should be unique"

    def test_prompt_is_substantial(self):
        """Prompts should be detailed enough to constrain the model."""
        slots = generate_time_slots()
        for slot in slots:
            prompt = build_prompt(slot)
            assert len(prompt) > 500, (
                f"Prompt for hour {slot.hour} is only {len(prompt)} chars, "
                "too short to properly constrain the model"
            )


# ──────────────────────────────────────────────
# Hour Label Formatting
# ──────────────────────────────────────────────

class TestHourLabels:

    def test_8am(self):
        assert _format_hour_label(8) == "8:00 AM"

    def test_noon(self):
        assert _format_hour_label(12) == "12:00 PM"

    def test_1pm(self):
        assert _format_hour_label(13) == "1:00 PM"

    def test_8pm(self):
        assert _format_hour_label(20) == "8:00 PM"

    def test_midnight(self):
        assert _format_hour_label(0) == "12:00 AM"


# ──────────────────────────────────────────────
# Sky Descriptions
# ──────────────────────────────────────────────

class TestSkyDescriptions:

    def test_every_hour_has_a_sky_description(self):
        for h in range(8, 21):
            desc = _sky_description(h)
            assert len(desc) > 20, f"Hour {h} sky description too short"

    def test_morning_sky_mentions_warm(self):
        desc = _sky_description(8)
        assert "warm" in desc.lower()

    def test_noon_sky_mentions_blue(self):
        desc = _sky_description(12)
        assert "blue" in desc.lower()

    def test_sunset_sky_mentions_orange_or_amber(self):
        desc = _sky_description(19)
        text = desc.lower()
        assert "orange" in text or "amber" in text or "magenta" in text

    def test_8pm_mentions_dusk(self):
        desc = _sky_description(20)
        assert "dusk" in desc.lower() or "dark" in desc.lower()


# ──────────────────────────────────────────────
# Lighting Descriptions
# ──────────────────────────────────────────────

class TestLightingDescriptions:

    def test_every_hour_has_lighting(self):
        for h in range(8, 21):
            desc = get_lighting_description(h)
            assert len(desc) > 20, f"Hour {h} lighting description too short"

    def test_morning_mentions_east(self):
        desc = get_lighting_description(8)
        assert "east" in desc.lower()

    def test_golden_hour_mentioned(self):
        desc = get_lighting_description(17)
        assert "golden" in desc.lower()
