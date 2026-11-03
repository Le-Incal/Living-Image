"""
Prompt v3 for Living Image, and the variants we benchmark against it.

v3 keeps v2's single solar model (40N, June solstice, solar noon at
1 PM, north-facing camera, frame language only) and answers the
second-round reviews of v2:

- Direction in finer steps. v2's eight 45-degree sectors gave 9, 10 and
  11 AM identical words. v3 states a camera-relative clock reading
  (half-hour steps, 15 degrees) plus words graded "slightly", "partly"
  and "halfway", so every hour reads differently.
- AHEAD and BEHIND are defined relative to the camera.
- The render's baked-in lighting is replaced, not layered over.
- Below 8 degrees, shadows are long but soft-edged and low-contrast,
  and the image is darker.
- Shadow length and color come from the unrounded elevation.
- Windows keep their original lit or unlit state, so frames generated
  independently cannot blink.

Variants:
    v2           previous prompt, unchanged (prompts.py), the baseline
    v3           v3 wording, even sun motion (15 degrees per hour)
    v3-true-sun  v3 wording, true solar azimuth
    v3-glow      v3 wording, even warm window glow below 20 degrees
"""

from __future__ import annotations

import math

import prompts
from prompts import (
    FIRST_HOUR, LAST_HOUR, LATITUDE, DECLINATION, SOLAR_NOON_HOUR, TimeSlot,
    _format_hour_label, _hour_angle, _color_temp_description, _sky_description,
    get_lighting_description,
)

VARIANTS = ("v2", "v3", "v3-true-sun", "v3-glow")
DEFAULT_VARIANT = "v3"

GLOW_BELOW_ELEVATION = 20.0
LOW_SUN_ELEVATION = 8.0


PROMPT_TEMPLATE = """Relight this architectural image to show it at {time_label} on a clear day. Change only the light. Keep everything else exactly as it is.

DIRECTIONS are relative to the camera: AHEAD means into the scene, BEHIND means behind the camera, LEFT and RIGHT are the frame sides. On a clock seen from above, 12 is straight ahead, 3 is right, 6 is directly behind the camera, 9 is left.

SUN AND SHADOWS:
- Sunlight comes {light_phrase} (from about {light_clock}), {sun_elevation}° above the horizon. Do not draw the sun itself.
- Shadows fall {shadow_phrase} (toward about {shadow_clock}). Building, tree, and ground shadows all point the same way.
- Shadow length: {shadow_length_description}.
{low_sun_note}- The original render has its own lighting: replace its sun, shadows, and highlights with this light, but keep contact shadows and shading in corners.

LIGHT:
- {lighting_description}
- Color temperature: {color_temp_description} ({color_temp_k}K)
- Sky: {sky_description}
- {interior_lighting_state}

CHANGE ONLY:
- Light direction, intensity, and color
- Shadow direction, length, and softness
- Sky color and brightness
- Reflections and specular highlights

DO NOT CHANGE:
- Building geometry, proportions, or structural lines
- Window positions, sizes, or shapes
- Materials, textures, reflectivity, or surface details
- Vegetation: keep every tree trunk and canopy outline in place. Do not add trees, plants, or shrubs; open areas stay empty
- Ground materials and layout
- Camera angle, focal length, and framing. Do not crop, zoom, pan, or recenter
- White balance: as in the original; tones change only with the light
- Text, signage, or labels
- The number or placement of objects

This is a professional architectural visualization. Structural accuracy matters more than drama."""


# ──────────────────────────────────────────────
# Solar model
# ──────────────────────────────────────────────

def exact_elevation(hour: float) -> float:
    """Unrounded sun elevation in degrees; zero below the horizon."""
    lat, dec = math.radians(LATITUDE), math.radians(DECLINATION)
    h = math.radians(_hour_angle(hour))
    s = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    return max(math.degrees(math.asin(max(-1.0, min(1.0, s)))), 0.0)


def even_azimuth(hour: float) -> float:
    """Hour-angle approximation: due south at solar noon, 15 degrees per hour."""
    return 180.0 + _hour_angle(hour)


def true_azimuth(hour: float) -> float:
    """True solar azimuth in degrees from north, clockwise."""
    lat, dec = math.radians(LATITUDE), math.radians(DECLINATION)
    h = math.radians(_hour_angle(hour))
    a = math.degrees(math.atan2(-math.sin(h), math.tan(dec) * math.cos(lat) - math.sin(lat) * math.cos(h)))
    return a % 360


def shadow_ratio(hour: float) -> float:
    """Shadow length as a multiple of object height, from the exact elevation."""
    el = exact_elevation(hour)
    return math.inf if el <= 0 else 1.0 / math.tan(math.radians(el))


def _color_temp(hour: float) -> int:
    s = math.sin(math.radians(exact_elevation(hour)))
    return int(round((2200 + 3600 * (s ** 0.75)) / 50.0) * 50)


# ──────────────────────────────────────────────
# Frame language
# ──────────────────────────────────────────────

_CARDINALS = ("AHEAD", "RIGHT", "BEHIND", "LEFT")  # 0, 90, 180, 270 relative to the camera

_LIGHT_PURE = {
    "AHEAD": "from in front of the camera, so the building is backlit",
    "RIGHT": "from the RIGHT side of the frame",
    "BEHIND": "from directly behind the camera",
    "LEFT": "from the LEFT side of the frame",
}
_LIGHT_FROM = {
    "AHEAD": "in front of the camera",
    "RIGHT": "the RIGHT side of the frame",
    "BEHIND": "behind the camera",
    "LEFT": "the LEFT side of the frame",
}
_LIGHT_LEAN = {
    "AHEAD": "in front of the camera",
    "RIGHT": "to the RIGHT",
    "BEHIND": "behind the camera",
    "LEFT": "to the LEFT",
}

_SHADOW_PURE = {
    "AHEAD": "straight away from the camera, into the scene",
    "RIGHT": "toward the RIGHT side of the frame",
    "BEHIND": "straight toward the camera",
    "LEFT": "toward the LEFT side of the frame",
}
_SHADOW_TOWARD = {
    "AHEAD": "away from the camera, into the scene",
    "RIGHT": "toward the RIGHT side of the frame",
    "BEHIND": "toward the camera",
    "LEFT": "toward the LEFT side of the frame",
}
_SHADOW_LEAN = {
    "AHEAD": "away from the camera",
    "RIGHT": "to the RIGHT",
    "BEHIND": "toward the camera",
    "LEFT": "to the LEFT",
}


def _decompose(bearing: float) -> tuple[str, str, float]:
    """Nearest camera cardinal, the neighbor it leans toward, and the lean in degrees."""
    b = bearing % 360
    i = int(math.floor((b + 45) / 90)) % 4
    d = ((b - i * 90 + 180) % 360) - 180          # signed offset in [-45, 45)
    j = (i + (1 if d > 0 else -1)) % 4
    return _CARDINALS[i], _CARDINALS[j], abs(d)


def light_phrase(bearing: float) -> str:
    """Where sunlight comes from, for a camera-relative bearing."""
    main, lean, d = _decompose(bearing)
    if d <= 7.5:
        return _LIGHT_PURE[main]
    if d <= 22.5:
        return f"from {_LIGHT_FROM[main]}, slightly {_LIGHT_LEAN[lean]}"
    if d <= 37.5:
        return f"from {_LIGHT_FROM[main]} and partly {_LIGHT_LEAN[lean]}"
    return f"from halfway between {_LIGHT_FROM[main]} and {_LIGHT_FROM[lean]}"


def shadow_phrase(bearing: float) -> str:
    """Where shadows fall, for the camera-relative bearing they point toward."""
    main, lean, d = _decompose(bearing)
    if d <= 7.5:
        return _SHADOW_PURE[main]
    if d <= 22.5:
        return f"{_SHADOW_TOWARD[main]}, angled slightly {_SHADOW_LEAN[lean]}"
    if d <= 37.5:
        return f"{_SHADOW_TOWARD[main]}, angled partly {_SHADOW_LEAN[lean]}"
    return f"diagonally, {_SHADOW_LEAN[main]} and {_SHADOW_LEAN[lean]}"


def clock_reading(bearing: float) -> str:
    """Camera-relative bearing as a clock reading in half-hour steps (12 = ahead)."""
    halves = int(math.floor((bearing % 360) / 15.0 + 0.5)) % 24
    hour, half = divmod(halves, 2)
    return f"{hour or 12}:{'30' if half else '00'}"


def _shadow_length_description(hour: float) -> str:
    r = shadow_ratio(hour)
    if r >= 10:
        return "very long, more than ten times the height of the objects casting them"
    if r >= 3:
        return f"long, about {r:.0f} times the height of the objects casting them"
    if r >= 1:
        return f"medium, about {r:.1f} times the height of the objects casting them"
    return f"short, about {r:.1f} times the height of the objects casting them"


def _low_sun_note(el: float) -> str:
    if el >= LOW_SUN_ELEVATION:
        return ""
    return (
        "- Low sun: long, soft-edged, low-contrast shadows. The image is noticeably "
        "darker than midday, with a dimmer sky and surfaces.\n"
    )


_WINDOWS_KEEP = "Windows: keep each window lit or unlit exactly as in the original."
_WINDOWS_GLOW = "Windows: every window shows the same soft, warm interior glow, evenly on all floors."


def _interior_lighting_state(el: float, variant: str) -> str:
    if variant == "v3-glow" and el < GLOW_BELOW_ELEVATION:
        return _WINDOWS_GLOW
    return _WINDOWS_KEEP


# ──────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────

def _check(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")


def generate_slots(variant: str = DEFAULT_VARIANT) -> list[TimeSlot]:
    """The 13 hourly slots for a variant."""
    _check(variant)
    if variant == "v2":
        return prompts.generate_time_slots()
    azimuth = true_azimuth if variant == "v3-true-sun" else even_azimuth
    slots = []
    for hour in range(FIRST_HOUR, LAST_HOUR + 1):
        el = exact_elevation(hour)
        az = azimuth(hour)
        slots.append(TimeSlot(
            hour=hour,
            label=_format_hour_label(hour),
            sun_elevation=round(el, 1),
            sun_azimuth=round(az, 1),
            color_temp_k=_color_temp(hour),
            shadow_description=f"Shadows fall {shadow_phrase(az + 180)}. "
                               f"Length: {_shadow_length_description(hour)}.",
            sky_description=_sky_description(hour),
            lighting_description=get_lighting_description(hour),
        ))
    return slots


def build(slot: TimeSlot, variant: str = DEFAULT_VARIANT) -> str:
    """The full prompt for one slot under a variant."""
    _check(variant)
    if variant == "v2":
        return prompts.build_prompt(slot)
    az = slot.sun_azimuth
    return PROMPT_TEMPLATE.format(
        time_label=slot.label,
        light_phrase=light_phrase(az),
        light_clock=clock_reading(az),
        shadow_phrase=shadow_phrase(az + 180),
        shadow_clock=clock_reading(az + 180),
        sun_elevation=slot.sun_elevation,
        shadow_length_description=_shadow_length_description(slot.hour),
        low_sun_note=_low_sun_note(slot.sun_elevation),
        lighting_description=slot.lighting_description,
        color_temp_description=_color_temp_description(slot.color_temp_k),
        color_temp_k=slot.color_temp_k,
        sky_description=slot.sky_description,
        interior_lighting_state=_interior_lighting_state(slot.sun_elevation, variant),
    )
