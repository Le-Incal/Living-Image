"""
Prompt template system for Living Image.

Generates time-specific prompts (8am-8pm) that instruct image models to
relight an architectural image while preserving structural fidelity.

Every statement in a prompt is derived from one solar model, so the
prompt cannot contradict itself:

- Solar position: 40N on the June solstice, solar noon at 1 PM (daylight
  saving time). Elevation uses the standard spherical formula. Azimuth
  uses the hour-angle approximation (15 degrees per hour from due south),
  which keeps the sun's motion across the frame even.
- Frame convention: the camera faces north. East is frame RIGHT, west is
  frame LEFT, south is behind the camera, north is ahead, into the scene.
  The prompt itself never names compass directions, because the real
  orientation of the building is unknown.
- Shadows fall opposite the light; their length is cot(elevation).
- Color temperature, lighting, and sky text follow elevation, not the clock.
"""

from dataclasses import dataclass
import math

# Time window: 8am (8) to 8pm (20) inclusive = 13 one-hour slots
FIRST_HOUR = 8
LAST_HOUR = 20

LATITUDE = 40.0          # degrees north
DECLINATION = 23.44      # June solstice
SOLAR_NOON_HOUR = 13     # 1 PM local daylight time


PROMPT_TEMPLATE = """Relight this architectural image to show how it looks at {time_label} on a clear day. Change the light. Keep everything else.

SUN AND SHADOWS at {time_label}:
- Sunlight comes {light_phrase}, {sun_elevation}° above the horizon. Do not draw the sun itself.
- Shadows fall {shadow_phrase}. Building, tree, and ground shadows all point the same way.
- Shadow length: {shadow_length_description}.
{evening_darker_note}
LIGHT:
- {lighting_description}
- Color temperature: {color_temp_description} ({color_temp_k}K)
- Sky: {sky_description}

CHANGE ONLY:
- Light direction, intensity, and color temperature
- Shadow direction, length, and softness
- Sky color and brightness
- Reflections and specular highlights

DO NOT CHANGE:
- Building geometry, proportions, or structural lines
- Window positions, sizes, or shapes
- Materials, textures, or surface details
- Vegetation: do not add trees, plants, or shrubs; empty courtyards and open areas stay empty
- Ground materials and layout
- Camera angle, focal length, and composition
- Text, signage, or labels
- The number or placement of objects

This is a professional architectural visualization. Structural accuracy matters more than drama."""


# ──────────────────────────────────────────────
# Frame language
# ──────────────────────────────────────────────

# Eight 45-degree sectors, as bearings relative to a north-facing camera.
_SECTORS = [
    "AHEAD", "AHEAD_RIGHT", "RIGHT", "BEHIND_RIGHT",
    "BEHIND", "BEHIND_LEFT", "LEFT", "AHEAD_LEFT",
]

OPPOSITE_SIDE = {s: _SECTORS[(i + 4) % 8] for i, s in enumerate(_SECTORS)}

LIGHT_PHRASES = {
    "AHEAD": "from in front of the camera, so the building is backlit",
    "AHEAD_RIGHT": "from in front of the camera and to the RIGHT",
    "RIGHT": "from the RIGHT side of the frame",
    "BEHIND_RIGHT": "from behind the camera and to the RIGHT",
    "BEHIND": "from behind the camera",
    "BEHIND_LEFT": "from behind the camera and to the LEFT",
    "LEFT": "from the LEFT side of the frame",
    "AHEAD_LEFT": "from in front of the camera and to the LEFT",
}

SHADOW_PHRASES = {
    "AHEAD": "straight away from the camera, into the scene",
    "AHEAD_RIGHT": "away from the camera and to the RIGHT",
    "RIGHT": "toward the RIGHT side of the frame",
    "BEHIND_RIGHT": "toward the camera and to the RIGHT",
    "BEHIND": "toward the camera",
    "BEHIND_LEFT": "toward the camera and to the LEFT",
    "LEFT": "toward the LEFT side of the frame",
    "AHEAD_LEFT": "away from the camera and to the LEFT",
}


def _side(bearing: float) -> str:
    """Map a compass bearing to a frame sector for a north-facing camera."""
    return _SECTORS[int(((bearing % 360) + 22.5) // 45) % 8]


def _light_side(sun_azimuth: float) -> str:
    """Frame side the sunlight comes from."""
    return _side(sun_azimuth)


def _shadow_side(sun_azimuth: float) -> str:
    """Frame side shadows fall toward: always opposite the light."""
    return _side(sun_azimuth + 180)


# ──────────────────────────────────────────────
# Solar model
# ──────────────────────────────────────────────

@dataclass
class TimeSlot:
    """Represents one hour in the 8am-8pm range with lighting metadata."""
    hour: int               # 24h format: 8-20
    label: str              # "8:00 AM", "12:00 PM", "8:00 PM", etc.
    sun_elevation: float    # degrees above horizon
    sun_azimuth: float      # degrees from north (90=E, 180=S, 270=W)
    color_temp_k: int       # Kelvin
    shadow_description: str
    sky_description: str
    lighting_description: str


def _format_hour_label(hour: int) -> str:
    """Convert 24h hour to readable label like '7:00 AM'."""
    if hour == 0 or hour == 24:
        return "12:00 AM"
    elif hour == 12:
        return "12:00 PM"
    elif hour < 12:
        return f"{hour}:00 AM"
    else:
        return f"{hour - 12}:00 PM"


def _hour_angle(hour: float) -> float:
    return 15.0 * (hour - SOLAR_NOON_HOUR)


def _compute_sun_elevation(hour: int) -> float:
    """Sun elevation in degrees at 40N on the June solstice. Zero below the horizon."""
    lat = math.radians(LATITUDE)
    dec = math.radians(DECLINATION)
    h = math.radians(_hour_angle(hour))
    sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    return round(max(elevation, 0.0), 1)


def _compute_sun_azimuth(hour: int) -> float:
    """
    Sun azimuth in degrees from north, by the hour-angle approximation:
    due south (180) at solar noon, 15 degrees per hour. Even steps keep
    the sun's motion across the frame uniform from slot to slot.
    """
    return round(180.0 + _hour_angle(hour), 0)


def _compute_color_temp(hour: int) -> int:
    """
    Kelvin from sun elevation: ~2200K at the horizon, ~5700K at the
    solstice noon peak. Lower sun is always warmer, at any hour.
    """
    el = _compute_sun_elevation(hour)
    s = math.sin(math.radians(el))
    kelvin = 2200 + 3600 * (s ** 0.75)
    return int(round(kelvin / 50.0) * 50)


def _color_temp_description(kelvin: int) -> str:
    """Human-readable color temperature description."""
    if kelvin < 3000:
        return "warm amber/orange"
    elif kelvin < 3800:
        return "warm golden"
    elif kelvin < 4500:
        return "warm neutral"
    elif kelvin < 5200:
        return "neutral daylight"
    elif kelvin < 6000:
        return "cool neutral daylight"
    else:
        return "cool bright daylight"


# ──────────────────────────────────────────────
# Shadows
# ──────────────────────────────────────────────

def _shadow_length_ratio(sun_elevation: float) -> float:
    """Shadow length as a multiple of object height: cot(elevation)."""
    if sun_elevation <= 0:
        return math.inf
    return 1.0 / math.tan(math.radians(sun_elevation))


def _shadow_length_description(sun_elevation: float) -> str:
    """Shadow length in words and as a multiple of object height."""
    r = _shadow_length_ratio(sun_elevation)
    if r >= 10:
        return "very long, more than ten times the height of the objects casting them"
    if r >= 3:
        return f"long, about {r:.0f} times the height of the objects casting them"
    if r >= 1:
        return f"medium, about {r:.1f} times the height of the objects casting them"
    return f"short, about {r:.1f} times the height of the objects casting them"


def _sun_position_in_frame(sun_azimuth: float) -> str:
    """Where the sunlight comes from, in frame language."""
    return f"Sunlight comes {LIGHT_PHRASES[_light_side(sun_azimuth)]}."


def _shadow_direction_explicit(sun_azimuth: float) -> str:
    """Where shadows fall, in frame language. Always opposite the light."""
    return f"Shadows fall {SHADOW_PHRASES[_shadow_side(sun_azimuth)]}"


def _shadow_description(hour: int, sun_elevation: float, sun_azimuth: float) -> str:
    """Direction plus length, derived from the same sun."""
    return (
        f"{_shadow_direction_explicit(sun_azimuth)}. "
        f"Length: {_shadow_length_description(sun_elevation)}."
    )


# ──────────────────────────────────────────────
# Light and sky, derived from elevation
# ──────────────────────────────────────────────

def get_lighting_description(hour: int) -> str:
    """Natural-language lighting conditions, derived from sun elevation."""
    el = _compute_sun_elevation(hour)
    morning = hour < SOLAR_NOON_HOUR
    if el < 8:
        word = "Dawn" if morning else "Dusk"
        return f"{word} light from a sun just above the horizon. Deep warm tones, low contrast, fading brightness."
    if el < 20:
        word = "Early morning" if morning else "Evening"
        return f"{word}. Golden light from a low sun. Warm, raking, strongly directional."
    if el < 40:
        word = "Morning" if morning else "Late afternoon"
        return f"{word} light from a sun partway up the sky. Clear, directional, warm-neutral."
    if hour == SOLAR_NOON_HOUR:
        return "Solar noon. The sun is at its peak for the day. Bright, crisp, high-contrast light."
    word = "Late morning" if morning else "Early afternoon"
    return f"{word} light from a high sun. Bright, crisp, high-contrast."


def _sky_description(hour: int) -> str:
    """Sky appearance, derived from sun elevation."""
    el = _compute_sun_elevation(hour)
    if el < 8:
        return "Deep orange and coral at the horizon, fading to dark blue above. Much dimmer than midday."
    if el < 20:
        return "Warm amber and orange near the horizon, clear blue above."
    if el < 40:
        return "Clear blue sky with warm tones near the horizon."
    return "Bright, deep blue sky, slightly paler near the horizon."


def _evening_darker_note(hour: int) -> str:
    """Only when the sun is low: an explicit instruction to darken the image."""
    el = _compute_sun_elevation(hour)
    if el >= 8:
        return ""
    word = "Early morning" if hour < SOLAR_NOON_HOUR else "Evening"
    return f"- {word}: the image must be noticeably darker than midday, with dimmer sky and surfaces.\n"


# ──────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────

def generate_time_slots() -> list[TimeSlot]:
    """Generate TimeSlots from 8am to 8pm (13 one-hour slots)."""
    slots = []
    for hour in range(FIRST_HOUR, LAST_HOUR + 1):
        sun_el = _compute_sun_elevation(hour)
        sun_az = _compute_sun_azimuth(hour)
        slots.append(TimeSlot(
            hour=hour,
            label=_format_hour_label(hour),
            sun_elevation=sun_el,
            sun_azimuth=sun_az,
            color_temp_k=_compute_color_temp(hour),
            shadow_description=_shadow_description(hour, sun_el, sun_az),
            sky_description=_sky_description(hour),
            lighting_description=get_lighting_description(hour),
        ))
    return slots


def build_prompt(slot: TimeSlot) -> str:
    """Build a complete API prompt for a given time slot."""
    return PROMPT_TEMPLATE.format(
        time_label=slot.label,
        light_phrase=LIGHT_PHRASES[_light_side(slot.sun_azimuth)],
        shadow_phrase=SHADOW_PHRASES[_shadow_side(slot.sun_azimuth)],
        shadow_length_description=_shadow_length_description(slot.sun_elevation),
        sun_elevation=slot.sun_elevation,
        evening_darker_note=_evening_darker_note(slot.hour),
        lighting_description=slot.lighting_description,
        color_temp_description=_color_temp_description(slot.color_temp_k),
        color_temp_k=slot.color_temp_k,
        sky_description=slot.sky_description,
    )
