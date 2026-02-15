"""
Prompt template system for Living Image crossfade test harness.

Generates time-specific prompts (8am-8pm) that instruct image generation
APIs to relight an architectural image while preserving structural fidelity.
Sun position (elevation and azimuth) progresses evenly across the day.
"""

from dataclasses import dataclass
import math

# Time window: 8am (8) to 8pm (20) inclusive = 13 one-hour slots
FIRST_HOUR = 8
LAST_HOUR = 20


PROMPT_TEMPLATE = """Relight this architectural image to accurately depict how it would look at {time_label} on a clear day.

LIGHTING CONDITIONS at {time_label}:
- {lighting_description}
- Sun elevation: approximately {sun_elevation} degrees above the horizon
- Sun azimuth: approximately {sun_azimuth} degrees (0=north, 90=east, 180=south, 270=west) — light and shadows align with this direction
- Light color temperature: {color_temp_description} ({color_temp_k}K)
- Shadows: {shadow_description}
- Sky: {sky_description}

CHANGE ONLY these elements:
- Sun position, lighting direction, and intensity
- Shadow angles, lengths, and softness
- Sky color gradient and brightness
- Ambient light color temperature on all surfaces
- Surface reflections and specular highlights consistent with {time_label} sunlight

DO NOT CHANGE any of these elements:
- Building geometry, proportions, or structural lines
- Window positions, sizes, or shapes
- Material textures, patterns, or surface details
- Vegetation shapes, positions, or density
- Ground materials and layout
- Camera angle, focal length, and composition
- Any text, signage, or labels in the image
- The number or placement of any objects

Preserve the EXACT architectural composition. This is a professional architectural visualization where structural accuracy is critical."""


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


def _compute_sun_elevation(hour: int) -> float:
    """
    Sun elevation for ~40N, long summer day. Sunrise before 8am, sunset after 8pm.
    Elevation progresses smoothly (sine) so sun moves evenly across the sky.
    Solar noon ~13:30.
    """
    sunrise = 6.0
    sunset = 20.5
    day_length = sunset - sunrise
    max_elevation = 58.0  # ~40N summer

    if hour <= sunrise or hour >= sunset:
        return 0.0

    fraction = (hour - sunrise) / day_length
    elevation = max_elevation * math.sin(math.pi * fraction)
    return round(elevation, 1)


def _compute_sun_azimuth(hour: int) -> float:
    """
    Sun azimuth in degrees from north. Progresses evenly: 8am ~90 (E), noon ~180 (S), 8pm ~270 (W).
    Linear in hour across the 8am-8pm window for consistent motion across the image.
    """
    # Map hour 8 -> 90°, hour 20 -> 270° (linear)
    t = (hour - FIRST_HOUR) / (LAST_HOUR - FIRST_HOUR)  # 0 at 8am, 1 at 8pm
    azimuth = 90 + t * 180  # 90 (E) to 270 (W)
    return round(azimuth, 0)


def _compute_color_temp(hour: int) -> int:
    """
    Color temperature in Kelvin across the day (8am-8pm).
    Warm at 8am/8pm, cool at midday.
    """
    temps = {
        8:  3000,
        9:  3800,
        10: 4500,
        11: 5300,
        12: 5900,
        13: 6200,
        14: 6000,
        15: 5500,
        16: 4800,
        17: 4000,
        18: 3300,
        19: 2900,
        20: 2600,
    }
    return temps.get(hour, 5500)


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


def _shadow_description(hour: int, sun_elevation: float, sun_azimuth: float) -> str:
    """Describe shadow direction and length from sun position (even progression 8am-8pm)."""
    if sun_elevation <= 5:
        return "Very long shadows in the direction opposite the sun. Soft, diffuse shadow edges from low-angle light."
    if sun_elevation <= 15:
        return "Long shadows opposite the sun azimuth. Moderately soft shadow edges."
    if sun_elevation <= 35:
        return "Medium-length shadows opposite the sun. Clearly defined shadow edges."
    if sun_elevation <= 55:
        return "Short shadows nearly directly opposite the sun. Sharp, well-defined shadow edges from high sun."
    return "Short shadows directly opposite the sun. Sharp shadow edges from overhead sun."


def _sky_description(hour: int) -> str:
    """Describe sky appearance at given hour (8am-8pm)."""
    descs = {
        8:  "Brightening sky with warm yellow tones near the horizon transitioning to clear blue above.",
        9:  "Clear blue sky with slight warmth near the horizon. Bright and luminous.",
        10: "Deep clear blue sky. Uniform brightness with minimal horizon haze.",
        11: "Bright, saturated blue sky. Even illumination across the dome.",
        12: "Intense blue overhead, slightly lighter near the horizon. Maximum brightness.",
        13: "Bright blue sky at peak intensity.",
        14: "Clear blue sky, beginning almost imperceptibly to warm.",
        15: "Blue sky with subtle warm shift. Light becoming slightly more directional.",
        16: "Sky transitioning to warmer blue. Noticeable golden quality to the light.",
        17: "Golden hour sky. Warm amber-gold tones from the western horizon.",
        18: "Late golden hour. Rich amber and coral near the western horizon.",
        19: "Sunset sky. Deep orange and magenta at the horizon fading to deep blue above.",
        20: "Dusk. Deep orange, coral, and purple at the horizon. Low ambient light.",
    }
    return descs.get(hour, "Clear blue sky.")


def get_lighting_description(hour: int) -> str:
    """Get a natural-language description of lighting conditions (8am-8pm)."""
    descs = {
        8:  "Morning light. Sun low in the east. Warm directional light with long shadows to the west.",
        9:  "Mid-morning. Sun climbing in the east. Balanced warm-neutral light with clear shadows.",
        10: "Late morning. Strong directional sun from the east. Neutral bright daylight.",
        11: "Approaching midday. Sun high in the sky. Bright, even illumination with short shadows.",
        12: "Noon. Sun near its highest. Maximum brightness with minimal shadows.",
        13: "Early afternoon. Sun beginning westward descent. Bright directional light.",
        14: "Mid-afternoon. Sun in the west-southwest. Clear directional light with lengthening shadows.",
        15: "Afternoon. Sun lowering in the west. Warm directional light.",
        16: "Late afternoon. Noticeably warm golden light with long shadows to the east.",
        17: "Golden hour. Low sun in the west. Rich golden-amber light. Long dramatic shadows.",
        18: "Late golden hour. Very low western sun. Warm amber light on all surfaces.",
        19: "Sunset. Very low sun near the horizon. Deep warm light. Extremely long shadows.",
        20: "Dusk. Sun at or just below horizon. Deep warm tones. Very long shadows.",
    }
    return descs.get(hour, "Standard daylight conditions.")


def generate_time_slots() -> list[TimeSlot]:
    """Generate TimeSlots from 8am to 8pm (13 one-hour slots). Sun position moves evenly."""
    slots = []
    for hour in range(FIRST_HOUR, LAST_HOUR + 1):
        sun_el = _compute_sun_elevation(hour)
        sun_az = _compute_sun_azimuth(hour)
        color_k = _compute_color_temp(hour)
        slots.append(TimeSlot(
            hour=hour,
            label=_format_hour_label(hour),
            sun_elevation=sun_el,
            sun_azimuth=sun_az,
            color_temp_k=color_k,
            shadow_description=_shadow_description(hour, sun_el, sun_az),
            sky_description=_sky_description(hour),
            lighting_description=get_lighting_description(hour),
        ))
    return slots


def build_prompt(slot: TimeSlot) -> str:
    """Build a complete API prompt for a given time slot."""
    return PROMPT_TEMPLATE.format(
        time_label=slot.label,
        lighting_description=slot.lighting_description,
        sun_elevation=slot.sun_elevation,
        sun_azimuth=int(slot.sun_azimuth),
        color_temp_description=_color_temp_description(slot.color_temp_k),
        color_temp_k=slot.color_temp_k,
        shadow_description=slot.shadow_description,
        sky_description=slot.sky_description,
    )
