"""
Prompt template system for Living Image crossfade test harness.

Generates 12 time-specific prompts (7am-6pm) that instruct image generation
APIs to relight an architectural image while preserving structural fidelity.
"""

from dataclasses import dataclass
import math


PROMPT_TEMPLATE = """Relight this architectural image to accurately depict how it would look at {time_label} on a clear day.

LIGHTING CONDITIONS at {time_label}:
- {lighting_description}
- Sun elevation: approximately {sun_elevation} degrees above the horizon
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
    """Represents one hour in the 7am-6pm range with lighting metadata."""
    hour: int               # 24h format: 7-18
    label: str              # "7:00 AM", "12:00 PM", etc.
    sun_elevation: float    # degrees above horizon
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
    Approximate sun elevation for a mid-latitude location (~40N)
    on an equinox-like day. Peaks at solar noon (~12:30).

    Uses a sinusoidal model: elevation = max_el * sin(pi * (h - sunrise) / day_length)
    """
    sunrise = 6.5   # approximate sunrise
    sunset = 18.5   # approximate sunset
    day_length = sunset - sunrise
    max_elevation = 55.0  # degrees at solar noon for ~40N equinox

    if hour <= sunrise or hour >= sunset:
        return 0.0

    fraction = (hour - sunrise) / day_length
    elevation = max_elevation * math.sin(math.pi * fraction)
    return round(elevation, 1)


def _compute_color_temp(hour: int) -> int:
    """
    Color temperature in Kelvin across the day.
    Dawn/dusk: ~2500K (warm amber)
    Golden hour: ~3500K (warm gold)
    Midday: ~5500-6500K (neutral daylight)
    """
    # Map hour to color temp using a curve that's warm at extremes, cool at noon
    temps = {
        7:  2800,
        8:  3200,
        9:  4000,
        10: 4800,
        11: 5500,
        12: 6000,
        13: 6200,
        14: 5800,
        15: 5200,
        16: 4500,
        17: 3500,
        18: 2700,
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


def _shadow_description(hour: int, sun_elevation: float) -> str:
    """Describe shadow characteristics based on time and sun position."""
    if hour <= 8:
        return "Very long shadows stretching to the west. Soft, diffuse shadow edges from low-angle light."
    elif hour <= 10:
        return "Long shadows angled to the west-northwest. Moderately soft shadow edges."
    elif hour <= 11:
        return "Medium-length shadows angled slightly west. Increasingly defined shadow edges."
    elif hour <= 13:
        return "Short shadows nearly directly below objects. Sharp, well-defined shadow edges from overhead sun."
    elif hour <= 14:
        return "Medium-length shadows angled slightly east. Well-defined shadow edges."
    elif hour <= 16:
        return "Long shadows stretching to the east-northeast. Moderately soft shadow edges."
    else:
        return "Very long shadows stretching to the east. Soft, warm-tinted shadow edges from low-angle light."


def _sky_description(hour: int) -> str:
    """Describe sky appearance at given hour."""
    descs = {
        7:  "Soft gradient from warm peach/coral at the horizon to pale blue above. Low haze near the horizon.",
        8:  "Brightening sky with warm yellow tones near the horizon transitioning to clear blue above.",
        9:  "Clear blue sky with slight warmth near the horizon. Bright and luminous.",
        10: "Deep clear blue sky. Uniform brightness with minimal horizon haze.",
        11: "Bright, saturated blue sky. Even illumination across the dome.",
        12: "Intense blue overhead, slightly lighter near the horizon. Maximum brightness.",
        13: "Bright blue sky, very slightly shifting from peak intensity.",
        14: "Clear blue sky, beginning almost imperceptibly to warm.",
        15: "Blue sky with subtle warm shift. Light becoming slightly more directional.",
        16: "Sky transitioning to warmer blue. Noticeable golden quality to the light.",
        17: "Golden hour sky. Warm amber-gold tones spreading from the western horizon. Rich saturated colors.",
        18: "Sunset sky with deep orange, coral, and magenta at the horizon fading to deep blue above.",
    }
    return descs.get(hour, "Clear blue sky.")


def get_lighting_description(hour: int) -> str:
    """Get a natural-language description of lighting conditions at this hour."""
    descs = {
        7:  "Early morning light. Low sun casting long warm shadows. Soft, diffuse golden dawn light with gentle contrast.",
        8:  "Morning light gaining strength. Sun climbing in the east. Warm directional light with moderate shadow length.",
        9:  "Mid-morning daylight. Sun well above the horizon. Balanced warm-neutral light with clear shadows.",
        10: "Late morning light. Strong directional sun from the east-southeast. Neutral bright daylight.",
        11: "Approaching midday. Nearly overhead sun. Bright, even illumination with short shadows.",
        12: "Noon / midday light. Sun at its highest point. Maximum brightness with minimal shadows directly below objects.",
        13: "Early afternoon. Sun beginning its westward descent. Bright, slightly warm directional light.",
        14: "Mid-afternoon light. Sun in the west-southwest. Clear directional light with lengthening shadows.",
        15: "Afternoon light. Sun lowering in the west. Increasingly warm and directional light.",
        16: "Late afternoon. Sun getting lower. Noticeably warm golden light with long shadows stretching eastward.",
        17: "Golden hour / late afternoon. Low sun casting rich golden-amber light. Long dramatic shadows. Warm color on all surfaces.",
        18: "Sunset / dusk. Very low sun near the horizon. Deep warm orange-amber light. Extremely long shadows. Dramatic contrast.",
    }
    return descs.get(hour, "Standard daylight conditions.")


def generate_time_slots() -> list[TimeSlot]:
    """Generate 12 TimeSlots from 7am to 6pm."""
    slots = []
    for hour in range(7, 19):
        sun_el = _compute_sun_elevation(hour)
        color_k = _compute_color_temp(hour)
        slots.append(TimeSlot(
            hour=hour,
            label=_format_hour_label(hour),
            sun_elevation=sun_el,
            color_temp_k=color_k,
            shadow_description=_shadow_description(hour, sun_el),
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
        color_temp_description=_color_temp_description(slot.color_temp_k),
        color_temp_k=slot.color_temp_k,
        shadow_description=slot.shadow_description,
        sky_description=slot.sky_description,
    )
