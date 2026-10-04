"""macOS idle-time detection."""

import Quartz


def idle_seconds() -> float:
    """Seconds since the last keyboard/mouse/trackpad event, system-wide."""
    return Quartz.CGEventSourceSecondsSinceLastEventType(
        Quartz.kCGEventSourceStateHIDSystemState,
        Quartz.kCGAnyInputEventType,
    )
