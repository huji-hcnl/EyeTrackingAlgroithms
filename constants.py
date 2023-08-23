"""
List of all the constants used as field names, column names, etc.
"""

MILLISECONDS_PER_SECOND = 1000
MICROSECONDS_PER_MILLISECOND = 1000
MICROSECONDS_PER_SECOND = MICROSECONDS_PER_MILLISECOND * MILLISECONDS_PER_SECOND  # 1,000,000

TRIAL = "trial"
TIME = "time"
TRIGGER = "trigger"
TARGET = "target"
DURATION = "duration"
DISTANCE = "distance"
ANGLE = "angle"
STIMULUS = "stimulus"
MILLISECONDS = "milliseconds"
MICROSECONDS = "microseconds"
EVENT_TYPE = "event_type"
LEFT = "left"
RIGHT = "right"
PUPIL = "pupil"

LEFT_X, RIGHT_X = f"{LEFT}_x", f"{RIGHT}_x"
LEFT_Y, RIGHT_Y = f"{LEFT}_y", f"{RIGHT}_y"
LEFT_PUPIL, RIGHT_PUPIL = f"{LEFT}_{PUPIL}", f"{RIGHT}_{PUPIL}"
