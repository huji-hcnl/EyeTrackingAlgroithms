"""
List of all the constants used as field names, column names, etc.
"""
EPSILON = 1e-8

MILLISECONDS_PER_SECOND = 1000
MICROSECONDS_PER_MILLISECOND = 1000
MICROSECONDS_PER_SECOND = MICROSECONDS_PER_MILLISECOND * MILLISECONDS_PER_SECOND  # 1,000,000

SUBJECT = "subject"
SUBJECT_ID = f"{SUBJECT}_id"
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
SAMPLING_RATE = "sampling_rate"

X, Y = 'x', 'y'
LEFT = "left"
RIGHT = "right"
PUPIL = "pupil"
LEFT_X, RIGHT_X = f"{LEFT}_{X}", f"{RIGHT}_{X}"
LEFT_Y, RIGHT_Y = f"{LEFT}_{Y}", f"{RIGHT}_{Y}"
LEFT_PUPIL, RIGHT_PUPIL = f"{LEFT}_{PUPIL}", f"{RIGHT}_{PUPIL}"
