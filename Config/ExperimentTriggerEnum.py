from enum import IntEnum


class ExperimentTriggerEnum(IntEnum):
    """ This class enumerates the triggers used in the experiment. """

    NULL = 0                # null trigger, transmitted on parallel-port after any other trigger

    START_RECORDING = 254   # eye tracker recording started
    END_RECORDING = 255     # eye tracker recording stopped
