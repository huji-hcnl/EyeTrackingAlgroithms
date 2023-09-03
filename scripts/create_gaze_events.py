import traceback
import warnings as w

import numpy as np
import pandas as pd
from typing import List, Union, Optional

import constants as cnst
import Utils.array_utils as arr_utils
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from GazeEvents.BlinkEvent import BlinkEvent
from GazeEvents.SaccadeEvent import SaccadeEvent
from GazeEvents.FixationEvent import FixationEvent


def create_gaze_events(data: pd.DataFrame,
                       candidates_column: str, viewer_distance: float, eye: str
                       ) -> List[Union[BlinkEvent, SaccadeEvent, FixationEvent]]:
    """
    Creates a list of gaze events from the given gaze data, based on the given candidates.
    This function assumes that the gaze data has the predefined column names from the Config.experiment_config module.
    """
    eye = eye.lower()
    if eye not in ["left", "right"]:
        raise ValueError(f"Invalid eye: {eye}")
    if candidates_column not in data.columns:
        raise ValueError(f"No column named {candidates_column} in the given `data`")
    candidates = data[candidates_column].values

    events = []
    chunk_idxs = arr_utils.get_chunk_indices(candidates)
    for chunk_idxs in chunk_idxs:
        event_type = GazeEventTypeEnum(candidates[chunk_idxs[0]])
        try:
            chunk_data = data.iloc[chunk_idxs]
            event = _create_event(event_type=event_type, event_data=chunk_data,
                                  viewer_distance=viewer_distance, eye=eye)
            events.append(event)
        except Exception as e:
            trace = traceback.format_exc()
            w.warn(f"{type(e)} when attempting to create a {event_type.name} event: {trace}\n")

    # filter None events and sort by start-time
    events = [event for event in events if event is not None]
    events.sort(key=lambda ev: ev.start_time)
    return events


def _create_event(event_type: GazeEventTypeEnum, event_data: pd.DataFrame,
                  viewer_distance: float, eye: str) -> Union[
        None, BlinkEvent, SaccadeEvent, FixationEvent]:
    """
    Creates a single gaze event from the given `event_data`, assuming it has the predefined column names from the
    Config.experiment_config module. This is True for all datasets that were parsed using a DataParser class.
    """
    if event_type == GazeEventTypeEnum.UNDEFINED:
        return None
    should_warn = True
    t_data = arr_utils.extract_column_safe(event_data, cnst.MILLISECONDS, warn=should_warn)
    if event_type == GazeEventTypeEnum.BLINK:
        return BlinkEvent(timestamps=t_data)

    x_data = arr_utils.extract_column_safe(event_data, cnst.LEFT_X if eye == "left" else cnst.RIGHT_X, warn=should_warn)
    y_data = arr_utils.extract_column_safe(event_data, cnst.LEFT_Y if eye == "left" else cnst.RIGHT_Y, warn=should_warn)
    if event_type == GazeEventTypeEnum.SACCADE:
        return SaccadeEvent(timestamps=t_data, x=x_data, y=y_data, viewer_distance=viewer_distance)
    pupil_data = arr_utils.extract_column_safe(event_data, cnst.LEFT_PUPIL if eye == "left" else cnst.RIGHT_PUPIL,
                                               warn=should_warn)
    if event_type == GazeEventTypeEnum.FIXATION:
        return FixationEvent(timestamps=t_data, x=x_data, y=y_data, pupil=pupil_data,
                             viewer_distance=viewer_distance)
    raise ValueError(f"Unknown event type: {event_type}")
