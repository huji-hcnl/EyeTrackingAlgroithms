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


def create_gaze_events(gaze_data: pd.DataFrame, candidates: List[GazeEventTypeEnum],
                       viewer_distance: float, eye: str
                       ) -> List[Union[BlinkEvent, SaccadeEvent, FixationEvent]]:
    """
    Creates a list of gaze events from the given gaze data, based on the given candidates.
    This function assumes that the gaze data has the predefined column names from the Config.experiment_config module.
    """
    eye = eye.lower()
    if eye not in ["left", "right"]:
        raise ValueError(f"Invalid eye: {eye}")

    events = []
    chunk_idxs = arr_utils.get_chunk_indices(candidates)
    for chunk_idxs in chunk_idxs:
        event_type: GazeEventTypeEnum = candidates[chunk_idxs[0]]
        try:
            chunk_data = gaze_data.iloc[chunk_idxs]
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
    x_name = cnst.LEFT_X if eye == "left" else cnst.RIGHT_X
    y_name = cnst.LEFT_Y if eye == "left" else cnst.RIGHT_Y
    pupil_name = cnst.LEFT_PUPIL if eye == "left" else cnst.RIGHT_PUPIL
    return _create_event_raw(event_type=event_type,
                             t=event_data[cnst.MILLISECONDS].values,
                             x=event_data[x_name].values,
                             y=event_data[y_name].values,
                             pupil=event_data[pupil_name].values,
                             viewer_distance=viewer_distance)


def _create_event_raw(event_type: GazeEventTypeEnum, t: np.ndarray,
                      x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                      pupil: Optional[np.ndarray] = None, viewer_distance: Optional[float] = None
                      ) -> Union[None, BlinkEvent, SaccadeEvent, FixationEvent]:
    """ Creates a single gaze event from the given event data. No assumptions are made about the data. """
    if event_type == GazeEventTypeEnum.UNDEFINED:
        return None
    if event_type == GazeEventTypeEnum.FIXATION:
        return FixationEvent(timestamps=t, x=x, y=y, pupil=pupil, viewer_distance=viewer_distance)
    if event_type == GazeEventTypeEnum.SACCADE:
        return SaccadeEvent(timestamps=t, x=x, y=y, viewer_distance=viewer_distance)
    if event_type == GazeEventTypeEnum.BLINK:
        return BlinkEvent(timestamps=t)
    raise ValueError(f"Unknown event type: {event_type}")
