import warnings as w
import pandas as pd
from typing import List, Union

import constants as cnst
import Utils.array_utils as arr_utils
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from GazeEvents.BlinkEvent import BlinkEvent
from GazeEvents.SaccadeEvent import SaccadeEvent
from GazeEvents.FixationEvent import FixationEvent


def create_gaze_events(gaze_data: pd.DataFrame, candidates: List[GazeEventTypeEnum],
                       viewer_distance: float, eye: str
                       ) -> List[Union[BlinkEvent, SaccadeEvent, FixationEvent]]:
    """ Creates a list of gaze events from the given gaze data, based on the given candidates """
    eye = eye.lower()
    if eye not in ["left", "right"]:
        raise ValueError(f"Invalid eye: {eye}")

    events = []
    chunk_idxs = arr_utils.get_chunk_indices(candidates)
    for chunk_idxs in chunk_idxs:
        event_type: GazeEventTypeEnum = candidates[chunk_idxs[0]]
        try:
            chunk_data = gaze_data.iloc[chunk_idxs]
            event = _create_event(event_data=chunk_data, event_type=event_type,
                                  viewer_distance=viewer_distance, eye=eye)
            events.append(event)
        except Exception as e:
            w.warn(f"Error when attempting to create a {event_type.name} event: {e}")

    # filter None events and sort by start-time
    events = [event for event in events if event is not None]
    events.sort(key=lambda ev: ev.start_time)
    return events


def _create_event(event_data: pd.DataFrame, event_type: GazeEventTypeEnum,
                  viewer_distance: float, eye: str) -> Union[
        None, BlinkEvent, SaccadeEvent, FixationEvent]:
    """ Creates a single gaze event from the given event data. """
    if event_type == GazeEventTypeEnum.UNDEFINED:
        return None
    if event_type == GazeEventTypeEnum.FIXATION:
        return FixationEvent(timestamps=event_data[cnst.MILLISECONDS].values,
                             x=event_data[f"{eye}_x"].values,
                             y=event_data[f"{eye}_y"].values,
                             pupil=event_data[f"{eye}_{cnst.PUPIL}"].values,
                             viewer_distance=viewer_distance)
    if event_type == GazeEventTypeEnum.SACCADE:
        return SaccadeEvent(timestamps=event_data[cnst.MILLISECONDS].values,
                            x=event_data[f"{eye}_x"].values,
                            y=event_data[f"{eye}_y"].values,
                            viewer_distance=viewer_distance)
    if event_type == GazeEventTypeEnum.BLINK:
        return BlinkEvent(timestamps=event_data[cnst.MILLISECONDS].values)
    raise ValueError(f"Unknown event type: {event_type}")
