"""
This file contains the configuration for each specific experiment.
"""

import os

from Config.ScreenMonitor import ScreenMonitor

# DIRECTORIES
BASE_DIR = ""
STIMULI_DIR = os.path.join(BASE_DIR, "Stimuli", "generated_stim1")
RAW_DATA_DIR = os.path.join(BASE_DIR, "RawData")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

# GLOBAL VARIABLES
ADDITIONAL_COLUMNS = []  # additional columns to be added to the gaze data
SCREEN_MONITOR: ScreenMonitor = ScreenMonitor.from_default()  # global variable: screen monitor object
