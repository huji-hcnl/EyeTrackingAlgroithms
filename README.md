# EyeTrackingAlgorithms
Implementation of various eye-tracking event identifiers

Data:
1. "Andersson Data" that is also used in the article:
    Andersson, R. et. al (2017): "One algorithm to rule them all? An evaluation and
    discussion of ten eye movement event-detection algorithms".
    url to the zip file: "http://www.kasprowski.pl/datasets/events.zip"

    assumptions: 
   - the frequency rate is 500Hz so there is a 2ms gap between every two samples.
   - labels are marked as follows: 1- fixation, 2- saccade, 3- PSOs, 4- smooth pursuit, 5- blinks, 6- undefined
   - the data wad labeled by 2 raters: MN, RA
   - there were 3 types of stimuli: static images, video clips and moving dots on the screen


LoadAnderssonData.py:
    - Loads the data into a dataframe, either from the data url or from a pickle file
    - The dataframe includes the following information:
        number of dataset, stimuli, rater's name, time stamp (difference of 2ms, starting from 0), 
        x coordinate of the right eye, y coordinate, labels (1-6)
    - If the timestamp is not given- completes it


Detectors- implemented algorithms to detect types of events in eye movements:
1. IVT- IVTDetector.py:
    * Input: velocity threshold to distinguish between fixation and saccades. Default = 0.5
    * Output: np ndarray of labels (1 or 2 only)
    - calculates the velocity between to samples (= location of the eye, xy coordinates)
    - thresholding the velocity and determining labels
2. IDT- IDTDetector.py:
    * Input: dispersion threshold to distinguish between fixation and saccades. Default = 3.5 px
            window dimension to calculate dispersion by the min and max of the samples in the window.
    * Our dispersion calculation (according to the article: "Review and Evaluation of Eye Movement Event
Detection Algorithms" by Birtukan Birawo and Pawel Kasprowski): D = [max(x) - min(x)] + [max(y) - min(y)]
    * Output: np ndarray of labels (1 or 2 only)
    
    
