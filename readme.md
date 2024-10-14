# SFM-based gaze estimation framework

⚠️ More detail will be added soon.

## Pipeline 🚥

1. `recordManager.py` to download all the files and save them.
2. `timestampDetector.py` to detect QR code starts and ends as well as synced video pairs and save them in a `csv` file.
3. `frameExtractor.py` to save frames and the gaze 2d info from video using the just found timestamps.  
4. `reconstruction.py` to made the SFM computation.  
    - _[optional]_ `localize.py`, localize a image in the SFM model.
5. `gazeTo3dpoints.py` to convert the gaze direction to the 3d gaze position.
6. `rerunLogger.py` for the ReRun visualization.  
