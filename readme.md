# SFM-based gaze estimation dataset

## Pipeline 🚥

1. `recordManager.py` to download all the files and save them.
2. `timestampDetector.py` to detect QR code starts and ends as well as synced video pairs and save them in a `csv` file.
3. `frameExtractor.py` to save frames from video using the just found timestamps.  
4. `reconstruction.py` to made the SFM computation.  
5. `rerunLogger.py` for the ReRun visualization.  
