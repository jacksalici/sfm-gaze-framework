from s3_frameExtractor import extractor
from s4_reconstruction import reconstruct
from s5_gazeTo3dPoints import g

import tomllib, os

def pipeline(session_id):
    """
1. ONCE - `recordManager.py` to download all the files and save them.
2. ONCE - `timestampDetector.py` to detect QR code starts and ends as well as synced video pairs and save them in a `csv` file.
3. PER SESSION ID - `frameExtractor.py` to save frames and the gaze 2d info from video using the just found timestamps.  
4. PER SESSION ID - `reconstruction.py` to made the SFM computation.  
5. _[optional]_ `localize.py`, localize a image in the SFM model.
6. `gazeTo3dpoints.py` to convert the gaze direction to the 3d gaze position.
7. OPTIONAL - `rerunLogger.py` for the ReRun visualization.  
    """
    
    
    config = tomllib.load(open("config.toml", "rb"))
    csv_file = config["aria_recordings"]["recordings_sheet"]
    
    frames_path_root = os.path.join(config["aria_recordings"]["frames_path_root"], session_id)
    gaze_path_root = os.path.join(config["aria_recordings"]["gaze_output"], session_id)
    model_path_root = os.path.join(config["aria_recordings"]["model_path"], session_id)
    
    extractor(csv_file, frames_path_root, gaze_path_root)
    reconstruct(gaze_path_root, model_path_root)
    
    
    
    
    
    
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process session argument.")
    parser.add_argument('--session', '-s', required=True, help="Specify the session ID")

    args = parser.parse_args()
    pipeline(args.session)

if __name__ == "__main__":
    main()