from s3_frameExtractor import extractor
from s4_reconstruction import reconstruct
from s5_gazeTo3dPoints import gaze2points

import tomllib, os
from pathlib import Path

def pipeline(session_id, starting_step = 0):
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
    csv_file_frames = config["aria_recordings"]["recordings_sheet"]
    os.makedirs(config["gaze_estimation"]["gaze_3d_output_folder"], exist_ok=True)
    csv_file_gaze = os.path.join(config["gaze_estimation"]["gaze_3d_output_folder"], session_id + '.csv')

    frames_path_root = os.path.join(config["aria_recordings"]["frames_path_root"], session_id)
    gaze_path_root = os.path.join(config["aria_recordings"]["gaze_output"], session_id)
    model_path_root = os.path.join(config["aria_recordings"]["model_path"], session_id)
    eye_tracking_device_id = config["gaze_estimation"]["eye_tracking_device_id"]
    
    if starting_step<2:
        print("\n######################\nSTEP 1: Extract Frames\n\n")
        extractor(csv_file_frames, frames_path_root, gaze_path_root)
    
    if starting_step<3:
        print("\n######################\nSTEP 2: Reconstruct SFM\n\n")
        reconstruct(Path(frames_path_root), Path(model_path_root))
    
    print("\n######################\nSTEP 3: Project gaze\n\n")
    gaze2points(csv_file_gaze, model_path_root, gaze_path_root, eye_tracking_device_id)

    
    
    
    
    
    
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process session argument.")
    parser.add_argument('--session', '-s', required=True, help="Specify the session ID")
    parser.add_argument('--from_step', '-f', type=int, required=False, default=0, help="Specify the starting point (default is 0)")

    args = parser.parse_args()
    pipeline(args.session, args.from_step)

if __name__ == "__main__":
    main()