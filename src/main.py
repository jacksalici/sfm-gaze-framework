from s3_frameExtractor import extractor
from s4_reconstruction import reconstruct
from s5_gazeTo3dPoints import gaze2points

import tomllib, os
from pathlib import Path

def pipeline(session_id, steps = "123"):
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

    frames_path_root = config["aria_recordings"]["frames_path_root"]
    gaze_path_root = config["aria_recordings"]["gaze_output"]
    model_path_root = config["aria_recordings"]["model_path"]
    eye_tracking_device_id = config["gaze_estimation"]["eye_tracking_device_id"]
    
    if "1" in steps:
        print("\n######################\nSTEP 1: Extract Frames\n\n")
        extractor(csv_file_frames, frames_path_root, gaze_path_root, session_id)
        
    try:
        import ast
        import pandas as pd
        df = pd.read_csv(csv_file_frames)
        
        for _, row in df.iterrows():            
            if row['session_id'] != session_id:
                continue
            
            # Parse the string representation of the list into an actual list
            file_paths = ast.literal_eval(row['file_paths'])
            start_timestamp = row['start_timestamp']
            end_timestamp = row['end_timestamp']
            scene = row['scene']
            participant = row['participant']
            take = row['take']
            et_device_id = row['et_device_id']       
            
            folder =  "{}_{}_{}".format(scene, participant, take)
        
            if "2" in steps:
                print("\n######################\nSTEP 2: Reconstruct SFM\n\n")
                reconstruct(Path(frames_path_root) / folder, Path(model_path_root)  / folder)

            if "3" in steps:
                print("\n######################\nSTEP 3: Project gaze\n\n")
                csv_file_gaze = os.path.join(config["gaze_estimation"]["gaze_3d_output_folder"], folder + '.csv')

                gaze2points(csv_file_gaze, os.path.join(model_path_root, folder), os.path.join(gaze_path_root, folder), eye_tracking_device_id)

    except Exception as e:
        print("Error.", e)
        return
    
    
    
    
    
def main():
    import argparse, glob, toml
    config = toml.load("config.toml")

    parser = argparse.ArgumentParser(description="Process session argument.")
    parser.add_argument('--session', '-s', required=False, help="Specify the session ID")
    parser.add_argument('--steps', '-p', type=str, required=False, default="123", help="Specify the steps to do")
    parser.add_argument('--glob_sessions_folder', '-g', action="store_true", default=False, required=False)
    args = parser.parse_args()
    if args.session:
        pipeline(args.session, args.steps)
    elif args.glob_sessions_folder:
        for folder in glob.glob("*", root_dir=config["aria_recordings"]["vrs_glob"]):
            pipeline(folder, args.steps)
    

if __name__ == "__main__":
    main()