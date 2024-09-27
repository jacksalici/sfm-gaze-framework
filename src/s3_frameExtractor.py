import pandas as pd
import ast
from aria_glasses_utils.BetterFrameExtractor import exportFrames
import os, glob, csv
from pathlib import Path


def extractor(csv_file, frames_path_root, gaze_path_root, chosen_session_id = ""):
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        session_id = row['session_id']
        
        if session_id != chosen_session_id:
            continue
        
        # Parse the string representation of the list into an actual list
        file_paths = ast.literal_eval(row['file_paths'])
        start_timestamp = row['start_timestamp']
        end_timestamp = row['end_timestamp']
        scene = row['scene']
        participant = row['participant']
        take = row['take']
        et_device_id = row['et_device_id']
        
        for file_path in file_paths: 
            device_id = Path(file_path).parts[-2]
            print(scene, participant, take)
            parent_folder_name = "{}_{}_{}".format(scene, participant, take)

                
            exportFrames(
                input_vrs_path= str(file_path),
                imgs_output_dir = os.path.join(frames_path_root, parent_folder_name, device_id),
                gaze_output_folder = os.path.join(gaze_path_root,  parent_folder_name, device_id),
                export_gaze_info = device_id == et_device_id,
                export_time_step = 1000000000,
                export_slam_camera_frames = False,
                min_confidence = 0.7,
                show_preview = False,
                range_limits_ns = (start_timestamp, end_timestamp),
                filename_w_timestamp = True
            )


if __name__=="__main__":
    import tomllib
    
    config = tomllib.load(open("config.toml", "rb"))
    csv_file = config["aria_recordings"]["recordings_sheet"]
    frames_path_root = config["aria_recordings"]["frames_path_root"]
    gaze_path_root = config["aria_recordings"]["gaze_output"]
    
    extractor(csv_file, frames_path_root, gaze_path_root)