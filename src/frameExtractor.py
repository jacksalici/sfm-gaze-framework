import pandas as pd
import ast
from aria_glasses_utils.BetterFrameExtractor import exportFrames
import os, glob, csv
import tomllib
from pathlib import Path

config = tomllib.load(open("config.toml", "rb"))
csv_file = config["aria_recordings"]["recordings_sheet"]
df = pd.read_csv(csv_file)
base_path_root = config["aria_recordings"]["frames_path_root"]


for _, row in df.iterrows():
    session_id = row['session_id']
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
        
        folder_name = f"{scene}_{participant}_{take}_{device_id}"
            
        exportFrames(
            input_vrs_path= str(file_path),
            imgs_output_dir = os.path.join(base_path_root, folder_name),
            gaze_output_folder = os.path.join(base_path_root, "gaze_info",  folder_name),
            export_gaze_info = device_id == et_device_id,
            export_time_step = 1000000000,
            export_slam_camera_frames = True,
            min_confidence = 0.7,
            show_preview = False,
            range_limits_ns = (start_timestamp, end_timestamp)
        )
