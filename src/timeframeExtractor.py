from aria_glasses_utils.common import *
from aria_glasses_utils.BetterEyeGaze import BetterEyeGaze
from aria_glasses_utils.BetterAriaProvider import *

import cv2
import numpy as np

import os
from pathlib import Path

SLAM = False

import tomllib
config = tomllib.load(open("config.toml", "rb"))

import os, glob

def getSyncedVRSFiles(vrs_path):
    sessionFolder = Path(vrs_path).parents[1]
    return glob.glob(str(sessionFolder / '**/*.vrs'), recursive=True)

def confidence(img1, img2):

    def process(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(img_gray, (43, 43), 21)

    res = cv2.matchTemplate(process(img1), process(img2), cv2.TM_CCOEFF_NORMED)
    return res.max()

def blurryness(img):
    return -cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

import csv   
def save_info(file_path, start_timestamp, end_timestamp,scene,participant,take):
    csv_file = config["aria_recordings"]["recordings_sheet"]
    csv_file_exists = True
    
    if not os.path.exists(csv_file):
        csv_file_exists = False
        

    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        
        if not csv_file_exists:
            #save header once
            fields=['session_id', 'file_paths', 'start_timestamp','end_timestamp','scene','participant','take']
            writer.writerow(fields)
        
        fields=[Path(file_path).parts[-3], getSyncedVRSFiles(file_path), start_timestamp,end_timestamp,scene,participant,take]
        writer.writerow(fields)



from pyzbar.pyzbar import decode
from PIL import Image

def extract_codes(number):
    assert type(number) == int, "Number must be an integer."
    assert number < 1_000_000_000, "Number must be less than 1,000,000,000."
    
    scene = number // 1_000_000
    participant = (number % 1_000_000) // 1000
    take = number % 1000
    
    return scene, participant, take

END_CODE = 0

def decode_qrcode(img):
    result = decode(img)
    if (len(result)>0):
        num = int(result[0].data)
        if num == END_CODE:
            return "end", 0,0,0
        else:
            scene, participant, take = extract_codes(num)
            return "start", scene, participant, take
    
    return None,None,None,None
        
            

def main():
 
    

    folder_path = config["aria_recordings"]["vrs_glob"]
    
    
    for filename in glob.iglob(os.path.join(folder_path,'**/*.vrs'), recursive=True):
        file_path = os.path.join(folder_path, filename)
        print(f"Elaborating file {file_path}")
    
    
        provider = BetterAriaProvider(vrs=file_path)
        eye_gaze = BetterEyeGaze(*provider.get_calibration())

        if False:
            output_folder = config["aria_recordings"]["output"]
            gaze_output_folder = config["aria_recordings"]["gaze_output"]
            from pathlib import Path
            Path( output_folder).mkdir( parents=True, exist_ok=True )
            Path( gaze_output_folder).mkdir( parents=True, exist_ok=True )
        start_timestamp, end_timestamp, scene, participant, take = 0, 0, 0, 0, 0
        
        started = False
        
        imgs = []
        imgs_et = []
        for time in provider.get_time_range(time_step=1_000_000_000):
            print(f"INFO: Checking frame at time {time}")
            frame = {}
            
            frame['rgb'], _ = provider.get_frame(Streams.RGB, time_ns=time)
            
            
            
            
            img_et, _ = provider.get_frame(Streams.ET, time, False, False)
            
            if SLAM:
                frame['slam_l'], _ = provider.get_frame(Streams.SLAM_L, time)
                frame['slam_r'], _ = provider.get_frame(Streams.SLAM_R, time)
            
            result, res_scene, res_scene, res_take = decode_qrcode(frame['rgb'])
            
            if result == "start" and not started:
                started = True
                start_timestamp = time
                scene, participant, take = res_scene, res_scene, res_take
                print("INFO: ### DETECTED START QR-CODE")
                
                
            if result == "end" and start_timestamp != 0:
                end_timestamp = time
                print("INFO: ### DETECTED END QR-CODE", file_path, start_timestamp, end_timestamp, scene, participant, take)
                save_info(file_path, start_timestamp, end_timestamp, scene, participant, take)
                started = False
                start_timestamp = 0
            
            """  
            if (len(imgs) > 0 and confidence(frame["rgb"], imgs[-1]["rgb"]) < 0.7) or len(imgs) == 0 or ALL_IMAGES:
                imgs.append(frame)
                imgs_et.append(img_et)
                
                print(f"INFO: Frame added to the list.")
            else:
                if blurryness(frame["rgb"]) < blurryness(imgs[-1]["rgb"]):
                    imgs[-1] = frame
                    print(
                        f"INFO: Frame substituted to the last in the list for better sharpness."
                    )"""

        # cv2.circle(img, eye_gaze.rotate_pixel_cw90(gaze_center_in_pixels) , 5, (255, 0, 0), 2)
        # sleep(0.3)
    """
    import torch
    
    rbg2cpf_camera_extrinsic = provider.calibration_device.get_transform_cpf_sensor(
        Streams.RGB.label()
    ).to_matrix()

    for index, frame in enumerate(imgs):
        
        for name,img in frame.items():
            cv2.imwrite(os.path.join(output_folder, f"img{index}{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        yaw, pitch = eye_gaze.predict(torch.tensor(imgs_et[index], device="cpu"))
        gaze_center_in_cpf, gaze_center_in_pixels = eye_gaze.get_gaze_center_raw(
            yaw, pitch
        ) 

        np.savez(
                os.path.join(gaze_output_folder, f"img{index}.npz"),
                gaze_yaw_pitch=np.array([yaw, pitch]),
                gaze_center_in_cpf=gaze_center_in_cpf,
                gaze_center_in_rgb_pixels=gaze_center_in_pixels,
                gaze_center_in_rgb_frame=(
                    np.linalg.inv(rbg2cpf_camera_extrinsic)
                    @ np.append(gaze_center_in_cpf, [1])
                )[:3],
                rbg2cpf_camera_extrinsic=rbg2cpf_camera_extrinsic,
                rbg_camera_intrinsic=eye_gaze.getCameraCalib().projection_params(),
            )
        print(f"INFO: File {index} saved.")

    """

if __name__ == "__main__":
    main()
