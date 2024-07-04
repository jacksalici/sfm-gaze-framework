import numpy as np
from pathlib import Path
import configparser, os
config  = configparser.ConfigParser()
config.read('config.ini')

import read_write_model 
model_path = Path(config["StructureFromMotion"]["model_path"]) / "sfm"
cameras, images, points3D = read_write_model.read_model(path=model_path, ext='.bin')

    

def cpf_to_rgb_transform(pitch_cpf, yaw_cpf, extrinsic_matrix):
    
    cpf_vector = pitch_yaw_to_vector(pitch_cpf, yaw_cpf)
    
    rgb_vector = np.linalg.inv(extrinsic_matrix[:3, :3]) @ cpf_vector
        
    return rgb_vector


def pitch_yaw_to_vector(pitch_rad, yaw_rad):
    # inspired by https://github.com/facebookresearch/projectaria_tools/blob/3f6079ffcd21b8975fed2ce2bef211473bc498ad/core/mps/EyeGazeReader.h#L40

    x = np.tan(yaw_rad)
    y = np.tan(pitch_rad)
    z = 1

    #return np.array([1, 0, 0])
    direction = np.array([x, y, z])
    return direction/np.linalg.norm(direction)







file = Path(config['StructureFromMotion']['gaze_output_path']) / 'img1.npz'
        
  
if not os.path.isfile(file):
    raise("Not a file.")
            
npz_file = np.load(file)
    # use exact intrinsic
params = npz_file['rbg_camera_intrinsic']
extrinsic_matrix = npz_file['rbg_camera_extrinsic']  

print(np.degrees(extrinsic_matrix[:3, :3]))
    
    
gaze_yaw_pitch = npz_file['gaze_yaw_pitch']
yaw_cpf, pitch_cpf = gaze_yaw_pitch[0], gaze_yaw_pitch[1]

# Transform from CPF to RGB
vector_rgb = cpf_to_rgb_transform(pitch_cpf, yaw_cpf, extrinsic_matrix)

# Convert to vector in RGB frame
vector_cpf = pitch_yaw_to_vector(pitch_cpf, yaw_cpf)

print(f"Vector in RGB frame: {vector_rgb}")
print(f"Vector in CPF frame: {vector_cpf}")
print(f"Difference degrees: {np.degrees(vector_cpf-vector_rgb)}")



