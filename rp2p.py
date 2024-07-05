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



img_name = "img4rgb.jpg"

npz_file = Path(config['StructureFromMotion']['gaze_output_path']) / f'{img_name[0:-7]}.npz'
image_rgb = Path(config['StructureFromMotion']['dataset_path']) / img_name
        
  
if not os.path.isfile(npz_file) or not os.path.isfile(image_rgb):
    raise("Not a file.")
            
npz_file = np.load(npz_file)
    # use exact intrinsic
params = npz_file['rbg_camera_intrinsic']
extrinsic_matrix_cpf2rgb = npz_file['rbg_camera_extrinsic']  

print(np.degrees(extrinsic_matrix_cpf2rgb[:3, :3]))
    
    
gaze_yaw_pitch = npz_file['gaze_yaw_pitch']
yaw_cpf, pitch_cpf = gaze_yaw_pitch[0], gaze_yaw_pitch[1]

# Transform from CPF to RGB
vector_rgb = cpf_to_rgb_transform(pitch_cpf, yaw_cpf, extrinsic_matrix_cpf2rgb)

# Convert to vector in RGB frame
vector_cpf = pitch_yaw_to_vector(pitch_cpf, yaw_cpf)

print(f"Vector in RGB frame: {vector_rgb}")
print(f"Vector in CPF frame: {vector_cpf}")
print(f"Difference degrees: {np.degrees(vector_cpf-vector_rgb)}")


image_id = 0
for i_id, i in images.items():
    if i.name == img_name:
        print(cameras[i.camera_id])
        image_id = i_id
        break


R = read_write_model.qvec2rotmat(images[image_id].qvec)
t = images[image_id].tvec.reshape(3, 1)
extrinsic_matrix_cam2w = np.hstack((R, t))



image_position = extrinsic_matrix_cam2w @ np.array([0,0,0,1]).reshape((4,1))

vector_w = R @ vector_rgb


distance_min = np.inf
distance_min_point_id = 0

for p_id, p in points3D.items():
    point_position = np.array(p.xyz)
    
    distance_cur = np.linalg.norm(np.cross(vector_w, point_position-image_position)) / np.linalg.norm(vector_w)
    
    if distance_cur<distance_min:
        distance_min = distance_cur
        distance_min_point_id = p_id
        
print(distance_min_point_id, distance_min)



point3D_position = np.array(points3D[distance_min_point_id].xyz)

# Intrinsic parameters
K = np.array([[params[0], 0, params[2], 0],
            [0, params[1], params[3], 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

tmp = np.vstack((extrinsic_matrix_cam2w, np.array([[0, 0, 0, 1]]))) @ np.append(point3D_position.reshape(3,1), [1]) 

point2D_rgb = K @ tmp

print(point2D_rgb)

point2D_rgb = (point2D_rgb/point2D_rgb[2])[0:2].astype(int)
    # Apply intrinsic transformation
print(point2D_rgb)


import cv2
import matplotlib.pyplot as plt


# Convert the image to RGB (OpenCV loads images in BGR format)
image_rgb_mat = cv2.imread(str(image_rgb), cv2.COLOR_BGR2RGB)

# Specify the point and circle parameters
#point = (x, y)  # Replace (x, y) with your coordinates
circle_radius = 10
circle_color = (255, 0, 0)  # Red color in RGB
circle_thickness = 2

# Draw the circle on the image
#cv2.circle(image_rgb_bgr, point, circle_radius, circle_color, circle_thickness)

# Display the image with the circle

cv2.circle(image_rgb_mat, point2D_rgb, 5, (255, 0, 0), -1)

cv2.imshow("CIAI", image_rgb_mat)
cv2.waitKey()


#print(points3D) """