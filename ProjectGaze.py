import numpy as np
import sys
import pycolmap
import read_write_model

def load_camera_parameters_from_colmap_model(model_path):
    cameras, images, points3D = read_write_model.read_model(path=model_path, ext='.bin')
    
    camera_parameters = {}
    
    for image_id, image in images.items():
        camera = cameras[image.camera_id]
        
        # Extrinsic parameters
        R = read_write_model.qvec2rotmat(image.qvec)
        t = image.tvec.reshape(3, 1)
        extrinsic_matrix = np.hstack((R, t))
        
        # Intrinsic parameters
        if camera.model == 'PINHOLE' or camera.model == 'OPENCV':
            K = np.array([[camera.params[0], 0, camera.params[2]],
                          [0, camera.params[1], camera.params[3]],
                          [0, 0, 1]])
        elif camera.model == 'SIMPLE_PINHOLE' or camera.model == 'SIMPLE_RADIAL' or camera.model == 'RADIAL':
            K = np.array([[camera.params[0], 0, camera.params[1]],
                          [0, camera.params[0], camera.params[2]],
                          [0, 0, 1]])
        else:
            raise Exception(f"Camera model {camera.model} is not supported")

        # Store the parameters
        camera_parameters[image_id] = {'image_name':image.name ,'intrinsic': K, 'extrinsic': extrinsic_matrix}
    
    return camera_parameters

import configparser
config  = configparser.ConfigParser()
config.read('config.ini')
model_path = config['StructureFromMotion']['model_path']
camera_parameters = load_camera_parameters_from_colmap_model(model_path)

for image_id, params in camera_parameters.items():
    print(f"Image ID: {image_id}, NAME: {params['image_name']}")
    print("Intrinsic Matrix:")
    print(params['intrinsic'])
    print("Extrinsic Matrix:")
    print(params['extrinsic'])


outfile = '/Users/jacksalici/Desktop/SfmTesting/Test2/GazeOutput/img0.npz'
npz = np.load(outfile)

    
def reproject_point(camera_params_1, camera_params_2, point3D):

    # Extract parameters for the first and second camera
    K1, extrinsic1 = camera_params_1['intrinsic'], camera_params_1['extrinsic']
    K2, extrinsic2 = camera_params_2['intrinsic'], camera_params_2['extrinsic']
    
    # Transform the 3D point from the first camera frame to the world frame
    point3D_homogeneous = np.append(point3D, 1)  # Convert to homogeneous coordinates
    world_point = np.linalg.inv(np.append(extrinsic1, [[0, 0, 0, 1]], axis=0)) @ point3D_homogeneous
    
    point3D_cam2_homogeneous = np.append(extrinsic2, [[0, 0, 0, 1]], axis=0) @ world_point
    
    # Project the 3D point to the image plane of the second camera
    point2D_cam2_homogeneous = K2 @ point3D_cam2_homogeneous[:3]
    
    # Convert to inhomogeneous coordinates
    point2D_cam2 = point2D_cam2_homogeneous[:2] / point2D_cam2_homogeneous[2]
    
    return point2D_cam2

def reproject_camera_position(camera_params_1, camera_params_2):
    # Extract parameters for the first and second camera
    K2, extrinsic2 = camera_params_2['intrinsic'], camera_params_2['extrinsic']
    
    # First camera position in world coordinates
    R1, t1 = camera_params_1['extrinsic'][:, :3], camera_params_1['extrinsic'][:, 3:]
    cam1_position_world = -R1.T @ t1  # This is the origin (0, 0, 0) of the first camera frame transformed to world coordinates
    
    # Transform the first camera position from world frame to the second camera frame
    R2, t2 = extrinsic2[:, :3], extrinsic2[:, 3:]
    cam1_position_cam2 = R2 @ cam1_position_world + t2
    
    # Project this position onto the image plane of the second camera
    cam1_position_cam2_homogeneous = K2 @ cam1_position_cam2
    cam1_position_2D_cam2 = cam1_position_cam2_homogeneous[:2] / cam1_position_cam2_homogeneous[2]
    
    return cam1_position_2D_cam2.reshape(1, -1)[0]


image_id_1 = 1
image_id_2 = 2
point3D = npz['gaze_center_in_rgb_frame']

camera_params_1 = camera_parameters[image_id_1]
camera_params_2 = camera_parameters[image_id_2]

point2D_cam2 = reproject_point(camera_params_1, camera_params_2, point3D).astype(int)

print(f"Reprojected 2D point in the second camera frame: {point2D_cam2}")

cam1_position_2D_cam2 = reproject_point(camera_params_1, camera_params_2).astype(int)
print(f"Reprojected position of the first camera in the second camera frame: {cam1_position_2D_cam2}")


import cv2, os



img = cv2.imread(os.path.join(config['StructureFromMotion']['dataset_path'],
                 camera_parameters[image_id_2]['image_name'] ))




cv2.line(img,point2D_cam2,cam1_position_2D_cam2,(0,255,255),5, 2)

cv2.imshow("test", img)
cv2.waitKey()

img = cv2.imread(os.path.join(config['StructureFromMotion']['dataset_path'],
                 camera_parameters[image_id_1]['image_name'] ))




cv2.circle(img,npz['gaze_center_in_rgb_pixels'].astype(int), 2,(255,0,0),2)

cv2.imshow("test", img)
cv2.waitKey()