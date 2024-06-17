import numpy as np
import sys
import pycolmap
import read_write_model
import cv2, os

import configparser
config  = configparser.ConfigParser()
config.read('config.ini')

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
    
def reproject_point(camera_params_1, camera_params_2, point3D):
    """Reproject 3d point from the frame of camera 1 to the frame of camera 2.

    Args:
        camera_params_1 (dict): ColMap camera params dict. Must contains "intrinsic" and "extrinsic" key.
        camera_params_2 (dict): As above
        point3D (list): 3-value list as camera 1 frame coordinates. 

    Returns:
        list: 2d point expressed as pixel of the 2 camera image.
    """

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

def main():
    
    model_path = os.path.join(config['StructureFromMotion']['model_path'], "sfm")
    camera_parameters = load_camera_parameters_from_colmap_model(model_path)

    for image_id, params in camera_parameters.items():
        print(f"INFO: Image ID: {image_id}, NAME: {params['image_name']}")
        print("INFO: Intrinsic Matrix:")
        print(params['intrinsic'])
        print("INFO: Extrinsic Matrix:")
        print(params['extrinsic'])
        
     
    
    FPV_image_id = int(config['StructureFromMotion']['FPV_image_id'])
    fpv_camera_params = camera_parameters[FPV_image_id]
    
    
    npz_file = np.load(os.path.join(config['StructureFromMotion']['gaze_output_path'], camera_parameters[FPV_image_id]['image_name'][:-4] + '.npz'))

    point3D = npz_file['gaze_center_in_rgb_frame']
    
    img_fpv = cv2.imread(os.path.join(config['StructureFromMotion']['dataset_path'],
                            camera_parameters[FPV_image_id]['image_name'] ))

    cv2.circle(img_fpv,npz_file['gaze_center_in_rgb_pixels'].astype(int), 4,(255,255,0),3)

    for image_id, params in camera_parameters.items():
        if image_id != FPV_image_id:
            camera_params = camera_parameters[image_id]

            point2D_cam2 = reproject_point(fpv_camera_params, camera_params, point3D).astype(int)

            print(f"INFO: Reprojected 2D point in the {image_id} camera frame: {point2D_cam2}")

            cam1_position_2D_cam2 = reproject_point(fpv_camera_params, camera_params, [0, 0, 0]).astype(int)
            print(f"Reprojected position of the FPV camera in the {image_id} camera frame: {cam1_position_2D_cam2}")



            img = cv2.imread(os.path.join(config['StructureFromMotion']['dataset_path'],
                            camera_parameters[image_id]['image_name'] ))


            cv2.line(img,point2D_cam2,cam1_position_2D_cam2,(255,255,0),5, 2)

            cv2.imshow(f"GAZE REPROJECTED IN FRAME {image_id}", img)
            cv2.imshow("FPV FRAME", img_fpv)
            cv2.waitKey()
            
    
if __name__ == "__main__":
    main()