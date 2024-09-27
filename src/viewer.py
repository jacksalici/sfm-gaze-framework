from s5_gazeTo3dPoints import reproject_point, calc_camera_parameters
import pandas as pd
import os
import numpy as np
from external.read_write_model import Camera, read_model, qvec2rotmat
from pathlib import Path

def project_point_proportionally(point, reference, img_shape):
    h, w = img_shape[:2]  # Image dimensions (height, width)
    
    # Vector from reference (center) to point
    vector_x = point[0] - reference[0]
    vector_y = point[1] - reference[1]

    # Calculate scaling factors for both x and y directions
    if vector_x > 0:
        scale_x = (w - 1 - reference[0]) / vector_x
    else:
        scale_x = (0 - reference[0]) / vector_x
    
    if vector_y > 0:
        scale_y = (h - 1 - reference[1]) / vector_y
    else:
        scale_y = (0 - reference[1]) / vector_y

    # Use the smaller scaling factor to maintain proportions
    scale = min(scale_x, scale_y)

    # Project point onto the border
    proj_x = reference[0] + vector_x * scale
    proj_y = reference[1] + vector_y * scale

    return int(proj_x), int(proj_y)

def get_paired_path(image_file_path):
        parts = image_file_path.split('/')
        file_id = os.path.splitext(parts[-1])[0]  
        new_device_code = '1WM093700T1276'  
        return  '/'.join(parts[:-4]) + f"/Frames/{parts[-3]}/{new_device_code}/{file_id}.jpg", '/'.join(
            parts[:-4]) + f"/Frames/{parts[-3]}/{parts[-2]}/{file_id}.jpg"
    

def viewer2D(csv_gaze_file, colmap_images):
    import cv2, ast
    
    df = pd.read_csv(csv_gaze_file)

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Assign each column's value to a variable
        npz_file_path = row['image_file_path']
        cpf = np.fromstring(row['cpf'].strip()[1:-1], sep=' ')
        nearest_point3d = row['nearest_point3d']
        distance_min = row['distance_min']
        print(row['reprojected_point3d'])
        reprojected_point3d = np.fromstring(row['reprojected_point3d'].strip()[1:-1], sep=' ')
        
        npz_file = np.load(
            npz_file_path
        )
        img_path, fpv_path = get_paired_path(npz_file_path)
        
        colmap_image_id = -1
        colmap_image_id_fpv = -1
        for key, image in colmap_images.items():
            if img_path.endswith(image.name):
                colmap_image_id = key
            if fpv_path.endswith(image.name):
                colmap_image_id_fpv = key
        
        
        if colmap_image_id != -1 and colmap_image_id_fpv != -1:
            print("######### FOUND IMAGE N°", colmap_image_id)
        
            E, K = calc_camera_parameters(colmap_images[colmap_image_id], npz_file)
        
            point = K @ reproject_point(E, reprojected_point3d, inv=True)
            point = (point / point[2])[:2]
            
            E, K = calc_camera_parameters(colmap_images[colmap_image_id_fpv], npz_file)
    
            r_cpf = K @ reproject_point(E, cpf, inv=True)
            r_cpf = (r_cpf / r_cpf[2])[:2]
    
            img = cv2.imread(img_path)
            print(img.shape)
            if img is None:
                print(f"Image not found")
                continue 
            
            print(point)
            if  0 <= point[0] < img.shape[0] and 0 <= point[1] < img.shape[1]:
                point = (int(point[0]), int(point[1]))  
                img = cv2.circle(img, point, 4, ( 0, 255, 255 ) , 2)   
                print ("added gazed point")
            
            print(r_cpf)
            if  0 <= r_cpf[0] < img.shape[0] and 0 <= r_cpf[1] < img.shape[1]:
                r_cpf = (int(r_cpf[0]), int(r_cpf[1]))  
                img = cv2.circle(img, r_cpf, 4, ( 0, 255, 255 ) , 2) 
                print ("added cpf")
            
            fpv_img = cv2.imread(fpv_path)
            fpv_img = cv2.circle(fpv_img, npz_file["gaze_center_in_rgb_pixels"], 4, (0, 255, 255), 4)
            cv2.imshow("Image Viewer 2", fpv_img)

            cv2.imshow("Image Viewer", img)

            key = cv2.waitKey(0)  
            
            if key == ord('q'):
                break
            
   
def previewModel(path):
    from hloc.utils import viz_3d
    from pycolmap import Reconstruction

    model = Reconstruction(path)

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
            fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
    )
    fig.show()
    

def printModelImages(images) -> None:
    for image in images.values():
        print(image.name)
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preview stuffs.")
    parser.add_argument('--sfm', required=False)
    parser.add_argument('--model', '-m', default=False, action="store_true")
    parser.add_argument('--gaze', required=False)

    args = parser.parse_args()
    

    if args.sfm:
        cameras, images, points3D = read_model(Path(args.sfm), ext=".bin")
        printModelImages(images)

        if args.model:
            previewModel(args.sfm)
            
        if args.gaze:
            viewer2D(args.gaze, images)

    
