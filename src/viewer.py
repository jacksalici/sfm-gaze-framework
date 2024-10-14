from s5_gazeTo3dPoints import reproject_point, calc_camera_parameters
import pandas as pd
import os
import numpy as np
from external.read_write_model import Camera, read_model, qvec2rotmat
from pathlib import Path

def get_paired_path(image_file_path):
        parts = image_file_path.split('/')
        file_id = str(os.path.splitext(parts[-1])[0])  
        new_device_code = '1WM093700T1276'  
        return (
            os.path.join('/'.join(parts[:-4]), "Frames", parts[-3], new_device_code, f"{file_id}.jpg"), 
            os.path.join('/'.join(parts[:-4]), "Frames", parts[-3], parts[-2], f"{file_id}.jpg")
        )

def viewer2D(csv_gaze_file, colmap_images):
    import cv2, ast
    
    df = pd.read_csv(csv_gaze_file)

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Assign each column's value to a variable
        npz_file_path = row['image_file_path']
        cpf = np.fromstring(row['cpf'].strip()[1:-1], sep=' ')
        nearest_point3d = np.fromstring(row['nearest_point3d'].strip()[1:-1], sep=' ')
        distance_min = row['distance_min']
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
        
        print(colmap_images[colmap_image_id])
        
        if colmap_image_id != -1 and colmap_image_id_fpv != -1:
                 
            
            
            print("######### FOUND IMAGE N°", colmap_image_id)
        
            E, K = calc_camera_parameters(colmap_images[colmap_image_id], npz_file)
        
            gaze_point_on_tpv = K @ reproject_point(E, nearest_point3d)
            gaze_point_on_tpv = (gaze_point_on_tpv / gaze_point_on_tpv[2])[:2]
            
            #E_fpv, K_fpv = calc_camera_parameters(colmap_images[colmap_image_id_fpv], npz_file)
    
            r_cpf = K @ reproject_point(E, cpf)
            r_cpf = (r_cpf / r_cpf[2])[:2]
    
            img = cv2.imread(img_path)
            if img is None:
                print(f"Image not found")
                continue 
            
            if  0 <= gaze_point_on_tpv[0] < img.shape[0] and 0 <= gaze_point_on_tpv[1] < img.shape[1]:
                gaze_point_on_tpv = (int(gaze_point_on_tpv[0]), int(gaze_point_on_tpv[1]))  
                img = cv2.circle(img, gaze_point_on_tpv, 4, ( 0, 255, 255 ) , 2)   
                print ("added gazed point")
            
            print(r_cpf)
            if  0 <= r_cpf[0] < img.shape[0] and 0 <= r_cpf[1] < img.shape[1]:
                r_cpf = (int(r_cpf[0]), int(r_cpf[1]))  
                img = cv2.circle(img, r_cpf, 4, ( 0, 255, 255 ) , 2) 
                print ("added cpf")
            
            fpv_img = cv2.imread(fpv_path)
            fpv_img = cv2.circle(fpv_img, npz_file["gaze_center_in_rgb_pixels"], 4, (255, 0, 255), 4)
            cv2.imshow("Image Viewer FPV", fpv_img)

            cv2.imshow("Image Viewer TPV", img)

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
    parser.add_argument('--sfm', required=False, default="/Volumes/jck-wrk-hdd/Aria Recordings/Models/6_1_1/sfm")
    parser.add_argument('--model', '-m', default=False, action="store_true")

    parser.add_argument('--gaze', required=False, default="/Volumes/jck-wrk-hdd/Aria Recordings/Output/6_1_1.csv")

    args = parser.parse_args()
    

    if args.sfm:
        cameras, images, points3D = read_model(Path(args.sfm), ext=".bin")
        printModelImages(images)

        if args.model:
            previewModel(args.sfm)
            
        if args.gaze:
            viewer2D(args.gaze, images)

    
