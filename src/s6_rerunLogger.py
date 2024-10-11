#!/usr/bin/env python3
"""Rerun to log and visualize the output of COLMAP's sparse reconstruction.
Partially taken from https://github.com/rerun-io/rerun/blob/latest/examples/python/structure_from_motion"""

from __future__ import annotations

import io
import os
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
import pandas as pd
from external.read_write_model import Camera, read_model, qvec2rotmat

import tomllib
config = tomllib.load(open("config.toml", "rb"))

FILTER_MIN_VISIBLE = 20

def get_cpf_and_reprojected_point3d(image_file_path, df: pd.DataFrame):
    row = df[df['image_file_path'].str.endswith("/".join(image_file_path.split("/")[-3:]))]
    if not row.empty:

        cpf = np.fromstring(str(row['cpf'].values[0]).strip()[1:-1], sep=" ")
        gaze_vector = np.fromstring(str(row['gaze_vector'].values[0]).strip()[1:-1],  sep=" ")
        reprojected_point3d = np.fromstring(str(row['reprojected_point3d'].values[0]).strip()[1:-1],  sep=" ")

        nearest_p = np.fromstring(str(row['nearest_point3d'].values[0]).strip()[1:-1],  sep=" ")

        return reprojected_point3d, cpf, nearest_p, gaze_vector

    return None, None, None, None

def import_model(model_path: Path):
    print("INFO: Reading sparse COLMAP reconstruction")
    cameras, images, points3D = read_model(model_path, ext=".bin")
    return cameras, images, points3D


def read_and_log_sparse_reconstruction(
    cameras, images, points3D, dataset_path: Path, gaze_path:Path, df: pd.DataFrame
) -> None:
    

    print("INFO: Building visualization by logging to Rerun")

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("plot/avg_reproj_err", rr.SeriesLine(color=[240, 45, 58]), static=True)

    # Iterate through images (video frames) logging data related to each frame.
    points = [point.xyz for point in points3D.values()]
    point_colors = [point.rgb for point in points3D.values()]
    point_errors = [point.error for point in points3D.values()]

    rr.log(
        "/points",
        rr.Points3D(points, colors=point_colors),
        rr.AnyValues(error=point_errors),
        static=True,
    )

    for image in sorted(images.values(), key=lambda im: im.name.split('/')[1]):  # type: ignore[no-any-return]
        name_parts = (image.name).split('/')
        
        rr.set_time_nanos("sync_timestamp", int(name_parts[1][:-4]))
        
        
        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]


        visible = [
            id != -1 and points3D.get(id) is not None for id in image.point3D_ids
        ]
        visible_ids = image.point3D_ids[visible]

        visible_xyzs = [points3D[id] for id in visible_ids]
        visible_xys = image.xys[visible]
  

        points = [point.xyz for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]
    
        rr.log("plot/avg_reproj_err", rr.Scalar(np.mean(point_errors)))

        #rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))

        # COLMAP's camera transform is "camera from world"
        rr.log(
            f"camera{name_parts[0]}",
            rr.Transform3D(
                translation=image.tvec,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                from_parent=True,
            ),
            #static=True,
        )
        rr.log(
            f"camera{name_parts[0]}",
            rr.ViewCoordinates.RDF,   static=True
        ) # X=Right, Y=Down, Z=Forward
 
        file = os.path.join(
             str(gaze_path), f"{(image.name).split('/')[1].split('.')[0]}.npz"
        )

        if os.path.isfile(file):

            npz_file = np.load(file)
            # use exact intrinsic
            params = npz_file["rbg_camera_intrinsic"]
            f, c = params[:2], params[2:]

        else:
            print("WARNING: npz file not loaded", file)
            f, c = camera.params[:2], camera.params[2:]

            if camera.model != "PINHOLE":
                f, c = [camera.params[1], camera.params[2]], [
                    camera.params[0],
                    camera.params[3],
                ]

        rr.log(
            f"camera{name_parts[0]}/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=f,
                principal_point=c,
            ),
        )


        rr.log(
            f"camera{name_parts[0]}/image",
            rr.ImageEncoded(path=dataset_path / name_parts[0] / name_parts[1]),
            )

        #rr.log(    f"camera{name_parts[0]}/image/keypoints",rr.Points2D(visible_xys, colors=[34, 138, 167]),# static=True)

        try:
            gaze3d, cpf3d, nearest_point, gaze_vector = get_cpf_and_reprojected_point3d(file, df)
            add_gaze_direction(gaze3d, cpf3d)
            add_gaze_direction_from_point(image, npz_file)
            rr.log(
            f"/nearest_point",
            rr.Points3D([nearest_point]),
            )
            
            rr.log(
            f"/gaze_point",
            rr.Points3D([gaze3d]),
            )
            
            rr.log(
                "gaze_vector",
                rr.Arrows3D(
                    vectors=[gaze_vector-cpf3d],
                    origins=[cpf3d],
                    colors=[1, 0.7, 0.7],
                ),
            )
        except Exception as e:
            print("ERROR: ", e )
        

def add_gaze_direction( vector_w, cpf_w):
    
    rr.log(
        "gaze",
        rr.Arrows3D(
            vectors=[vector_w-cpf_w],
            origins=[cpf_w],
            colors=[1, 0.7, 0.7],
        ),
    )
    
    
from s5_gazeTo3dPoints import reproject_point, calc_camera_parameters, inv_transformation_matrix

def add_gaze_direction_from_point(image, npz_file):
    E, K = calc_camera_parameters(image, npz_file)

    gaze_cpf = npz_file["gaze_center_in_cpf"]
    
    E_cpf2rgb = inv_transformation_matrix(npz_file["rbg2cpf_camera_extrinsic"])

    cpf_rgb = reproject_point(E_cpf2rgb, [0, 0, 0])
    cpf_w = reproject_point(E, cpf_rgb)
    
    gaze_rgb = reproject_point(E_cpf2rgb, gaze_cpf)
    gaze_w = reproject_point(E, gaze_rgb)
    
    rr.log(
        "gaze_computed_in_rerun",
        rr.Arrows3D(
            vectors=[(gaze_w-cpf_w)*10],
            origins=[cpf_w],
            colors=[0.7, 1, 0.7],
        ),
    )
    
    
    
   

def main() -> None:
    parser = ArgumentParser()

    rr.script_add_args(parser)
    
    parser.add_argument('--scene', '-s', type=str, required=False, default="5_1_1")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(config["gaze_estimation"]["gaze_3d_output_folder"], f"{args.scene}.csv"))

    blueprint = rrb.Vertical(
        rrb.Spatial3DView(name="3D", origin="/"),
        rrb.Horizontal(
            rrb.TimeSeriesView(origin="/plot"),
            rrb.Spatial2DView(name="FPV Camera", origin=f"/camera{config['gaze_estimation']['eye_tracking_device_id']}/image")
        )

    )

    rr.script_setup(
        args, "sfm_gaze_dataset", default_blueprint=blueprint
    )
    folder = args.scene
    cameras, images, points3D = import_model(
        Path(config["aria_recordings"]["model_path"]) / folder / "sfm"
    )

    read_and_log_sparse_reconstruction(
        cameras,
        images,
        points3D,
        Path(config["aria_recordings"]["frames_path_root"]) / folder,
        Path(config["aria_recordings"]["gaze_output"]) / folder / config["gaze_estimation"]["eye_tracking_device_id"] ,
        df
    )


    rr.script_teardown(args)


if __name__ == "__main__":
    main()
