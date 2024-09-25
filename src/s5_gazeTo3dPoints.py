#!/usr/bin/env python3

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


from external.read_write_model import Camera, read_model, qvec2rotmat


import csv




def scale_camera(
    camera: Camera, resize: tuple[int, int]
) -> tuple[Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    assert camera.model == "PINHOLE"
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    # For PINHOLE camera model, params are: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
    new_params = np.append(
        camera.params[:2] * scale_factor, camera.params[2:] * scale_factor
    )

    return (
        Camera(camera.id, camera.model, new_width, new_height, new_params),
        scale_factor,
    )


def inv_transformation_matrix(E):
    R = E[:3, :3]
    T = E[:3, -1].reshape((3, 1))
    assert R.shape == (3, 3) and T.shape == (3, 1)
    return np.hstack((R.T, -R.T @ T))


def reproject_point(E, point3D, inv=True):
    """Reproject 3d point from the frame of camera 1 to the global frame

    E: Extrinsic Matrix from "World Space" to Point Space

    Returns:
        list: 2d point expressed as pixel of the 2 camera image.
    """

    if inv:
        E = inv_transformation_matrix(E)

    if E.shape == (3, 4):
        E = np.append(E, [[0, 0, 0, 1]], axis=0)

    point3D_homogeneous = np.append(point3D, 1)  # Convert to homogeneous coordinates
    point3d_E = E @ point3D_homogeneous

    return point3d_E[:3]


def import_model(model_path: Path):
    print("INFO: Reading sparse COLMAP reconstruction")
    cameras, images, points3D = read_model(model_path, ext=".bin")
    return cameras, images, points3D


def calc_camera_parameters(image, npz_file):

    # Extrinsic parameters
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    E = np.hstack((R, t))



        # use exact intrinsic
    params = npz_file["rbg_camera_intrinsic"]
    f, c = params[:2], params[2:]
    K = np.array(
            [
                [f[0], 0, c[0]],
                [0, f[1], c[1]],
                [0, 0, 1],
            ]
        )

    return E, K


def add_gaze_direction(image, npz_file):
    E, K = calc_camera_parameters(image, npz_file)

    gaze_yaw_pitch = npz_file["gaze_yaw_pitch"]
    yaw_cpf, pitch_cpf = gaze_yaw_pitch[0], gaze_yaw_pitch[1]

    vector_cpf = pitch_yaw_to_vector(pitch_cpf, yaw_cpf)

    E_cpf2rgb = npz_file["rbg2cpf_camera_extrinsic"]

    cpf_w = reproject_point(E, reproject_point(E_cpf2rgb, [0, 0, 0]))

    vector_rgb = (inv_transformation_matrix(E_cpf2rgb) @ np.append(vector_cpf, [1]))[:3]

    vector_w = reproject_point(E, vector_rgb)

    return vector_w, cpf_w


def pitch_yaw_to_vector(yaw_rad, pitch_rad):
    # inspired by https://github.com/facebookresearch/projectaria_tools/blob/3f6079ffcd21b8975fed2ce2bef211473bc498ad/core/mps/EyeGazeReader.h#L40

    x = np.tan(yaw_rad)
    y = np.tan(pitch_rad)
    z = 1

    direction = np.array([x, y, z])
    return direction / np.linalg.norm(direction)


def select_nearest(vector, origin, points3D):
    distance_min = np.inf
    distance_min_point_id = 0

    for p_id, p in points3D.items():
        point_position = np.array(p.xyz)

        distance_cur = np.linalg.norm(
            np.cross(vector - origin, point_position - origin)
        ) / np.linalg.norm(vector - origin)

        if distance_cur < distance_min:
            distance_min = distance_cur
            distance_min_point_id = p_id

    print(f"MIN DISTANCE: {distance_min}")

    point3D_position = np.array(points3D[distance_min_point_id].xyz)
    return point3D_position, distance_min

def find_lying_point(vector, cpf, point):
    distance = np.linalg.norm(cpf-point)
    v_norm = vector / np.linalg.norm(vector)
    return cpf + distance * v_norm
    

def save_info(csv_file, image_file_path, cpf, gaze_vector, point3D_position, distance_min):
    print("INFO: Saving image gaze")
    csv_file_exists = True

    if not os.path.exists(csv_file):
        csv_file_exists = False

    with open(csv_file, "a") as f:
        writer = csv.writer(f)

        if not csv_file_exists:
            # save header once
            fields = [
                "image_file_path",
                "cpf",
                "gaze_vector",
                "nearest_point3d",
                "distance_min",
                "reprojected_point3d"
            ]
            writer.writerow(fields)

        fields = [
            image_file_path, cpf, gaze_vector, point3D_position, distance_min, find_lying_point(gaze_vector, cpf, point3D_position)
        ]
        writer.writerow(fields)


def gaze2points(csv_file, model_path, gaze_base_path, eye_tracking_device_id ) -> None:

    cameras, images, points3D = import_model(Path(model_path) / "sfm")
    for image in images.values():
        folder = (image.name).split('/')[0]
        if eye_tracking_device_id not in folder.split('_'):
            continue
        
        camera = cameras[image.camera_id]
        
        npz_file_path = os.path.join(
                gaze_base_path, (image.name).split('/')[0],
                f"{(image.name).split('/')[1][:-4]}.npz",
            )
        npz_file = np.load(
            npz_file_path
        )

        cpf, vector = add_gaze_direction(image, npz_file)
        point3D_position, distance_min = select_nearest(vector, cpf, points3D)
        save_info(csv_file, npz_file_path, cpf, vector, point3D_position, distance_min)


if __name__ == "__main__":
    import tomllib
    config = tomllib.load(open("config.toml", "rb"))
    csv_file = config["gaze_estimation"]["gaze_3d_output"]
    gaze_base_path = config["gaze_output_path"]
    model_path = config["model_path"]
    eye_tracking_device_id = config["gaze_estimation"]["eye_tracking_device_id"]
    
    gaze2points(csv_file, model_path, gaze_base_path, eye_tracking_device_id)
    

