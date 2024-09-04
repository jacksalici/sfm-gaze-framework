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

from external.read_write_model import Camera, read_model, qvec2rotmat

import tomllib
config = tomllib.load(open("config.toml", "rb"))

FILTER_MIN_VISIBLE = 20
FPV_IMAGE_NAME = "img8rgb.jpg"

PHOTOS = "1"  # ALL / ALL_RGB / 1 / None


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
    T = E[:3, -1].reshape((3,1))
    assert R.shape == (3,3) and T.shape == (3, 1)
    return np.hstack((R.T,-R.T@T))
    


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
    point3d_E = (
        E
        @ point3D_homogeneous
    )

    return point3d_E[:3]


def import_model(model_path: Path):
    print("INFO: Reading sparse COLMAP reconstruction")
    cameras, images, points3D = read_model(model_path, ext=".bin")
    return cameras, images, points3D


def read_and_log_sparse_reconstruction(
    cameras, images, points3D, dataset_path: Path, resize: tuple[int, int] | None
) -> None:

    print("INFO: Building visualization by logging to Rerun")

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
    # rr.log("plot/avg_reproj_err", rr.SeriesLine(color=[240, 45, 58]))

    # Iterate through images (video frames) logging data related to each frame.
    points = [point.xyz for point in points3D.values()]
    point_colors = [point.rgb for point in points3D.values()]
    point_errors = [point.error for point in points3D.values()]

    rr.log(
        "/points",
        rr.Points3D(points, colors=point_colors),
        rr.AnyValues(error=point_errors),
        timeless=True,
    )

    for image in sorted(images.values(), key=lambda im: im.name):  # type: ignore[no-any-return]
        image_file = dataset_path / image.name

        if PHOTOS == "ALL_RGB" and "rgb" not in image.name:
            continue

        elif PHOTOS == None:
            continue

        elif PHOTOS == "1" and FPV_IMAGE_NAME != image.name:
            continue

        if not os.path.exists(image_file):
            continue

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        if resize:
            camera, scale_factor = scale_camera(camera, resize)
        else:
            scale_factor = np.array([1.0, 1.0])

        visible = [
            id != -1 and points3D.get(id) is not None for id in image.point3D_ids
        ]
        visible_ids = image.point3D_ids[visible]

        visible_xys = image.xys[visible]
        if resize:
            visible_xys *= scale_factor

        # rr.set_time_sequence("frame", frame_idx)

        # COLMAP's camera transform is "camera from world"
        rr.log(
            f"camera{image.camera_id}",
            rr.Transform3D(
                translation=image.tvec,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                from_parent=True,
            ),
            # timeless=True,
        )
        rr.log(
            f"camera{image.camera_id}",
            rr.ViewCoordinates.RDF,  # timeless=True
        )  # X=Right, Y=Down, Z=Forward

        file = os.path.join(
            config["gaze_output_path"], f"img{image.name[3]}.npz"
        )

        if os.path.isfile(file):

            npz_file = np.load(file)
            # use exact intrinsic
            params = npz_file["rbg_camera_intrinsic"]
            f, c = params[:2], params[2:]

        else:
            f, c = camera.params[:2], camera.params[2:]

            if camera.model != "PINHOLE":
                f, c = [camera.params[1], camera.params[2]], [
                    camera.params[0],
                    camera.params[3],
                ]

        rr.log(
            f"camera{image.camera_id}/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=f,
                principal_point=c,
            ),
            # timeless=True,
        )

        if resize:
            bgr = cv2.imread(str(image_file))
            bgr = cv2.resize(bgr, resize)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log(
                f"camera{image.camera_id}/image",
                rr.Image(rgb).compress(jpeg_quality=75),
            )
        else:
            rr.log(
                f"camera{image.camera_id}/image",
                rr.ImageEncoded(path=dataset_path / image.name),
                # timeless=True
            )

        rr.log(
            f"camera{image.camera_id}/image/keypoints",
            rr.Points2D(visible_xys, colors=[34, 138, 167]),
            # timeless=True
        )


def calc_fpv_camera_parameters(cameras, images):

    for image_id, image in images.items():
        if image.name != FPV_IMAGE_NAME:
            continue
        
        camera = cameras[image.camera_id]

        # Extrinsic parameters
        R = qvec2rotmat(image.qvec)
        t = image.tvec.reshape(3, 1)
        E = np.hstack((R, t))

        # Intrinsic parameters
        file = os.path.join(
            config["gaze_output_path"], f"img{image.name[3]}.npz"
        )

        if os.path.isfile(file):

            npz_file = np.load(file)
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
            
        else:
            if camera.model == "PINHOLE" or camera.model == "OPENCV":
                K = np.array(
                    [
                        [camera.params[0], 0, camera.params[2]],
                        [0, camera.params[1], camera.params[3]],
                        [0, 0, 1],
                    ]
                )
            elif (
                camera.model == "SIMPLE_PINHOLE"
                or camera.model == "SIMPLE_RADIAL"
                or camera.model == "RADIAL"
            ):
                K = np.array(
                    [
                        [camera.params[0], 0, camera.params[1]],
                        [0, camera.params[0], camera.params[2]],
                        [0, 0, 1],
                    ]
                )
            else:
                raise Exception(f"Camera model {camera.model} is not supported")

    return E, K


def add_gaze_direction(cameras, images):
    E, K = calc_fpv_camera_parameters(cameras, images)

    npz_file = np.load(
        os.path.join(
            config["gaze_output_path"],
            FPV_IMAGE_NAME[0:4] + ".npz",
        )
    )


    gaze_yaw_pitch = npz_file["gaze_yaw_pitch"]
    yaw_cpf, pitch_cpf = gaze_yaw_pitch[0], gaze_yaw_pitch[1]

    vector_cpf = pitch_yaw_to_vector(pitch_cpf, yaw_cpf)
    
    E_cpf2rgb = npz_file["rbg2cpf_camera_extrinsic"]
    
    cpf_w = reproject_point(E, reproject_point(E_cpf2rgb, [0, 0, 0]))
    
    vector_rgb = (inv_transformation_matrix(E_cpf2rgb) @ np.append(vector_cpf, [1]))[:3]

    vector_w = reproject_point(E, vector_rgb)
    
    
    rr.log(
        "gaze",
        rr.Arrows3D(
            vectors=[(vector_w-cpf_w)*10],
            origins=[cpf_w],
            colors=[1, 0.7, 0.7],
        ),
    )
    
    return vector_w, cpf_w

def add_gaze_direction_from_point(cameras, images):
    E, K = calc_fpv_camera_parameters(cameras, images)

    npz_file = np.load(
        os.path.join(
            config["gaze_output_path"],
            FPV_IMAGE_NAME[0:4] + ".npz",
        )
    )
    
    gaze_rgb = npz_file["gaze_center_in_rgb_frame"]
    
    rgb_w = reproject_point(E, [0, 0, 0])
    gaze_w = reproject_point(E, gaze_rgb)
    
    rr.log(
        "gaze2",
        rr.Arrows3D(
            vectors=[(gaze_w-rgb_w)*10],
            origins=[rgb_w],
            colors=[0.7, 1, 0.7],
        ),
    )
    
   

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
        
        distance_cur = np.linalg.norm(np.cross(vector-origin, point_position-origin)) / np.linalg.norm(vector-origin)
        
        if distance_cur<distance_min:
            distance_min = distance_cur
            distance_min_point_id = p_id
            
    print(distance_min_point_id, distance_min)



    point3D_position = np.array(points3D[distance_min_point_id].xyz)
    
    rr.log(
        "/nearestPoint",
        rr.Points3D(point3D_position, colors=[0,0,0]),
        timeless=True,
    )


def main() -> None:
    parser = ArgumentParser(
        description="Visualize the output of COLMAP's sparse reconstruction on a video."
    )
    parser.add_argument(
        "--resize", action="store", help="Target resolution to resize images"
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    if args.resize:
        args.resize = tuple(int(x) for x in args.resize.split("x"))

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D", origin="/"),
    )

    rr.script_setup(
        args, "rerun_example_structure_from_motion", default_blueprint=blueprint
    )

    cameras, images, points3D = import_model(
        Path(config["model_path"]) / "sfm"
    )

    read_and_log_sparse_reconstruction(
        cameras,
        images,
        points3D,
        Path(config["dataset_path"]),
        resize=args.resize,
    )

    cpf, vector = add_gaze_direction(cameras, images)
    select_nearest(vector, cpf, points3D)
    add_gaze_direction_from_point(cameras, images)


    rr.script_teardown(args)


if __name__ == "__main__":
    main()
