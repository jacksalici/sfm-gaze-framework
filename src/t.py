import numpy as np



def pitch_yaw_to_vector(yaw_rad, pitch_rad):
    # inspired by https://github.com/facebookresearch/projectaria_tools/blob/3f6079ffcd21b8975fed2ce2bef211473bc498ad/core/mps/EyeGazeReader.h#L40

    x = np.tan(yaw_rad)
    y = np.tan(pitch_rad)
    z = 1

    direction = np.array([x, y, z])
    return direction / np.linalg.norm(direction)


def get_eyegaze_point_at_depth(y, p, d):
    return pitch_yaw_to_vector(y, p) * d

gaze_x, gaze_y, gaze_z = get_eyegaze_point_at_depth(
            -0.102910490016660, -0.288851886987686, 1.179526637404006
        )

print(gaze_x, gaze_y, gaze_z)