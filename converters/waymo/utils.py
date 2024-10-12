"""Utilities for dataset conversion."""

import cv2
import google
import numpy as np
from scipy.spatial.transform import Rotation


def rotX(deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix about the X-axis.

    Args:
        deg: Euler angle in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler("x", t).as_matrix()


def rotZ(deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix about the Z-axis.

    Args:
        deg: Euler angle in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler("z", t).as_matrix()


def rotY(deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix about the Y-axis.

    Args:
        deg: Euler angle in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler("y", t).as_matrix()


def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """ """
    q_scipy = Rotation.from_matrix(R).as_quat()
    x, y, z, w = q_scipy
    q_argo = w, x, y, z
    return q_argo


def undistort_image(
    img: np.ndarray,
    calib_data: google.protobuf.pyext._message.RepeatedCompositeContainer,
    camera_name: int,
) -> np.ndarray:
    """Undistort the image from the Waymo dataset given camera calibration data."""
    for camera_calib in calib_data:
        if camera_calib.name == camera_name:
            f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = camera_calib.intrinsic
            # k1, k2 and k3 are the tangential distortion coefficients
            # p1, p2 are the radial distortion coefficients
            camera_matrix = np.array([[f_u, 0, c_u], [0, f_v, c_v], [0, 0, 1]])
            dist_coeffs = np.array([k1, k2, p1, p2, k3])
            return cv2.undistort(img, camera_matrix, dist_coeffs)
