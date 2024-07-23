import math
from typing import Final

import numba as nb
import numpy as np
import polars as pl
from av2.geometry.se3 import SE3
from scipy.spatial.transform import Rotation, Slerp

from torchbox3d.datasets.argoverse.constants import (
    LASER_MAPPING,
    LOG_IDS,
    ROW_MAPPING_32,
    ROW_MAPPING_64,
)

RANGE_VIEW_SCHEMA: Final = {
    "x": pl.Float32,
    "y": pl.Float32,
    "z": pl.Float32,
    "intensity": pl.UInt8,
    "laser_number": pl.UInt8,
    "is_within_roi": pl.Boolean,
    "timedelta_ns": pl.Float32,
    "range": pl.Float32,
}

TRANSLATION: Final = ("tx_m", "ty_m", "tz_m")
QUATERNION_WXYZ: Final = ("qx", "qy", "qz", "qw")


def build_range_view(
    lidar: pl.DataFrame,
    extrinsics: pl.DataFrame,
    features: np.ndarray,
    sensor_name: str,
    height: int,
    width: int,
    build_uniform_inclination: bool = False,
) -> pl.DataFrame:
    cart = lidar.select(pl.col(["x_p", "y_p", "z_p"])).to_numpy()

    lidar_offset = (
        extrinsics.filter(pl.col("sensor_name") == sensor_name)
        .select(pl.col("tx_m", "ty_m", "tz_m"))
        .to_numpy()
    )
    rotation = (
        extrinsics.filter(pl.col("sensor_name") == sensor_name)
        .select(pl.col("qx", "qy", "qz", "qw"))
        .to_numpy()
    )
    rotation = Rotation.from_quat(rotation).as_matrix().squeeze(0)
    sensor_SE3_egovehicle = SE3(
        rotation=rotation, translation=lidar_offset.squeeze(0)
    ).inverse()
    cart_lidar = sensor_SE3_egovehicle.transform_point_cloud(cart)

    laser_number = (
        lidar.select(pl.col("laser_number")).to_numpy().astype(int).squeeze(1)
    )

    timestamp_ns = lidar.select(pl.col("offset_ns")).to_numpy().astype(float).squeeze(1)
    # timestamp_ns = (offset_ns - offset_ns.min()) / (offset_ns.max() - offset_ns.min())
    # timestamp_ns = np.zeros_like(laser_number)

    laser_mapping = np.arange(height)
    sph = cart_to_sph(cart_lidar)
    hybrid_coordinates = build_range_view_coordinates(
        cart_lidar,
        sph,
        laser_numbers=laser_number,
        laser_mapping=laser_mapping,
        n_inclination_bins=height,
        n_azimuth_bins=width,
        build_uniform_inclination=build_uniform_inclination,
    )

    indices = hybrid_coordinates[:, :2].astype(int).T
    distances = hybrid_coordinates[:, -1]

    # distances = lidar.select(pl.col("offset_ns")).to_numpy().squeeze(-1)
    features = np.concatenate(
        (features, timestamp_ns[:, None], distances[:, None]), axis=-1
    ).T

    # distances = lidar.select(pl.col("offset_ns")).to_numpy().squeeze(-1)

    num_features = features.shape[0]
    features = z_buffer(indices, distances, features, height=height, width=width)
    features = features.reshape(num_features, -1)
    frame = pl.DataFrame(
        {
            "x": features[0, :],
            "y": features[1, :],
            "z": features[2, :],
            "intensity": features[3, :],
            "laser_number": features[4, :],
            "is_within_roi": features[5, :],
            "timedelta_ns": features[6, :],
            "range": features[7, :],
        },
        schema=RANGE_VIEW_SCHEMA,
    )
    return frame


def build_range_view_coordinates(
    cart: np.ndarray,
    sph: np.ndarray,
    laser_numbers: np.ndarray,
    laser_mapping: np.ndarray,
    n_inclination_bins: int,
    n_azimuth_bins: int,
    build_uniform_inclination: bool = False,
) -> np.ndarray:
    """Convert a set of points in R^3 (x,y,z) to range image of shape (n_inclination_bins,n_azimuth_bins,range).

    Args:
        cart: (N,3) Cartesian coordinates.
        sph: (N,3) Spherical coordinates.
        laser_numbers: (N,) Integer laser ids.
        laser_mapping: Mapping from laser number to row.
        n_inclination_bins: Number of inclination bins.
        n_azimuth_bins: Number of azimuthal bins.

    Returns:
        The range image containing range, intensity, and nanosecond offset.
    """
    azimuth = sph[..., 0]
    inclination = sph[..., 1]
    radius = sph[..., 2]
    azimuth += math.pi

    # azimuth = (azimuth + np.pi) % (2 * np.pi) # Center azimuth.
    azimuth *= n_azimuth_bins / math.tau
    azimuth_index = n_azimuth_bins - np.round(azimuth)
    if build_uniform_inclination:
        # FOV = np.abs(np.array([-30.0 / 180.0 * math.pi, 10 / 180.0 * math.pi]))
        FOV = np.abs(np.array([-10.0 / 180.0 * math.pi, 10 / 180.0 * math.pi]))

        fov_bottom, fov_top = FOV[0], FOV[1]
        inclination_index = 1.0 - (inclination + fov_bottom) / (fov_bottom + fov_top)
        inclination_index = inclination_index * n_inclination_bins
        inclination_index = inclination_index.round().clip(0, n_inclination_bins - 1)
    else:
        inclination_index = n_inclination_bins - laser_mapping[laser_numbers] - 1
    azimuth_index = np.clip(azimuth_index, 0, n_azimuth_bins - 1)
    hybrid = np.zeros_like(cart)
    hybrid[:, 0] = inclination_index
    hybrid[:, 1] = azimuth_index
    hybrid[:, 2] = radius
    return hybrid


def cart_to_sph(
    cart: np.ndarray,
) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    Reference:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        cart: (N,3) Cartesian coordinates.

    Returns:
        (N,3) Spherical coordinates: azimuth, inclination, and radius.
    """
    x = cart[..., 0]
    y = cart[..., 1]
    z = cart[..., 2]

    hypot_xy = np.hypot(x, y)
    radius = np.hypot(hypot_xy, z)
    inclination = np.arctan2(z, hypot_xy)
    azimuth = np.arctan2(y, x)

    sph = np.zeros_like(cart)
    sph[..., 0] = azimuth
    sph[..., 1] = inclination
    sph[..., 2] = radius
    return sph


@nb.njit(nogil=True)  # type: ignore
def z_buffer(
    indices: np.ndarray,
    distances: np.ndarray,
    features: np.ndarray,
    height: int,
    width: int,
    min_distance: float = 1.0,
) -> np.ndarray:
    num_channels = features.shape[0]
    num_pixels = height * width
    image = np.zeros((num_channels, num_pixels), dtype=np.float32)
    buffer = np.full(num_pixels, np.inf, dtype=np.float32)

    row_indices, column_indices = indices
    raveled_indices = row_indices * width + column_indices
    for i, img_idx in enumerate(raveled_indices):
        if distances[i] < min_distance:
            continue
        if distances[i] < buffer[img_idx]:
            image[:, img_idx] = features[:, i]
            buffer[img_idx] = distances[i]
    return np.ascontiguousarray(image.reshape((num_channels, height, width)))


def correct_laser_numbers(
    laser_numbers: np.ndarray, log_id: str, height: int
) -> np.ndarray:
    if log_id in LOG_IDS:
        laser_numbers[laser_numbers >= 32] = (
            LASER_MAPPING[laser_numbers[laser_numbers >= 32] - 32] + 32
        )
        laser_numbers[laser_numbers < 32] = LASER_MAPPING[
            laser_numbers[laser_numbers < 32]
        ]

    if height == 32:
        laser_mapping = ROW_MAPPING_32.copy()
    else:
        laser_mapping = ROW_MAPPING_64.copy()
    return laser_mapping[laser_numbers]


def unmotion_compensate(
    sweep: pl.DataFrame, poses: pl.DataFrame, timestamp_ns: int, slerp: Slerp
) -> pl.DataFrame:
    min_timestamp_ns = poses["timestamp_ns"].min()
    max_timestamp_ns = poses["timestamp_ns"].max()

    sweep = sweep.with_columns(
        timestamp_ns=timestamp_ns + sweep["offset_ns"].cast(pl.Int64)
    ).filter(
        (pl.col("timestamp_ns") > min_timestamp_ns)
        & (pl.col("timestamp_ns") < max_timestamp_ns)
    )
    point_timestamps_ns = sweep["timestamp_ns"]

    point_timestamps_ns_npy = point_timestamps_ns.to_numpy()
    indices = poses["timestamp_ns"].search_sorted(sweep["timestamp_ns"], side="left")
    xyz = sweep.select(["x", "y", "z"]).to_numpy()

    lower = poses[indices - 1]
    high = poses[indices]

    ts_low = lower["timestamp_ns"].to_numpy()
    ts_high = high["timestamp_ns"].to_numpy()

    t_low = lower.select(TRANSLATION).to_numpy()
    t_high = high.select(TRANSLATION).to_numpy()
    per_point_poses = slerp(point_timestamps_ns_npy).as_matrix()

    target_pose = (
        poses.filter(pl.col("timestamp_ns").eq(timestamp_ns))
        .select(QUATERNION_WXYZ)
        .to_numpy()
    )
    target_t = (
        poses.filter(pl.col("timestamp_ns").eq(timestamp_ns))
        .select(TRANSLATION)
        .to_numpy()
    )
    target_rot = Rotation.from_quat(target_pose).as_matrix()

    city_se3_roll = np.eye(4)
    city_se3_roll[:3, :3] = target_rot
    city_se3_roll[:3, 3] = target_t

    city_se3_laser = np.eye(4)[None].repeat(len(per_point_poses), axis=0)
    city_se3_laser[:, :3, :3] = per_point_poses

    alpha = ((point_timestamps_ns_npy - ts_low) / (ts_high - ts_low))[:, None]
    per_point_t = t_low * alpha + (1 - alpha) * t_high
    city_se3_laser[:, :3, 3] = per_point_t

    # Compute fast inverse.
    rot_inv = city_se3_laser[:, :3, :3].transpose(0, 2, 1)
    t = city_se3_laser[:, :3, 3]
    t_inv = np.einsum("bij,bj->bi", rot_inv, -t)

    laser_se3_city = np.zeros_like(city_se3_laser)
    laser_se3_city[:, :3, :3] = rot_inv
    laser_se3_city[:, :3, 3] = t_inv
    laser_se3_roll = np.einsum("bij,jk->bik", laser_se3_city, city_se3_roll)

    xyz_hom = np.ones((len(xyz), 4))
    xyz_hom[:, :3] = xyz
    xyz_hom = np.einsum("bi,bji->bj", xyz_hom, laser_se3_roll)
    xyz = xyz_hom[:, :3]
    x, y, z = xyz.transpose(1, 0)
    return sweep.with_columns(x_p=pl.lit(x), y_p=pl.lit(y), z_p=pl.lit(z))
