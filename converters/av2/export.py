import shutil
from pathlib import Path
from typing import Any, Dict, Final, Tuple

import cv2
import matplotlib
import numpy as np
import polars as pl
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.utils.io import read_city_SE3_ego
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
from utils import build_range_view, correct_laser_numbers, unmotion_compensate

turbo: Final = matplotlib.colormaps["turbo"]

TRANSLATION: Final = ("tx_m", "ty_m", "tz_m")
DIMS: Final = ("length_m", "width_m", "height_m")
QUATERNION_WXYZ: Final = ("qx", "qy", "qz", "qw")

FEATURE_COLUMN_NAMES: Tuple[str, ...] = (
    "x",
    "y",
    "z",
    "intensity",
    "laser_number",
    "is_within_roi",
)


def main(
    range_view_config: Dict[str, Any],
    enable_write: bool = False,
) -> None:
    build_uniform_inclination = range_view_config["build_uniform_inclination"]
    sensor_name = range_view_config["sensor_name"]
    height = range_view_config["height"]
    width = range_view_config["width"]
    export_range_view = range_view_config["export_range_view"]
    enable_motion_uncompensation = range_view_config["enabled_motion_uncompensation"]

    root_dir = Path.home() / "data" / "datasets" / "av2_original" / "sensor"
    dst_dir = Path.home() / "data" / "datasets" / "av2-64" / "sensor"
    splits = ["val"]

    for split in splits:
        split_dir = root_dir / split
        for i, log_dir in enumerate(tqdm(sorted(split_dir.glob("*")))):
            avm = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=True)

            log_id = log_dir.stem
            dst_log_dir = dst_dir / split / log_id
            dst_log_dir.mkdir(exist_ok=True, parents=True)

            annotations_path = log_dir / "annotations.feather"
            annotations = (
                pl.scan_ipc(annotations_path)
                .filter(pl.col("num_interior_pts") > 0)
                .collect()
            )

            poses = pl.read_ipc(log_dir / "city_SE3_egovehicle.feather")
            rots = Rotation.from_quat(poses.select(QUATERNION_WXYZ).to_numpy())

            times = poses["timestamp_ns"].to_numpy()
            slerp = Slerp(times, rots)

            extrinsics = pl.read_ipc(
                log_dir / "calibration" / "egovehicle_SE3_sensor.feather",
                memory_map=False,
            )

            city_SE3_egovehicle = read_city_SE3_ego(log_dir)

            sweep_paths = sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
            for sweep_path in sweep_paths:
                timestamp_ns = int(sweep_path.stem)
                lidar_lazy = (
                    pl.scan_ipc(sweep_path)
                    .select(("x", "y", "z", "intensity", "laser_number", "offset_ns"))
                    .cast({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})
                )

                if sensor_name == "up_lidar":
                    lidar_lazy = lidar_lazy.filter(pl.col("laser_number") <= 31)
                elif sensor_name == "down_lidar":
                    lidar_lazy = lidar_lazy.filter(
                        pl.col("laser_number") >= 32
                    ).with_columns(laser_number=pl.col("laser_number") - 32)
                lidar = lidar_lazy.collect()

                city_xyz = city_SE3_egovehicle[timestamp_ns].transform_from(
                    lidar.select(("x", "y", "z")).to_numpy()
                )
                mask = avm.get_raster_layer_points_boolean(
                    city_xyz, RasterLayerType.ROI
                )
                lidar = lidar.with_columns(is_within_roi=pl.lit(mask))

                if enable_motion_uncompensation:
                    lidar = unmotion_compensate(
                        lidar,
                        poses,
                        timestamp_ns,
                        slerp,
                    )
                else:
                    lidar = lidar.with_columns(
                        x_p=pl.col("x"), y_p=pl.col("y"), z_p=pl.col("z")
                    )

                features = lidar.select(FEATURE_COLUMN_NAMES)
                laser_number = lidar["laser_number"].to_numpy().copy()

                laser_number = correct_laser_numbers(
                    laser_number,
                    log_id,
                    height=height,
                )

                if sensor_name == "down_lidar":
                    laser_number = 32 - laser_number - 1

                lidar = lidar.with_columns(pl.Series("laser_number", laser_number))
                if export_range_view:
                    range_view = build_range_view(
                        lidar,
                        extrinsics=extrinsics,
                        features=features.to_numpy(),
                        sensor_name=sensor_name if sensor_name != "all" else "up_lidar",
                        height=height,
                        width=width,
                        build_uniform_inclination=build_uniform_inclination,
                    )

                    if enable_write:
                        range_view_dst = (
                            Path(dst_log_dir)
                            / "sensors"
                            / "range_view"
                            / f"{timestamp_ns}.feather"
                        )
                        range_view_dst.parent.mkdir(parents=True, exist_ok=True)
                        range_view.write_ipc(range_view_dst)

                    dists = range_view["range"].to_numpy().reshape(height, width)
                    dists = turbo(dists / dists.max()) * 255.0

                    fname = f"{sensor_name}.png"
                    cv2.imwrite(fname, dists[:, :, :3])
                    breakpoint()

                if enable_write:
                    lidar_dst = (
                        dst_log_dir / "sensors" / "lidar" / f"{timestamp_ns}.feather"
                    )
                    lidar_dst.parent.mkdir(parents=True, exist_ok=True)
                    lidar.write_ipc(lidar_dst)

            if enable_write:
                annotations_dst = dst_log_dir / "annotations.feather"
                annotations.write_ipc(annotations_dst)

                poses_dst = dst_log_dir / "city_SE3_egovehicle.feather"
                poses.write_ipc(poses_dst)

                shutil.copytree(
                    log_dir / "map", dst_log_dir / "map", dirs_exist_ok=True
                )


if __name__ == "__main__":
    range_view_config = {
        "build_uniform_inclination": False,
        "sensor_name": "all",
        "height": 64,
        "width": 1800,
        "export_range_view": True,
        "enabled_motion_uncompensation": False,
    }
    main(range_view_config=range_view_config, enable_write=False)
