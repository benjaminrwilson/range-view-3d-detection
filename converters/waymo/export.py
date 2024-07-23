import glob
import uuid
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import torch
import waymo_open_dataset
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils, range_image_utils, transform_utils

tf.enable_eager_execution()

SPLIT_MAPPING: Final = {"training": "train", "validation": "val", "testing": "test"}


# Mapping from Argo Camera names to Waymo Camera names
# The indices correspond to Waymo's cameras
CAMERA_NAMES = [
    "unknown",  # 0, 'UNKNOWN',
    "ring_front_center",  # 1, 'FRONT'
    "ring_front_left",  # 2, 'FRONT_LEFT',
    "ring_front_right",  # 3, 'FRONT_RIGHT',
    "ring_side_left",  # 4, 'SIDE_LEFT',
    "ring_side_right",  # 5, 'SIDE_RIGHT'
]

# Mapping from Argo Label types to Waymo Label types
# Argo label types are on the left, Waymo's are on the right
# Argoverse labels: https://github.com/argoai/argoverse-api/blob/master/argoverse/data_loading/object_classes.py#L6
# The indices correspond to Waymo's label types
LABEL_TYPES = [
    "UNKNOWN",  # 0, TYPE_UNKNOWN
    "VEHICLE",  # 1, TYPE_VEHICLE
    "PEDESTRIAN",  # 2, TYPE_PEDESTRIAN
    "SIGN",  # 3, TYPE_SIGN
    "CYCLIST",  # 4, TYPE_CYCLIST
]


def convert_range_image_to_cartesian(
    frame, range_images, range_image_top_pose, ri_index=0, keep_polar_features=False
):
    """Convert range images from polar coordinates to Cartesian coordinates.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
      dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
        will be 3 if keep_polar_features is False (x, y, z) and 6 if
        keep_polar_features is True (range, intensity, elongation, x, y, z).
    """
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4])
    )

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims,
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    for c in frame.context.laser_calibrations:
        if c.name != dataset_pb2.LaserName.TOP:
            continue

        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims
        )
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )

        range_image_mask = range_image_tensor[..., 0] > 0
        nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ

        range_image_mask = tf.cast((range_image_mask & nlz_mask), tf.float32)

        # print((~nlz_mask).numpy().sum())
        # breakpoint()

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        if keep_polar_features:
            # If we want to keep the polar coordinate features of range, intensity,
            # and elongation, concatenate them to be the initial dimensions of the
            # returned Cartesian range image.
            range_image_cartesian = tf.concat(
                [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1
            )

        cartesian_range_images[c.name] = (
            range_image_cartesian * range_image_mask[:, :, None]
        )

    return cartesian_range_images


def get_log_ids_from_files(record_dir: str) -> Dict[str, str]:
    """Get the log IDs of the Waymo records from the directory
       where they are stored

    Args:
        record_dir: The path to the directory where the Waymo data
                    is stored
                    Example: "/path-to-waymo-data"
                    The args.waymo_dir is used here by default
    Returns:
        log_ids: A map of log IDs to tf records from the Waymo dataset
    """
    files = glob.glob(f"{record_dir}/*.tfrecord")
    log_ids = {}
    for i, file in enumerate(files):
        file = file.replace(record_dir, "")
        file = file.replace("/segment-", "")
        file = file.replace(".tfrecord", "")
        file = file.replace("_with_camera_labels", "")
        log_ids[file] = files[i]
    return log_ids


def _helper(item):
    track_id_dict = {}

    annotations_list = []
    poses_list = []

    log_id, tf_fpath, ARGO_WRITE_DIR = item
    dataset = tf.data.TFRecordDataset(tf_fpath, compression_type="")
    dataset = [d for d in dataset]
    for data in tqdm(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Checking if we extracted the correct log ID
        assert log_id == frame.context.name
        # Frame start time, which is the timestamp
        # of the first top lidar spin within this frame, in microseconds
        timestamp_ms = frame.timestamp_micros
        timestamp_ns = int(timestamp_ms * 1000)  # to nanoseconds
        SE3_flattened = np.array(frame.pose.transform)
        city_SE3_egovehicle = SE3_flattened.reshape(4, 4)

        pose = dump_pose(city_SE3_egovehicle, timestamp_ns, log_id, ARGO_WRITE_DIR)
        poses_list.append(pose)

        # frame_utils.convert_frame_to_dict(frame)

        # Reading lidar data and saving it in point cloud format
        # We are only using the first range image (Waymo provides two range images)
        # If you want to use the second one, you can change it in the arguments
        (
            range_images,
            _,
            _,
            range_image_top_pose,
        ) = frame_utils.parse_range_image_and_camera_projection(frame)

        first_return_cartesian_range_images = convert_range_image_to_cartesian(
            frame,
            range_images,
            range_image_top_pose,
            ri_index=0,
            keep_polar_features=True,
        )

        range_image = first_return_cartesian_range_images[1].numpy()
        # points_all_ri = range_image.reshape(-1, 6)
        # points_all_ri = np.concatenate(points_ri[:1], axis=0)
        # laser_number = np.zeros_like(points_all_ri[:, 0:1])

        range_image = (
            torch.as_tensor(range_image)
            .permute(2, 0, 1)
            .contiguous()
            .type(torch.bfloat16)
        )

        perm = [1, 2, 0, 3, 4, 5]
        range_image = range_image[perm].contiguous()
        range_image[0] = range_image[0].tanh()

        # lidar_frame = pd.DataFrame(
        #     points_all_ri,
        #     columns=["range", "intensity", "elongation", "x", "y", "z"],
        # )
        # lidar_frame = lidar_frame[
        #     ["x", "y", "z", "range", "intensity", "elongation"]
        # ]

        # # # https://github.com/open-mmlab/mmdetection3d/issues/299
        # lidar_frame["intensity"] = np.tanh(lidar_frame["intensity"])

        dst = Path(ARGO_WRITE_DIR) / log_id / "sensors" / "lidar" / f"{timestamp_ns}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(range_image, dst)
        # lidar_frame.to_feather(dst, compression="uncompressed")

        annotations = dump_object_labels(
            frame.laser_labels,
            timestamp_ns,
            log_id,
            ARGO_WRITE_DIR,
            track_id_dict,
        )
        annotations_list.append(annotations)
    annotations = pd.concat(annotations_list).reset_index(drop=True)
    dst = Path(ARGO_WRITE_DIR) / log_id / "annotations.feather"
    annotations.to_feather(dst, compression="uncompressed")

    dst = Path(ARGO_WRITE_DIR) / log_id / "city_SE3_egovehicle.feather"
    poses = pd.concat(poses_list).reset_index(drop=True)
    poses.to_feather(dst, compression="uncompressed")


# def form_calibration_json(
#     calib_data: google.protobuf.pyext._message.RepeatedCompositeContainer,
# ) -> Dict[str, Any]:
#     """Create a JSON file per log containing calibration information, in the Argoverse format.

#     Argoverse expects to receive "egovehicle_T_camera", i.e. from camera -> egovehicle, with
#         rotation parameterized as a quaternion.
#     Waymo provides the same SE(3) transformation, but with rotation parameterized as a 3x3 matrix.
#     """
#     calib_dict = {"camera_data_": []}
#     for camera_calib in calib_data:
#         cam_name = CAMERA_NAMES[camera_calib.name]
#         # They provide "Camera frame to vehicle frame."
#         # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
#         egovehicle_SE3_waymocam = np.array(camera_calib.extrinsic.transform).reshape(
#             4, 4
#         )
#         standardcam_R_waymocam = transform_utils.rotY(-90) @ transform_utils.rotX(90)
#         standardcam_SE3_waymocam = SE3(
#             rotation=standardcam_R_waymocam, translation=np.zeros(3)
#         )
#         egovehicle_SE3_waymocam = SE3(
#             rotation=egovehicle_SE3_waymocam[:3, :3],
#             translation=egovehicle_SE3_waymocam[:3, 3],
#         )
#         standardcam_SE3_egovehicle = standardcam_SE3_waymocam.compose(
#             egovehicle_SE3_waymocam.inverse()
#         )
#         egovehicle_SE3_standardcam = standardcam_SE3_egovehicle.inverse()
#         egovehicle_q_camera = transform_utils.rotmat2quat(
#             egovehicle_SE3_standardcam.rotation
#         )
#         x, y, z = egovehicle_SE3_standardcam.translation
#         qw, qx, qy, qz = egovehicle_q_camera
#         f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = camera_calib.intrinsic
#         cam_dict = {
#             "key": "image_raw_" + cam_name,
#             "value": {
#                 "focal_length_x_px_": f_u,
#                 "focal_length_y_px_": f_v,
#                 "focal_center_x_px_": c_u,
#                 "focal_center_y_px_": c_v,
#                 "skew_": 0,
#                 "distortion_coefficients_": [0, 0, 0],
#                 "vehicle_SE3_camera_": {
#                     "rotation": {"coefficients": [qw, qx, qy, qz]},
#                     "translation": [x, y, z],
#                 },
#             },
#         }
#         calib_dict["camera_data_"] += [cam_dict]
#     return calib_dict


def dump_pose(
    city_SE3_egovehicle: np.ndarray, timestamp: int, log_id: str, parent_path: str
) -> None:
    """Saves the pose of the egovehicle in the city coordinate frame at a particular timestamp.

    The SE(3) transformation is stored as a quaternion and length-3 translation vector.

    Args:
        city_SE3_egovehicle: A (4,4) numpy array representing the
                            SE3 transformation from city to egovehicle frame
        timestamp: Timestamp in nanoseconds when the lidar reading occurred
        log_id: Log ID that the reading belongs to
        parent_path: The directory that the converted data is written to
    """
    x, y, z = city_SE3_egovehicle[:3, 3]
    R = city_SE3_egovehicle[:3, :3]
    assert np.allclose(city_SE3_egovehicle[3], np.array([0, 0, 0, 1]))
    q = rotmat2quat(R)
    qw, qx, qy, qz = q

    frame = pd.DataFrame.from_dict(
        {
            "timestamp_ns": timestamp,
            "qw": qw,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "tx_m": x,
            "ty_m": y,
            "tz_m": z,
        },
        orient="index",
    )
    return frame.T


def dump_object_labels(
    labels: List[waymo_open_dataset.label_pb2.Label],
    timestamp: int,
    log_id: str,
    parent_path: str,
    track_id_dict: Dict[str, str],
) -> None:
    """Saves object labels from Waymo dataset as json files.

    Args:
        labels: A list of Waymo labels
        timestamp: Timestamp in nanoseconds when the lidar reading occurred
        log_id: Log ID that the reading belongs to
        parent_path: The directory that the converted data is written to
        track_id_dict: Dictionary to store object ID to track ID mappings
    """
    argoverse_labels = []
    for label in labels:
        # We don't want signs, as that is not a category in Argoverse
        if label.type != LABEL_TYPES.index("SIGN") and label.type != LABEL_TYPES.index(
            "UNKNOWN"
        ):
            annotation = build_argo_label(label, timestamp, track_id_dict)
            argoverse_labels.append(annotation)

    columns = [
        "timestamp_ns",
        "track_uuid",
        "category",
        "length_m",
        "width_m",
        "height_m",
        "qw",
        "qx",
        "qy",
        "qz",
        "tx_m",
        "ty_m",
        "tz_m",
        "num_interior_pts",
        "difficulty_level",
    ]
    if len(argoverse_labels) == 0:
        return pd.DataFrame(columns=columns)
    annotations = pd.concat(argoverse_labels).reset_index()
    annotations = annotations[columns]
    return annotations


def build_argo_label(
    label: waymo_open_dataset.label_pb2.Label,
    timestamp: int,
    track_id_dict: Dict[str, str],
) -> Dict[str, Any]:
    """Builds a dictionary that represents an object detection in Argoverse format from a Waymo label

    Args:
        labels: A Waymo label
        timestamp: Timestamp in nanoseconds when the lidar reading occurred
        track_id_dict: Dictionary to store object ID to track ID mappings

    Returns:
        label_dict: A dictionary representing the object label in Argoverse format
    """
    label_dict = {}
    label_dict["tx_m"] = label.box.center_x
    label_dict["ty_m"] = label.box.center_y
    label_dict["tz_m"] = label.box.center_z
    label_dict["length_m"] = label.box.length
    label_dict["width_m"] = label.box.width
    label_dict["height_m"] = label.box.height
    qx, qy, qz, qw = yaw_to_quaternion3d(label.box.heading)
    label_dict["qx"] = qx
    label_dict["qy"] = qy
    label_dict["qz"] = qz
    label_dict["qw"] = qw
    label_dict["category"] = LABEL_TYPES[label.type]
    label_dict["timestamp_ns"] = timestamp
    label_dict["num_interior_pts"] = label.num_lidar_points_in_box
    label_dict["difficulty_level"] = label.detection_difficulty_level
    if label.id not in track_id_dict.keys():
        track_id = uuid.uuid4().hex
        track_id_dict[label.id] = track_id
    else:
        track_id = track_id_dict[label.id]
    label_dict["track_uuid"] = track_id
    return pd.DataFrame.from_dict(label_dict, orient="index").T


def yaw_to_quaternion3d(yaw: float) -> Tuple[float, float, float, float]:
    """
    Args:
        yaw: rotation about the z-axis

    Returns:
        qx,qy,qz,qw: quaternion coefficients
    """
    qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
    return qx, qy, qz, qw


def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """ """
    q_scipy = Rotation.from_matrix(R).as_quat()
    x, y, z, w = q_scipy
    q_argo = w, x, y, z
    return q_argo


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


def main() -> None:
    split = "training"
    SPLIT_MAPPING = {"training": "train", "validation": "val", "testing": "test"}
    TFRECORD_DIR = str(Path.home() / "data" / "datasets" / "waymo_original" / split)
    ARGO_WRITE_DIR = str(
        Path.home()
        / "data"
        / "datasets"
        / "waymo-bf16"
        / "sensor"
        / SPLIT_MAPPING[split]
    )
    track_id_dict = {}
    log_ids = get_log_ids_from_files(TFRECORD_DIR)

    annotations_list = []
    poses_list = []

    from tqdm.contrib.concurrent import process_map

    iters = [(a, b, ARGO_WRITE_DIR) for (a, b) in log_ids.items()]
    # for iterable in tqdm(iters):
    #     _helper(iterable)
    process_map(_helper, iters, max_workers=16)


if __name__ == "__main__":
    main()
