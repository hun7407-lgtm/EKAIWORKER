# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import h5py
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime

from lerobot.datasets.lerobot_dataset import LeRobotDataset

_FFW_SH5_JOINT_NAMES = (
    [f"arm_l_joint{i}" for i in range(1, 8)]
    + [f"finger_l_joint{i}" for i in range(1, 21)]
    + [f"arm_r_joint{i}" for i in range(1, 8)]
    + [f"finger_r_joint{i}" for i in range(1, 21)]
    + ["head_joint1", "head_joint2", "lift_joint"]
)

ROBOT_CONFIGS = {
    "OMY": {
        "expected_dim": 7,
        "joint_names": [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "rh_r1_joint",
        ],
        "cameras": {
            "cam_wrist": {"height": 480, "width": 848},
            "cam_top": {"height": 480, "width": 848},
        }
    },
    "FFW_SG2": {
        "expected_dim": 19,
        "joint_names": [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
            "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
            "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
            "head_joint1", "head_joint2", "lift_joint",
        ],
        "cameras": {
            "cam_head": {"height": 376, "width": 672},
        }
    },
    "FFW_SH5": {
        "expected_dim": 57,
        "joint_names": list(_FFW_SH5_JOINT_NAMES),
        "cameras": {
            "cam_head": {"height": 376, "width": 672},
        }
    },
}

# Map the HDF5 obs camera key (the ObsTerm name in the env cfg) to the LeRobot key used by
# the real robot datasets (lift_box_merged_lerobot_*). Real stores each stream under
# observation.images.rgb.<name> as channels-first uint8 [3, H, W]. Only keys present in the
# recorded HDF5 are exported, so single-camera tasks still convert without error.
#   cam_head -> cam_left_head: the sim head camera is the ZED left eye.
CAMERA_NAME_MAP = {
    "cam_head": "cam_left_head",
    "cam_right_head": "cam_right_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
    # OMY streams keep their own names (no real stereo counterpart).
    "cam_wrist": "cam_wrist",
    "cam_top": "cam_top",
}


def detect_cameras(demo_group: h5py.Group) -> list[dict]:
    """Find recorded camera streams in a demo and read their real resolution from the data.

    Resolution is taken from the actual array (not hard-coded) so it always matches what the
    env rendered. Returns [{hdf5_key, lerobot_name, height, width}, ...].
    """
    cameras = []
    obs = demo_group["obs"]
    for hdf5_key, lerobot_name in CAMERA_NAME_MAP.items():
        if hdf5_key not in obs:
            continue
        # mdp.image records HWC uint8: (frames, H, W, 3).
        _, height, width, _ = obs[hdf5_key].shape
        cameras.append(
            {"hdf5_key": hdf5_key, "lerobot_name": lerobot_name, "height": int(height), "width": int(width)}
        )
    return cameras


# Drivable-base tasks record obs/base_velocity = [linear_x, linear_y, angular_z] (base frame),
# appended to the 19 joints so state/action reach 22 dims (real ffw_sg2_rev1 parity).
BASE_VELOCITY_KEY = "base_velocity"
BASE_VELOCITY_NAMES = ["linear_x", "linear_y", "angular_z"]


def detect_base_velocity(demo_group: h5py.Group) -> bool:
    """True when the recording carries a base-velocity stream (mobile task)."""
    return BASE_VELOCITY_KEY in demo_group["obs"]


def get_env_features(fps: int, robot_type: str, cameras: list[dict], has_base_velocity: bool = False):
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    config = ROBOT_CONFIGS[robot_type]

    # 19 joints, or 22 when the base velocity is appended (mobile task).
    joint_names = list(config["joint_names"])
    if has_base_velocity:
        joint_names = joint_names + BASE_VELOCITY_NAMES
    state_dim = len(joint_names)

    # Build action and observation.state features
    features = {
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        }
    }

    # Add camera features. Channels-first [3, H, W] under the observation.images.rgb.* key to
    # match the real ffw_sg2_rev1 datasets.
    for cam in cameras:
        height, width = cam["height"], cam["width"]
        features[f"observation.images.rgb.{cam['lerobot_name']}"] = {
            "dtype": "video",
            "shape": [3, height, width],
            "names": ["channels", "height", "width"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return features

def process_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str, frame_skip: int, robot_type: str, cameras: list[dict], has_base_velocity: bool = False) -> bool:
    """
    Process a single demonstration group from the HDF5 dataset
    and add it into the LeRobot dataset.
    """
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    config = ROBOT_CONFIGS[robot_type]

    try:
        # Load action and joint position data
        actions = np.array(demo_group['actions'], dtype=np.float32)
        joint_pos = np.array(demo_group['obs/joint_pos'], dtype=np.float32)

        # Base velocity [linear_x, linear_y, angular_z] for 22-dim mobile recordings.
        base_velocity = None
        if has_base_velocity:
            base_velocity = np.array(demo_group[f"obs/{BASE_VELOCITY_KEY}"], dtype=np.float32)

        # Load camera images (recorded HWC uint8).
        camera_data = {}
        for cam in cameras:
            camera_data[cam["hdf5_key"]] = np.array(demo_group[f"obs/{cam['hdf5_key']}"], dtype=np.uint8)

    except KeyError as e:
        print(f"Demo {demo_name} is not valid (missing key: {e}), skipping...")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has insufficient frames ({actions.shape[0]}), skipping...")
        return False

    # Ensure actions and joint positions are 2D arrays
    if actions.ndim == 1:
        actions = actions.reshape(-1, config["expected_dim"])
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, config["expected_dim"])
    
    # Mobile datagen may hand us pre-packed actions ([joints(19) | base velocity(3)] = 22).
    # Accept either the plain joint dim (base appended below) or the pre-packed dim.
    packed_dim = config["expected_dim"] + len(BASE_VELOCITY_NAMES)
    action_prepacked = base_velocity is not None and actions.shape[1] == packed_dim
    valid_action_dims = {config["expected_dim"]}
    if base_velocity is not None:
        valid_action_dims.add(packed_dim)
    if actions.shape[1] not in valid_action_dims:
        print(
            f"Demo {demo_name} action dim {actions.shape[1]} != "
            f"expected {sorted(valid_action_dims)} for {robot_type}, skipping..."
        )
        return False
    if joint_pos.shape[1] != config["expected_dim"]:
        print(
            f"Demo {demo_name} joint_pos dim {joint_pos.shape[1]} != "
            f"expected {config['expected_dim']} for {robot_type}, skipping..."
        )
        return False

    total_state_frames = actions.shape[0]

    # Process each frame
    for frame_index in tqdm(range(total_state_frames), desc=f"Processing demo {demo_name}"):
        if frame_index < frame_skip:
            continue
        
        action_row = actions[frame_index]
        state_row = joint_pos[frame_index]
        if base_velocity is not None:
            # state is built from the 19-dim joint_pos, so always append the measured base twist.
            # action: append only when not already packed upstream (mobile datagen packs it in).
            base_row = base_velocity[frame_index]
            state_row = np.concatenate([state_row, base_row])
            if not action_prepacked:
                action_row = np.concatenate([action_row, base_row])

        # Build frame dictionary
        frame = {
            "action": action_row,
            "observation.state": state_row,
        }

        # Add camera images, HWC -> CHW to match the real ffw_sg2_rev1 layout.
        for cam in cameras:
            img_hwc = camera_data[cam["hdf5_key"]][frame_index]
            frame[f"observation.images.rgb.{cam['lerobot_name']}"] = np.transpose(img_hwc, (2, 0, 1))

        dataset.add_frame(frame=frame, task=task)

    return True

def convert_isaaclab_to_lerobot(
    task: str, repo_id: str, robot_type: str, dataset_file: str,
    fps: int, push_to_hub: bool = False, frame_skip: int = 3, root: str = "./datasets/lerobot/sim2real_data"
):
    """
    Convert an IsaacLab HDF5 dataset into LeRobot dataset format.
    """
    hdf5_files = [dataset_file]
    now_episode_index = 0

    # Detect which streams this recording actually contains, from its first demo.
    # The feature schema must be fixed before the dataset is created, so this happens first.
    with h5py.File(dataset_file, "r") as f:
        first_demo = f["data"][list(f["data"].keys())[0]]
        cameras = detect_cameras(first_demo)
        has_base_velocity = detect_base_velocity(first_demo)
    if cameras:
        print("Detected cameras: " + ", ".join(
            f"{c['hdf5_key']} -> observation.images.rgb.{c['lerobot_name']} "
            f"[{c['height']}x{c['width']}]" for c in cameras))
    else:
        print("WARNING: no known camera streams found in HDF5 (state/action only).")
    print(f"Base velocity: {'present -> 22-dim state/action' if has_base_velocity else 'absent -> 19-dim'}")

    # Create a new LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=get_env_features(fps, robot_type, cameras, has_base_velocity),
        root=root,
    )

    # Process each HDF5 dataset file
    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]

                # Skip unsuccessful demonstrations
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} not successful, skipping...")
                    continue

                valid = process_data(dataset, task, demo_group, demo_name, frame_skip, robot_type, cameras, has_base_velocity)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saved episode {now_episode_index} successfully")

    # Optionally push to HuggingFace Hub
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot format")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., OMY_Pickup)")
    parser.add_argument(
        "--robot_type",
        type=str,
        default="OMY",
        choices=["OMY", "FFW_SG2", "FFW_SH5"],
        help="Robot type (OMY, FFW_SG2, or FFW_SH5)",
    )
    parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Path to dataset HDF5 file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for dataset (default: 10)")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to HuggingFace Hub")
    parser.add_argument("--frame_skip", type=int, default=2, help="Frame skip rate (default: 2)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_repo_id = f"./datasets/lerobot/{timestamp}"
    parser.add_argument("--repo_id", type=str, default=default_repo_id, help=f"Repo ID (default: {default_repo_id})")

    args = parser.parse_args()

    convert_isaaclab_to_lerobot(
        task=args.task,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        dataset_file=args.dataset_file,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        frame_skip=args.frame_skip,
        root=default_repo_id,
    )
