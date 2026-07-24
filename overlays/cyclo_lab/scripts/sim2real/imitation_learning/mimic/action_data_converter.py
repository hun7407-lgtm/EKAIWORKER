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

"""Script to convert recorded demonstration actions between IK and joint space."""

import argparse
import multiprocessing
import os
from copy import deepcopy

import torch
from tqdm import tqdm

from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

def convert_joint_to_ik_omy(ep_data: EpisodeData) -> EpisodeData:
    """Convert joint actions to IK (EEF state + gripper) for OMY robot."""
    try:
        eef_pose = ep_data.data["obs"]["eef_pose"]
        joint_actions = ep_data.data["actions"]

        gripper_action = joint_actions[:, -1:]
        new_actions = torch.cat([eef_pose, gripper_action], dim=1)

        ep_data.data["actions"] = new_actions
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert joint to IK for OMY: {str(e)}")

def _ffw_sg2_joint_action_indices(num_actions: int) -> tuple[int, int, slice, slice]:
    """Map FFW_SG2 joint-space action layout to gripper/head/lift indices.

    Standard layout (19): [arm_l(7), gripper_l(1), arm_r(7), gripper_r(1), lift(1), head(2)]
    Legacy layout (22): extra gripper_l mimic joints at 8-10; only joint1 is used for IK.
    """
    if num_actions == 19:
        # Record/action-manager order: [..., lift_joint(16), head_joint1(17), head_joint2(18)]
        return 7, 15, slice(17, 19), slice(16, 17)
    if num_actions >= 22:
        return 7, 18, slice(20, 22), slice(19, 20)
    raise ValueError(f"FFW_SG2 joint actions expected 19 or 22 dims, got {num_actions}")


# Drivable-base (mobile) tasks record obs/base_velocity = [linear_x, linear_y, angular_z].
# To keep the whole datagen pipeline at 22 dims on mobile tasks (matching real ffw_sg2_rev1),
# the base velocity is appended as the last 3 action channels here. Non-mobile (teleport) tasks
# have no base_velocity obs and stay 19-dim.
BASE_VELOCITY_OBS_KEY = "base_velocity"


def _base_velocity_or_none(ep_data: EpisodeData) -> torch.Tensor | None:
    obs = ep_data.data.get("obs")
    if isinstance(obs, dict) and BASE_VELOCITY_OBS_KEY in obs:
        return obs[BASE_VELOCITY_OBS_KEY]
    return None


def _append_base_velocity(new_actions: torch.Tensor, ep_data: EpisodeData) -> torch.Tensor:
    """Append recorded base velocity to the action so mobile actions are 22-dim; no-op otherwise."""
    base_velocity = _base_velocity_or_none(ep_data)
    if base_velocity is None:
        return new_actions
    if base_velocity.shape[0] != new_actions.shape[0]:
        raise ValueError(
            f"base_velocity length {base_velocity.shape[0]} != action length {new_actions.shape[0]}"
        )
    return torch.cat([new_actions, base_velocity.to(new_actions.dtype)], dim=1)


_FFW_SH5_JOINT_DIM = 57
_FFW_SH5_FINGER_L = slice(7, 27)
_FFW_SH5_FINGER_R = slice(34, 54)
_FFW_SH5_HEAD = slice(54, 56)
_FFW_SH5_LIFT = slice(56, 57)


def convert_joint_to_ik_ffw_sh5(ep_data: EpisodeData) -> EpisodeData:
    """Convert joint actions to IK (EEF + fingers + head + lift) for FFW SH5."""
    try:
        left_eef_pose = ep_data.data["obs"]["left_eef_pose"]
        right_eef_pose = ep_data.data["obs"]["right_eef_pose"]
        joint_actions = ep_data.data["actions"]

        if joint_actions.shape[1] != _FFW_SH5_JOINT_DIM:
            raise ValueError(
                f"FFW_SH5 joint actions expected {_FFW_SH5_JOINT_DIM} dims, "
                f"got {joint_actions.shape[1]}"
            )

        finger_l = joint_actions[:, _FFW_SH5_FINGER_L]
        finger_r = joint_actions[:, _FFW_SH5_FINGER_R]
        head_action = joint_actions[:, _FFW_SH5_HEAD]
        lift_action = joint_actions[:, _FFW_SH5_LIFT]

        # IK layout (57): [left_eef(7), finger_l(20), right_eef(7), finger_r(20), head(2), lift(1)]
        new_actions = torch.cat(
            [
                left_eef_pose,
                finger_l,
                right_eef_pose,
                finger_r,
                head_action,
                lift_action,
            ],
            dim=1,
        )

        ep_data.data["actions"] = new_actions
        return ep_data
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError(f"Failed to convert joint to IK for FFW_SH5: {str(e)}") from e


def convert_joint_to_ik_ffw_sg2(ep_data: EpisodeData) -> EpisodeData:
    """Convert joint actions to IK (EEF state + gripper + lift + head) for FFW SG2 robot."""
    try:
        # FFW SG2 has dual arms, need to handle both left and right EEF states
        left_eef_pose = ep_data.data["obs"]["left_eef_pose"]
        right_eef_pose = ep_data.data["obs"]["right_eef_pose"]
        joint_actions = ep_data.data["actions"]

        grip_l_i, grip_r_i, head_sl, lift_sl = _ffw_sg2_joint_action_indices(
            joint_actions.shape[1]
        )
        gripper_l_action = joint_actions[:, grip_l_i : grip_l_i + 1]
        gripper_r_action = joint_actions[:, grip_r_i : grip_r_i + 1]
        head_action = joint_actions[:, head_sl]
        lift_action = joint_actions[:, lift_sl]

        # IK action order (total 19):
        # [left_eef(7), gripper_l(1), right_eef(7), gripper_r(1), head(2), lift(1)]
        # On mobile tasks, base velocity [linear_x, linear_y, angular_z] is appended -> 22.
        new_actions = torch.cat([
            left_eef_pose,    # 0-6: left EEF (pos + quat)
            gripper_l_action,  # 7: left gripper
            right_eef_pose,   # 8-14: right EEF (pos + quat)
            gripper_r_action,  # 15: right gripper
            head_action,        # 16-17: head joints
            lift_action       # 18: lift joint
        ], dim=1)

        ep_data.data["actions"] = _append_base_velocity(new_actions, ep_data)
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert joint to IK for FFW_SG2: {str(e)}")

def convert_joint_to_ik(ep_data: EpisodeData, robot_type: str) -> EpisodeData:
    """Convert joint actions to IK based on robot type."""
    if robot_type == "OMY":
        return convert_joint_to_ik_omy(ep_data)
    elif robot_type == "FFW_SG2":
        return convert_joint_to_ik_ffw_sg2(ep_data)
    elif robot_type == "FFW_SH5":
        return convert_joint_to_ik_ffw_sh5(ep_data)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

def convert_ik_to_joint(ep_data: EpisodeData) -> EpisodeData:
    """Convert IK actions to joint targets (mobile: append base velocity -> 22 dims)."""
    try:
        joint_targets = ep_data.data["obs"]["joint_pos_target"]
        ep_data.data["actions"] = _append_base_velocity(joint_targets, ep_data)
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert IK to joint: {str(e)}")

def process_dataset(input_file: str, output_file: str, action_type: str, robot_type: str) -> None:
    """Process dataset episodes and convert actions to the desired type."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input dataset file does not exist: {input_file}")

    input_handler = HDF5DatasetFileHandler()
    output_handler = HDF5DatasetFileHandler()

    input_handler.open(input_file)
    output_handler.create(output_file)

    try:
        episode_names = list(input_handler.get_episode_names())
        skipped_episodes = []
        
        for name in tqdm(episode_names, desc="Processing episodes"):
            try:
                ep_data = input_handler.load_episode(name, device="cpu")

                if ep_data.success is not None and not ep_data.success:
                    continue

                processed = deepcopy(ep_data)
                
                # Apply conversion based on action type
                if action_type == "ik":
                    processed = convert_joint_to_ik(processed, robot_type)
                elif action_type == "joint":
                    processed = convert_ik_to_joint(processed)
                
                output_handler.write_episode(processed)
                
            except Exception as e:
                skipped_episodes.append((name, str(e)))
                print(f"\nWarning: Skipping episode '{name}' due to error: {str(e)}")
                continue
        
        if skipped_episodes:
            print(f"\n\nSummary: Skipped {len(skipped_episodes)} episode(s) due to errors:")
            for ep_name, error_msg in skipped_episodes:
                print(f"  - {ep_name}: {error_msg}")

    finally:
        input_handler.close()
        output_handler.flush()
        output_handler.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert recorded demonstration actions between IK and joint space."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./datasets/annotated_dataset.hdf5",
        help="Path to input dataset file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/processed_annotated_dataset.hdf5",
        help="Path to save processed dataset file."
    )
    parser.add_argument(
        "--action_type",
        choices=["ik", "joint"],
        required=True,
        help="Target action representation: 'ik' or 'joint'."
    )
    parser.add_argument(
        "--robot_type",
        choices=["OMY", "FFW_SG2", "FFW_SH5"],
        required=True,
        help="Robot type: 'OMY', 'FFW_SG2', or 'FFW_SH5'.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    process_dataset(args.input_file, args.output_file, args.action_type, args.robot_type)

if __name__ == "__main__":
    main()
