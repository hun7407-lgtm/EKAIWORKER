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

"""
Main data generation script.
"""


"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import gymnasium as gym
import inspect
import numpy as np
import random
import torch

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
import os
import pathlib
import sys

_MIMIC_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_MIMIC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_MIMIC_SCRIPT_DIR))

from cyclo_mimic_datagen import enable_cyclo_body_joint_replay, setup_cyclo_async_generation
from isaaclab_mimic.datagen.generation import env_loop, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

import cyclo_lab  # noqa: F401


def _postprocess_base_velocity(output_file_path: str) -> None:
    """Append obs/base_velocity to each generated episode's actions so mobile output is 22-dim.

    The datagen recorder writes the 19-dim joint/IK action; on mobile tasks the base is driven
    off-action and captured as obs/base_velocity. This runs after the sim app closes (file handle
    released) and rewrites the actions dataset to [19-dim action | 3-dim base velocity]. No-op for
    teleport tasks (no base_velocity) or episodes whose actions were already packed.
    """
    import h5py

    if not output_file_path.endswith(".hdf5"):
        output_file_path = output_file_path + ".hdf5"
    if not os.path.exists(output_file_path):
        return

    base_key = "base_velocity"
    packed = 0
    with h5py.File(output_file_path, "r+") as f:
        if "data" not in f:
            return
        for demo_name in list(f["data"].keys()):
            g = f["data"][demo_name]
            if "obs" not in g or base_key not in g["obs"] or "actions" not in g:
                continue
            actions = g["actions"][:]
            base = g["obs"][base_key][:]
            ref_dim = g["obs"]["joint_pos"].shape[1] if "joint_pos" in g["obs"] else actions.shape[1]
            # Skip if already packed (actions already carry the base channels).
            if actions.shape[1] != ref_dim:
                continue
            if base.shape[0] != actions.shape[0]:
                continue
            import numpy as np

            new_actions = np.concatenate([actions, base.astype(actions.dtype)], axis=1)
            del g["actions"]
            g.create_dataset("actions", data=new_actions)
            packed += 1
    if packed:
        print(f"[base_velocity] Repacked {packed} episode(s) to 22-dim actions in {output_file_path}")


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    output_file_path = os.path.join(output_dir, output_file_name + ".hdf5")
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )
    env_cfg.init_action_cfg("mimic_ik")
    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    enable_cyclo_body_joint_replay()

    # Check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting
    env.reset()

    motion_planners = None
    if args_cli.use_skillgen:
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # Create one motion planner per environment
        motion_planners = {}
        for env_id in range(num_envs):
            print(f"Initializing motion planner for environment {env_id}")
            # Create a config instance from the task name
            planner_config = CuroboPlannerCfg.from_task_name(env_name)

            # Ensure visualization is only enabled for the first environment
            # If not, sphere and plan visualization will be too slow in isaac lab
            # It is efficient to visualize the spheres and plan for the first environment in rerun
            if env_id != 0:
                planner_config.visualize_spheres = False
                planner_config.visualize_plan = False

            motion_planners[env_id] = CuroboPlanner(
                env=env,
                robot=env.scene["robot"],
                config=planner_config,  # Pass the config object
                env_id=env_id,  # Pass environment ID
            )

        env.cfg.datagen_config.use_skillgen = True

    # Setup and run async data generation
    async_components = setup_cyclo_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,  # Pass the motion planners dictionary
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        # Cancel all async tasks when env_loop finishes
        data_gen_tasks.cancel()
        try:
            # Wait for tasks to be cancelled
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")
        # Cleanup of motion planners and their visualizers
        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            motion_planners.clear()

    # Re-attach base velocity so generated mobile actions are 22-dim. Run BEFORE
    # simulation_app.close(): Isaac's app teardown can hard-exit the process, so work placed after
    # it may never run. Wrapped defensively so a repack issue never discards a long datagen run;
    # if it is skipped, rerun action_data_converter (joint step) which also packs base velocity.
    try:
        _postprocess_base_velocity(output_file_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[base_velocity] WARNING: post-process skipped ({exc}); actions left at env dim.")

    return output_file_path


if __name__ == "__main__":
    output_file_path = None
    try:
        output_file_path = main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
