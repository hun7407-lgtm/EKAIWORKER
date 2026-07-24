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

from __future__ import annotations

import time

import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.datasets import EpisodeData

from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place.pick_place_mimic_env import (
    FFWSG2PickPlaceMimicEnv,
)

from .ltable_kinematic_l_motion import LTableKinematicLMotion


def _state_to_device(state: dict, device: torch.device) -> dict:
    output: dict = {}
    for asset_type, assets in state.items():
        output[asset_type] = {}
        for asset_name, fields in assets.items():
            output[asset_type][asset_name] = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in fields.items()
            }
    return output


def apply_recorded_step_state(env, episode: EpisodeData, state_index: int, env_ids: torch.Tensor) -> bool:
    """Apply recorded post-step scene state (same as demo playback)."""
    step_state = episode.get_state(state_index)
    if step_state is None:
        return False
    step_state = _state_to_device(step_state, env.device)
    env.scene.reset_to(step_state, env_ids=env_ids, is_relative=True)
    env.sim.forward()
    return True


def _expand_state_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast single-env recorded tensors across vectorized envs."""
    if tensor.shape[0] == batch_size:
        return tensor.clone()
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1).clone()
    raise ValueError(f"Cannot broadcast state tensor shape {tensor.shape} to {batch_size} envs")


def apply_recorded_robot_root_state(env, episode: EpisodeData, state_index: int, env_ids: torch.Tensor) -> bool:
    """Replay recorded robot base pose for L-motion (box handled via rigid carry).

    The recorded world-frame root velocity is replayed too, so the ``base_velocity`` observation
    reads the real driving velocity on the mobile task (else it would be zero during the teleport
    and the 22-dim export would carry a flat base channel). Non-mobile states without a recorded
    velocity fall back to zero.
    """
    step_state = episode.get_state(state_index)
    if step_state is None:
        return False
    step_state = _state_to_device(step_state, env.device)
    n = len(env_ids)

    robot_state = step_state["articulation"]["robot"]
    robot = env.scene["robot"]
    robot_pose = _expand_state_batch(robot_state["root_pose"], n)
    robot_pose[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)

    if "root_velocity" in robot_state:
        robot_vel = _expand_state_batch(robot_state["root_velocity"], n)
    else:
        robot_vel = torch.zeros(n, 6, device=env.device)
    robot.write_root_velocity_to_sim(robot_vel, env_ids=env_ids)

    env.sim.forward()
    return True


class FFWSG2PickPlaceLTableMimicEnv(FFWSG2PickPlaceMimicEnv):
    """Mimic env for L-table: dual-arm IK; L-motion via recorded state replay in datagen."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._l_motion_ctrl = LTableKinematicLMotion(self)
        self._mimic_recorded_state: tuple[EpisodeData, int] | None = None
        self._mimic_carry_latch: bool = False
        # Per-env variants used by the multi-env (num_envs > 1) datagen path.
        self._mimic_recorded_states: dict[int, tuple[EpisodeData, int]] = {}
        self._mimic_carry_latch_envs: set[int] = set()
        self._kinematic_step_last_time: float | None = None
        self._last_step_was_kinematic: bool = False

        # Physics base driving (Plan B): on mobile tasks, replay the recorded base velocity as a
        # swerve cmd_vel so the base physically drives during the move phase (wheels spin, no root
        # teleport, no teleport/physics jitter) instead of the kinematic pose replay above. Gated
        # on the mobile cfg flag; the stock (welded-base) mimic keeps the teleport path untouched.
        self._base_physics_drive: bool = bool(getattr(cfg, "teleop_base_drive", False))
        self._swerve_ready: bool = False
        if self._base_physics_drive:
            self._init_base_physics_drive()

    def reset(self, *args, **kwargs):
        self._mimic_body_joint_cmd = None
        self._mimic_recorded_state = None
        self._mimic_carry_latch = False
        self._mimic_recorded_states = {}
        self._mimic_carry_latch_envs = set()
        self._kinematic_step_last_time = None
        self._last_step_was_kinematic = False
        self._l_motion_ctrl.reset()
        ret = super().reset(*args, **kwargs)
        self.sim.forward()
        return ret

    # ------------------------------------------------------------------
    # Physics base driving (Plan B mobile datagen)
    # ------------------------------------------------------------------
    def _init_base_physics_drive(self) -> None:
        """Resolve swerve joint ids + module geometry so the base can be driven from cmd_vel.

        Uses the same SG2 swerve constants and geometry as the VR-recording swerve stack, batched
        across all envs so multi-env datagen drives every env's base at once.
        """
        import math

        from cyclo_lab.assets.robots.FFW_SG2 import (
            SG2_SWERVE_MODULE_ANGLE_OFFSETS,
            SG2_SWERVE_MODULE_X_OFFSETS,
            SG2_SWERVE_MODULE_Y_OFFSETS,
            SG2_SWERVE_STEERING_JOINTS,
            SG2_SWERVE_WHEEL_JOINTS,
            SG2_SWERVE_WHEEL_RADIUS,
        )

        robot = self.scene["robot"]
        try:
            steer_ids, steer_names = robot.find_joints(list(SG2_SWERVE_STEERING_JOINTS))
            drive_ids, drive_names = robot.find_joints(list(SG2_SWERVE_WHEEL_JOINTS))
        except Exception as exc:  # noqa: BLE001
            print(f"[BasePhysicsDrive] swerve joints not found ({exc}); falling back to teleport.")
            return

        # find_joints returns articulation order, not the requested order -- map geometry by the
        # module key (name prefix) so each wheel gets its own offset, and align drive to steer.
        def _key(name: str) -> str:
            return name.split("_")[0]

        geom = {
            _key(n): (x, y, a)
            for n, x, y, a in zip(
                SG2_SWERVE_STEERING_JOINTS,
                SG2_SWERVE_MODULE_X_OFFSETS,
                SG2_SWERVE_MODULE_Y_OFFSETS,
                SG2_SWERVE_MODULE_ANGLE_OFFSETS,
            )
        }
        steer_keys = [_key(n) for n in steer_names]
        drive_by_key = {_key(n): i for n, i in zip(drive_names, drive_ids)}
        try:
            drive_ids_ordered = [drive_by_key[k] for k in steer_keys]
        except KeyError as exc:  # noqa: BLE001
            print(f"[BasePhysicsDrive] steer/drive module mismatch ({exc}); falling back to teleport.")
            return

        device = self.device
        self._swerve_steer_ids = steer_ids
        self._swerve_drive_ids = drive_ids_ordered
        self._swerve_module_x = torch.tensor([geom[k][0] for k in steer_keys], device=device)
        self._swerve_module_y = torch.tensor([geom[k][1] for k in steer_keys], device=device)
        self._swerve_angle_offset = torch.tensor([geom[k][2] for k in steer_keys], device=device)
        self._swerve_wheel_radius = float(SG2_SWERVE_WHEEL_RADIUS)
        self._swerve_ready = True
        print("[BasePhysicsDrive] enabled: base is driven physically from recorded cmd_vel.")

    def _swerve_targets(self, twist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Body-frame twist ``(N,3)`` -> (steer angles ``(N,M)``, wheel speeds ``(N,M)``).

        Takes the shorter steer path (flip wheel direction past 90 deg) so reversing does not
        swing each module a full 180 deg, matching the recording-time swerve controller.
        """
        import math

        robot = self.scene["robot"]
        vx = twist[:, 0:1]
        vy = twist[:, 1:2]
        om = twist[:, 2:3]
        mvx = vx - om * self._swerve_module_y  # (N,M)
        mvy = vy + om * self._swerve_module_x
        angles = torch.atan2(mvy, mvx) - self._swerve_angle_offset
        speeds = torch.hypot(mvx, mvy) / self._swerve_wheel_radius

        two_pi = 2.0 * math.pi
        current = robot.data.joint_pos[:, self._swerve_steer_ids]  # (N,M)
        delta = (angles - current + math.pi) % two_pi - math.pi
        flip = delta.abs() > (math.pi / 2.0)
        angles = torch.where(flip, (angles + math.pi + math.pi) % two_pi - math.pi, angles)
        speeds = torch.where(flip, -speeds, speeds)
        angles = current + ((angles - current + math.pi) % two_pi - math.pi)
        return angles, speeds

    def _apply_recorded_base_drive(self, recorded: dict[int, tuple[EpisodeData, int]]) -> None:
        """Set swerve wheel targets from each env's recorded base velocity before the physics step.

        Envs with no recorded state this step (e.g. the stationary grasp phase) are commanded to
        zero so the base stays planted. Must run BEFORE the physics step so physics drives the
        wheels during it.
        """
        if not self._swerve_ready:
            return
        robot = self.scene["robot"]
        twist = torch.zeros(self.num_envs, 3, device=self.device)
        for env_id, (episode, state_index) in recorded.items():
            obs = episode.data.get("obs")
            if not isinstance(obs, dict) or "base_velocity" not in obs:
                continue
            bv = obs["base_velocity"]
            idx = min(int(state_index), bv.shape[0] - 1)
            twist[env_id] = bv[idx].to(self.device)

        angles, speeds = self._swerve_targets(twist)
        pos_target = robot.data.joint_pos_target.clone()
        vel_target = robot.data.joint_vel_target.clone()
        pos_target[:, self._swerve_steer_ids] = angles
        vel_target[:, self._swerve_drive_ids] = speeds
        robot.set_joint_position_target(pos_target)
        robot.set_joint_velocity_target(vel_target)

    def _physics_drive_step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Datagen step that drives the base physically from recorded cmd_vel (no teleport).

        The arms/gripper/lift/head follow the datagen action; the base wheels follow the recorded
        base velocity of the move phase. The grasped box is carried by grip friction (no rigid
        carry), so the whole demo runs on physics.
        """
        action = self._inject_mimic_body_joints(action)
        if self.num_envs > 1:
            recorded = dict(self._mimic_recorded_states)
            self._mimic_recorded_states = {}
            self._mimic_carry_latch_envs = set()
        else:
            recorded = {0: self._mimic_recorded_state} if self._mimic_recorded_state is not None else {}
            self._mimic_recorded_state = None
            self._mimic_carry_latch = False
        self._apply_recorded_base_drive(recorded)
        return super(FFWSG2PickPlaceMimicEnv, self).step(action)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Mobile task: drive the base physically from the recorded cmd_vel (wheels spin, no
        # teleport/physics jitter). Falls back to teleport if swerve joints were not resolved.
        if self._base_physics_drive and self._swerve_ready:
            return self._physics_drive_step(action)
        # num_envs == 1 keeps the validated single-env kinematic path untouched.
        # num_envs > 1 needs per-env handling because a shared physics step cannot
        # mix "teleport (no physics)" and "grasp (physics)" envs in one call.
        if self.num_envs > 1:
            return self._multi_env_step(action)
        return self._legacy_step(action)

    def _legacy_step(self, action: torch.Tensor) -> VecEnvStepReturn:
        action = self._inject_mimic_body_joints(action)
        recorded = self._mimic_recorded_state
        carry_latch = self._mimic_carry_latch
        self._mimic_recorded_state = None
        self._mimic_carry_latch = False
        if recorded is not None:
            if not self._last_step_was_kinematic:
                self._kinematic_step_last_time = None
            episode, state_index = recorded
            self._last_step_was_kinematic = True
            return self._recorded_state_kinematic_step(action, episode, state_index, carry_latch=carry_latch)
        if self._last_step_was_kinematic:
            self._l_motion_ctrl.reset()
        self._last_step_was_kinematic = False
        return super(FFWSG2PickPlaceMimicEnv, self).step(action)

    def _multi_env_step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Per-env datagen step: physics for all envs, then teleport-override L-motion envs.

        The action already carries per-env lift/head (baked by the datagen coroutine),
        so it is NOT re-injected here (that would overwrite all rows with one env's cmd).
        Recorded root/box replay is applied only to the envs that requested it this step.
        """
        recorded = dict(self._mimic_recorded_states)
        carry_ids = set(self._mimic_carry_latch_envs)
        self._mimic_recorded_states = {}
        self._mimic_carry_latch_envs = set()

        ret = super(FFWSG2PickPlaceMimicEnv, self).step(action)

        if not recorded:
            return ret

        for env_id, (episode, state_index) in recorded.items():
            env_ids = torch.tensor([env_id], device=self.device)
            if env_id in carry_ids:
                self._l_motion_ctrl.update_carry_latch(env_ids)
                if bool(self._l_motion_ctrl.is_l_motion_latched(env_ids).any()):
                    apply_recorded_robot_root_state(self, episode, state_index, env_ids=env_ids)
                    self._l_motion_ctrl.apply_latched_carry(env_ids)
            else:
                self._l_motion_ctrl.clear_carry(env_ids)

        self.sim.forward()
        # Refresh obs so the datagen coroutine reads the post-override state.
        self.obs_buf = self.observation_manager.compute()
        return ret

    def _recorded_state_kinematic_step(
        self, action: torch.Tensor, episode: EpisodeData, state_index: int, *, carry_latch: bool = False
    ) -> VecEnvStepReturn:
        """Recording-style step: robot base replay only when bimanual grasp is latched."""
        self.action_manager.process_action(action.to(self.device))
        env_ids = torch.arange(self.num_envs, device=self.device)

        self.scene.update(dt=self.physics_dt)
        if carry_latch:
            self._l_motion_ctrl.update_carry_latch(env_ids)
            if self._l_motion_ctrl.is_l_motion_latched(env_ids).any():
                apply_recorded_robot_root_state(self, episode, state_index, env_ids=env_ids)
                self.scene.update(dt=self.physics_dt)
                self._l_motion_ctrl.apply_latched_carry(env_ids)
        else:
            self._l_motion_ctrl.clear_carry(env_ids)

        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        self._pace_kinematic_step(is_rendering)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _pace_kinematic_step(self, is_rendering: bool) -> None:
        """Hold one env step interval so L-motion matches VR recording / playback speed."""
        step_dt = float(self.step_dt)
        now = time.monotonic()
        if self._kinematic_step_last_time is not None:
            elapsed = now - self._kinematic_step_last_time
            remaining = step_dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
        self._kinematic_step_last_time = time.monotonic()
        if is_rendering:
            self.sim.render()
