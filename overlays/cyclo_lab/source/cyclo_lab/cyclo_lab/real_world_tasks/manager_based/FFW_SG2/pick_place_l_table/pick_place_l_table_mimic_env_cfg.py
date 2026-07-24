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

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.mimic_dual_arm import configure_ltable_dual_arm_mimic
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place_l_table.joint_pos_env_cfg import (
    EventCfg,
    FFWSG2PickPlaceLTableEnvCfg,
    FFWSG2PickPlaceLTableMobileEnvCfg,
)

from . import mdp


@configclass
class LTableMimicEventCfg(EventCfg):
    """Scene events for mimic datagen (includes kinematic L-motion between grasp and place)."""

    scripted_l_motion = EventTerm(
        func=mdp.scripted_l_motion_step,
        mode="interval",
        interval_range_s=(0.05, 0.05),
    )


@configclass
class FFWSG2PickPlaceLTableMimicEnvCfg(FFWSG2PickPlaceLTableEnvCfg, MimicEnvCfg):
    """Mimic config for L-table pick and place."""

    scripted_l_motion_enable: bool = False
    events: LTableMimicEventCfg = LTableMimicEventCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_ltable_dual_arm_mimic(
            self,
            datagen_name="ltable_pick_place_box",
            place_signal="box_on_left_table",
            place_description="Place box on left table",
        )


@configclass
class FFWSG2PickPlaceLTableMobileMimicEnvCfg(FFWSG2PickPlaceLTableMobileEnvCfg, MimicEnvCfg):
    """Mimic config for the drivable-base (Plan B) L-table task.

    Same L-table datagen as the stock mimic, but built on the mobile env so generated demos
    keep the ``base_velocity`` observation (linear_x/y + angular_z). The IK action stays 19-dim
    throughout datagen; the LeRobot converter appends base_velocity -> 22-dim, matching the real
    ffw_sg2_rev1 mobile data. Base motion is reproduced by replaying the recorded root pose AND
    velocity (see ``apply_recorded_robot_root_state``), so base_velocity is non-zero during the
    drive. The stock 19-dim mimic task is left untouched.
    """

    scripted_l_motion_enable: bool = False
    events: LTableMimicEventCfg = LTableMimicEventCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_ltable_dual_arm_mimic(
            self,
            datagen_name="ltable_pick_place_box_mobile",
            place_signal="box_on_left_table",
            place_description="Place box on left table",
        )
