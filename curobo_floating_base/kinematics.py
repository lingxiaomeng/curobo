# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward kinematics module.

Differentiable forward kinematics with Jacobian computation, center of mass
calculation, and collision sphere generation.

Example:
    ```python
    from curobo_floating_base.kinematics import Kinematics, KinematicsCfg
    from curobo_floating_base.types import JointState

    config = KinematicsCfg.from_robot_yaml_file("franka.yml")
    kin = Kinematics(config)
    js = JointState.from_position(q, joint_names=kin.joint_names)
    state = kin.compute_kinematics(js)
    ```
"""

from curobo_floating_base._src.robot.kinematics.kinematics import Kinematics
from curobo_floating_base._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo_floating_base._src.robot.kinematics.kinematics_state import KinematicsState

__all__ = [
    "Kinematics",
    "KinematicsCfg",
    "KinematicsState",
]
