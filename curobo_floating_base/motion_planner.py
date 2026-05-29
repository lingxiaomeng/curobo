# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion planning module.

This module provides high-level motion planning combining trajectory optimization
and graph planning.

Example:
    ```python
    from curobo_floating_base import MotionPlanner, MotionPlannerCfg

    config = MotionPlannerCfg.create(robot="franka.yml")
    planner = MotionPlanner(config)
    result = planner.plan_pose(goal_tool_poses, current_state)
    ```
"""

from curobo_floating_base._src.motion.motion_planner import MotionPlanner
from curobo_floating_base._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo_floating_base._src.motion.motion_planner_result import GraspPlanResult

__all__ = [
    "MotionPlanner",
    "MotionPlannerCfg",
    "GraspPlanResult",
]
