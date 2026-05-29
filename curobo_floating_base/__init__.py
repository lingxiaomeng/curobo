# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuRobo provides accelerated modules for robotics which can be used to build high-performance
robotics applications. The library has several modules for numerical optimization, robot kinematics,
geometry processing, collision checking, graph search planning. cuRobo provides high-level APIs for
performing tasks like collision-free inverse kinematics, model predictive control, and motion
planning.

Public API Quick Start:

.. code-block:: python

    from curobo_floating_base.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo_floating_base.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
    from curobo_floating_base.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo_floating_base.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo_floating_base.kinematics import Kinematics, KinematicsCfg
    from curobo_floating_base.scene import Scene, Cuboid, Sphere, Mesh
    from curobo_floating_base.types import JointState

    # Forward kinematics
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
    js = JointState.from_position(q, joint_names=kin.joint_names)
    state = kin.compute_kinematics(js)

    # Scene representation
    scene = Scene(
        cuboid=[Cuboid(name="table", dims=[1, 1, 0.1], pose=[0, 0, 0.5, 1, 0, 0, 0])],
    )

    # Inverse kinematics
    ik_config = InverseKinematicsCfg.create(robot="franka.yml")
    ik = InverseKinematics(ik_config)
    result = ik.solve_pose(goal_tool_poses=target_poses)

    # Trajectory optimization
    trajopt = TrajectoryOptimizer(trajopt_config)
    trajectory = trajopt.solve_pose(goal_tool_poses=target_poses)

    # Model predictive control
    mpc = ModelPredictiveControl(mpc_config)
    result = mpc.optimize_action_sequence(current_state)


Public API Modules:

- :mod:`curobo_floating_base.kinematics` - Forward kinematics
- :mod:`curobo_floating_base.inverse_kinematics` - Inverse kinematics solver
- :mod:`curobo_floating_base.trajectory_optimizer` - Trajectory optimization
- :mod:`curobo_floating_base.motion_planner` - Motion planning
- :mod:`curobo_floating_base.model_predictive_control` - Model predictive control
- :mod:`curobo_floating_base.motion_retargeter` - Motion retargeting using inverse kinematics and model predictive control
- :mod:`curobo_floating_base.scene` - Scene representation with obstacles
- :mod:`curobo_floating_base.collision_checking` - Robot collision checking (for custom pipelines)
- :mod:`curobo_floating_base.perception` - Perception utilities (robot segmentation)
- :mod:`curobo_floating_base.robot_builder` - Build robot configs from URDF
- :mod:`curobo_floating_base.viewer` - Visualization (Rerun, Viser)
- :mod:`curobo_floating_base.types` - Common data types (JointState, Pose, etc.)

"""

from curobo_floating_base._version import __version__
