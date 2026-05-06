#!/usr/bin/env python3
"""World-frame reactive obstacle benchmark for a moving-base Panda.

The benchmark keeps goals and obstacles fixed in the world frame while the
robot base moves. At every control step, the current base pose is used to
transform the world-frame goal and obstacles into the robot base frame before
solving. This makes the obstacle locations change continuously in the arm base
frame even though the world scene is static.

Two methods are available:

1. curobo: cuRobo reactive MPC, following the structure of
   examples/getting_started/reactive_control.py.
2. mpc_python: the pure-Python Crocoddyl/floating_mpc base-frame OCP from
   panda_nmpc/scripts/base_frame_numeric_sim_main.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import torch

from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml, write_yaml
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PANDA_NMPC_SCRIPTS = WORKSPACE_ROOT / "src" / "panda_nmpc" / "scripts"
DEFAULT_MPC_CONFIG = (
    WORKSPACE_ROOT / "src" / "panda_nmpc" / "config" / "base_frame_numeric_sim.yaml"
)
DEFAULT_OUTPUT_PREFIX = Path("benchmark/log/world_base_motion_reactive")
PANDA_TORQUE_LIMITS_NM = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
DEFAULT_COLLISION_LINKS = (
    "panda_link1_0",
    "panda_link2_0",
    "panda_link3_0",
    "panda_link4_0",
    "panda_link5_0",
    "panda_link6_0",
    "panda_link7_0",
    "panda_hand_0",
)


@dataclass
class WorldScene:
    name: str
    start_q: np.ndarray
    goal_world_pose_wxyz: np.ndarray
    obstacles_world: Dict[str, Dict[str, Dict[str, Any]]]
    goal_tolerance_m: float = 0.03


@dataclass
class BaseMotionSample:
    pose_xyzw: np.ndarray
    twist: np.ndarray
    accel: np.ndarray


@dataclass
class MethodState:
    name: str
    q: np.ndarray
    dq: np.ndarray = field(default_factory=lambda: np.zeros(7))
    prev_dq: np.ndarray = field(default_factory=lambda: np.zeros(7))
    reached: bool = False
    failed: bool = False
    previous_xs: Optional[List[np.ndarray]] = None
    previous_us: Optional[List[np.ndarray]] = None


def load_mpc_module():
    if str(PANDA_NMPC_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PANDA_NMPC_SCRIPTS))
    import base_frame_numeric_sim_main as mpc_module

    return mpc_module


def normalize_xyzw(quat: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    return quat / norm


def xyzw_to_wxyz(quat: Sequence[float]) -> np.ndarray:
    q = normalize_xyzw(quat)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def wxyz_to_xyzw(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    return normalize_xyzw([q[1], q[2], q[3], q[0]])


def pose_xyzw_to_matrix(pose: Sequence[float], mpc_module) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = mpc_module.quat_xyzw_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose_xyzw(transform: np.ndarray, mpc_module) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3], mpc_module.matrix_to_quat_xyzw(transform[:3, :3])]
    )


def pose_wxyz_to_xyzw(pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return np.concatenate([pose[:3], wxyz_to_xyzw(pose[3:7])])


def pose_xyzw_to_wxyz(pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return np.concatenate([pose[:3], xyzw_to_wxyz(pose[3:7])])


def transform_world_pose_to_base_wxyz(
    world_pose_wxyz: Sequence[float],
    base_pose_xyzw: Sequence[float],
    mpc_module,
) -> np.ndarray:
    world_t_base = pose_xyzw_to_matrix(base_pose_xyzw, mpc_module)
    world_t_object = pose_xyzw_to_matrix(pose_wxyz_to_xyzw(world_pose_wxyz), mpc_module)
    base_t_object = np.linalg.inv(world_t_base) @ world_t_object
    return pose_xyzw_to_wxyz(matrix_to_pose_xyzw(base_t_object, mpc_module))


def transform_world_obstacles_to_base(
    obstacles_world: Dict[str, Dict[str, Dict[str, Any]]],
    base_pose_xyzw: Sequence[float],
    mpc_module,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    obstacles_base = deepcopy(obstacles_world)
    for kind, obstacles in obstacles_base.items():
        for obstacle in obstacles.values():
            obstacle["pose"] = transform_world_pose_to_base_wxyz(
                obstacle["pose"], base_pose_xyzw, mpc_module
            ).tolist()
    return obstacles_base


def sample_base_motion(args: argparse.Namespace, time_s: float, mpc_module) -> BaseMotionSample:
    omega = 2.0 * math.pi * args.base_frequency_hz
    phase = np.deg2rad(np.asarray(args.base_phase_deg, dtype=float))
    translation_amp = np.asarray(args.base_translation_amp_m, dtype=float)
    rotation_amp = np.deg2rad(np.asarray(args.base_rotation_amp_deg, dtype=float))

    signal = np.sin(omega * time_s + phase)
    signal_dot = omega * np.cos(omega * time_s + phase)
    signal_ddot = -omega * omega * np.sin(omega * time_s + phase)

    translation_world = translation_amp * signal[:3]
    translation_dot_world = translation_amp * signal_dot[:3]
    translation_ddot_world = translation_amp * signal_ddot[:3]
    rpy = rotation_amp * signal[3:]
    rpy_dot = rotation_amp * signal_dot[3:]
    rpy_ddot = rotation_amp * signal_ddot[3:]

    rotation = mpc_module.euler_xyz_to_matrix(*rpy)
    world_to_base = rotation.T
    # Small-angle angular rates match the base-motion convention used in
    # base_motion_plan_benchmark.py and cuRobo's randomized moving-base tests.
    angular_velocity_body = rpy_dot
    angular_acceleration_body = rpy_ddot
    linear_velocity_body = world_to_base @ translation_dot_world
    linear_acceleration_body = (
        world_to_base @ translation_ddot_world
        - np.cross(angular_velocity_body, linear_velocity_body)
    )

    return BaseMotionSample(
        pose_xyzw=np.concatenate([translation_world, mpc_module.matrix_to_quat_xyzw(rotation)]),
        twist=np.concatenate([linear_velocity_body, angular_velocity_body]),
        accel=np.concatenate([linear_acceleration_body, angular_acceleration_body]),
    )


def base_prediction(
    args: argparse.Namespace, start_time_s: float, horizon: int, dt: float, mpc_module
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = np.zeros((horizon, 7))
    twists = np.zeros((horizon, 6))
    accels = np.zeros((horizon, 6))
    for k in range(horizon):
        sample = sample_base_motion(args, start_time_s + k * dt, mpc_module)
        poses[k] = sample.pose_xyzw
        twists[k] = sample.twist
        accels[k] = sample.accel
    return poses, twists, accels


def curobo_spatial_base_motion(
    base_pose_xyzw: np.ndarray,
    base_twist: np.ndarray,
    base_accel: np.ndarray,
    mpc_module,
    gravity_world: np.ndarray = np.array([0.0, 0.0, -9.81]),
) -> Tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros_like(base_twist, dtype=np.float32)
    acceleration = np.zeros_like(base_accel, dtype=np.float32)
    velocity[:, :3] = base_twist[:, 3:6]
    velocity[:, 3:6] = base_twist[:, 0:3]
    acceleration[:, :3] = base_accel[:, 3:6]
    default_gravity_linear = -gravity_world
    for k, pose in enumerate(base_pose_xyzw):
        rotation_world_base = mpc_module.quat_xyzw_to_matrix(pose[3:7])
        gravity_linear_base = -(rotation_world_base.T @ gravity_world)
        acceleration[k, 3:6] = base_accel[k, 0:3] + gravity_linear_base - default_gravity_linear
    return velocity.astype(np.float32), acceleration.astype(np.float32)


def default_scenes() -> List[WorldScene]:
    return [
        WorldScene(
            name="single_box_corridor",
            start_q=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float),
            goal_world_pose_wxyz=np.array([0.52, -0.34, 0.42, 0.0, 1.0, 0.0, 0.0], dtype=float),
            obstacles_world={
                "cuboid": {
                    "box_mid": {
                        "dims": [0.14, 0.14, 0.45],
                        "pose": [0.45, -0.45, 0.28, 1.0, 0.0, 0.0, 0.0],
                    }
                }
            },
            goal_tolerance_m=0.035,
        ),
    ]


def selected_scenes(names: Sequence[str]) -> List[WorldScene]:
    scenes = default_scenes()
    if not names:
        return scenes
    keep = set(names)
    return [scene for scene in scenes if scene.name in keep]


def scene_cache(obstacles: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, int]:
    # This benchmark currently uses simple primitive scenes. The dict shape is
    # kept generic so more scenes can be added without changing the runner.
    return {"obb": max(1, len(obstacles.get("cuboid", {})))}


def make_curobo_robot_cfg(tool_frame: str, load_dynamics: bool) -> Dict[str, Any]:
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    if "robot_cfg" in robot_cfg:
        robot_cfg = robot_cfg["robot_cfg"]
    robot_cfg = deepcopy(robot_cfg)
    kinematics = robot_cfg["kinematics"]
    kinematics["tool_frames"] = [tool_frame]
    if "attached_object" in kinematics.get("collision_link_names", []):
        kinematics["collision_link_names"].remove("attached_object")
    kinematics["lock_joints"] = {
        "panda_finger_joint1": 0.025,
        "panda_finger_joint2": 0.025,
    }
    robot_cfg["load_dynamics"] = load_dynamics
    return robot_cfg


def make_goal_tool_pose(goal_base_wxyz: Sequence[float], tool_frame: str, ordered_frames: Sequence[str]):
    return GoalToolPose.from_poses(
        {tool_frame: Pose.from_list(list(goal_base_wxyz))},
        ordered_tool_frames=list(ordered_frames),
        num_goalset=1,
    )


def make_joint_state(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, joint_names: Sequence[str], device_cfg) -> JointState:
    state = JointState.from_position(
        torch.as_tensor(q, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1),
        joint_names=list(joint_names),
    )
    state.velocity = torch.as_tensor(dq, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1)
    state.acceleration = torch.as_tensor(ddq, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1)
    return state


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_curobo_controller(scene: WorldScene, args: argparse.Namespace, mpc_module):
    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo method requires CUDA; run with --methods mpc_python to skip it.")
    base0 = sample_base_motion(args, 0.0, mpc_module)
    obstacles_base = transform_world_obstacles_to_base(
        scene.obstacles_world, base0.pose_xyzw, mpc_module
    )
    cfg = ModelPredictiveControlCfg.create(
        robot=make_curobo_robot_cfg(args.tool_frame, load_dynamics=True),
        scene_model=obstacles_base,
        collision_cache=scene_cache(obstacles_base),
        use_cuda_graph=not args.disable_cuda_graph,
        optimization_dt=args.curobo_dt,
        interpolation_steps=args.curobo_interpolation_steps,
        optimizer_collision_activation_distance=args.collision_activation_distance,
        position_tolerance=args.position_tolerance,
        orientation_tolerance=args.orientation_tolerance,
        warm_start_optimization_num_iters=args.curobo_warm_iters,
        cold_start_optimization_num_iters=args.curobo_cold_iters,
        device_cfg=DeviceCfg(),
    )
    controller = ModelPredictiveControl(cfg)
    controller.update_links_inertial({"attached_object": {"mass": args.mass}})
    horizon = max(2, int(controller.action_horizon))
    base_velocity = torch.zeros((horizon, 6), device=controller.device_cfg.device, dtype=torch.float32)
    base_acceleration = torch.zeros_like(base_velocity)
    base_pose, base_twist, base_accel = base_prediction(
        args, 0.0, horizon, args.curobo_dt, mpc_module
    )
    velocity_np, acceleration_np = curobo_spatial_base_motion(
        base_pose, base_twist, base_accel, mpc_module
    )
    base_velocity.copy_(torch.as_tensor(velocity_np, device=base_velocity.device))
    base_acceleration.copy_(torch.as_tensor(acceleration_np, device=base_acceleration.device))
    controller.core.set_base_motion(base_velocity, base_acceleration)
    current = make_joint_state(
        scene.start_q,
        np.zeros(7),
        np.zeros(7),
        controller.joint_names,
        controller.device_cfg,
    )
    controller.setup(current)
    goal_base = transform_world_pose_to_base_wxyz(
        scene.goal_world_pose_wxyz, base0.pose_xyzw, mpc_module
    )
    controller.update_goal_tool_poses(
        make_goal_tool_pose(goal_base, controller.tool_frames[0], controller.tool_frames),
        run_ik=args.curobo_run_ik,
        use_best_effort_ik=True,
    )
    return controller, base_velocity, base_acceleration


def update_curobo_world_and_goal(
    controller,
    scene: WorldScene,
    base_sample: BaseMotionSample,
    args: argparse.Namespace,
    mpc_module,
) -> None:
    obstacles_base = transform_world_obstacles_to_base(
        scene.obstacles_world, base_sample.pose_xyzw, mpc_module
    )
    for obstacles in obstacles_base.values():
        for name, obstacle in obstacles.items():
            controller.scene_collision_checker.update_obstacle_pose(
                name, Pose.from_list(obstacle["pose"])
            )
    goal_base = transform_world_pose_to_base_wxyz(
        scene.goal_world_pose_wxyz, base_sample.pose_xyzw, mpc_module
    )
    controller.update_goal_tool_poses(
        make_goal_tool_pose(goal_base, controller.tool_frames[0], controller.tool_frames),
        run_ik=args.curobo_run_ik,
        use_best_effort_ik=True,
    )


def shift_warm_start(xs: Optional[List[np.ndarray]], us: Optional[List[np.ndarray]], x0: np.ndarray):
    if xs is None or us is None or len(xs) < 2 or len(us) + 1 != len(xs):
        return None, None
    xs_init = [x.copy() for x in xs[1:]] + [xs[-1].copy()]
    us_init = [u.copy() for u in us[1:]] + [us[-1].copy()]
    xs_init[0] = x0.copy()
    return xs_init, us_init


def solve_python_mpc_step(
    state: MethodState,
    scene: WorldScene,
    time_s: float,
    args: argparse.Namespace,
    mpc_module,
) -> Tuple[float, Dict[str, Any]]:
    config = mpc_module.load_config(args.mpc_config)
    config.planner.T = args.mpc_horizon
    config.planner.dt_ocp = args.mpc_dt
    config.planner.nb_iterations_max = args.mpc_iterations
    config.planner.max_qp_iter = args.mpc_max_qp_iter
    config.planner.ee_frame_name = args.tool_frame
    config.planner.collision_safety_margin = args.mpc_collision_safety_margin

    base_pose, base_twist, base_accel = base_prediction(
        args, time_s, config.planner.T + 1, config.planner.dt_ocp, mpc_module
    )
    obstacles_base = transform_world_obstacles_to_base(
        scene.obstacles_world, base_pose[0], mpc_module
    )
    goal_base_wxyz = transform_world_pose_to_base_wxyz(
        scene.goal_world_pose_wxyz, base_pose[0], mpc_module
    )
    config.simulation.target_pose_in_base = pose_wxyz_to_xyzw(goal_base_wxyz)

    floating_model = mpc_module.load_floating_panda_model(config.robot_model.urdf_path)
    fixed_model = mpc_module.load_panda_model(config.robot_model.urdf_path)
    collision_model = mpc_module.build_collision_model_with_obstacles(
        floating_model,
        config.robot_model.urdf_path,
        config.robot_model.package_dirs,
        obstacles_base,
        args.mpc_collision_links,
        args.mpc_ignore_unsupported_obstacles,
    )
    planner = mpc_module.BaseFrameReachingPy(floating_model, collision_model, config.planner)
    x0 = np.concatenate([state.q, state.dq])
    planner.ocp.problem.x0 = x0
    planner.set_base_motion_prediction(list(base_pose), list(base_twist), list(base_accel))
    planner.set_ee_ref_base_placement_list_constant_weights(
        config.simulation.target_pose_in_base,
        np.zeros(6),
        True,
        1.0,
    )
    posture_ref = np.zeros(14)
    posture_ref[:7] = state.q
    planner.set_posture_ref(posture_ref)

    xs_init, us_init = shift_warm_start(state.previous_xs, state.previous_us, x0)
    if xs_init is None or us_init is None:
        xs_init = [x0.copy() for _ in range(config.planner.T + 1)]
        us_init = [
            mpc_module.compute_floating_inverse_dynamics(
                floating_model,
                state.q,
                state.dq,
                np.zeros(7),
                base_pose[i],
                base_twist[i],
                base_accel[i],
            )
            for i in range(config.planner.T)
        ]

    start = time.perf_counter()
    planner.solve(xs_init, us_init)
    solve_time = time.perf_counter() - start
    xs = [np.asarray(x, dtype=float).copy() for x in planner.ocp.xs]
    us = [np.asarray(u, dtype=float).copy() for u in planner.ocp.us]
    state.previous_xs = xs
    state.previous_us = us

    command_index = min(max(1, args.mpc_command_state_index), len(xs) - 1)
    next_q = xs[command_index][:7].copy()
    next_dq = xs[command_index][7:].copy()
    ddq = (next_dq - state.dq) / max(args.sim_dt, 1e-9)
    tau = mpc_module.compute_floating_inverse_dynamics(
        floating_model, state.q, state.dq, ddq, base_pose[0], base_twist[0], base_accel[0]
    )
    state.prev_dq = state.dq.copy()
    state.q = next_q
    state.dq = next_dq
    min_distance = mpc_module.min_collision_distance(floating_model, collision_model, xs)
    return solve_time, {
        "tau": tau,
        "min_collision_distance_m": min_distance,
        "trajectory_length": len(xs),
    }


def solve_curobo_step(
    state: MethodState,
    controller,
    base_velocity: torch.Tensor,
    base_acceleration: torch.Tensor,
    scene: WorldScene,
    time_s: float,
    args: argparse.Namespace,
    mpc_module,
) -> Tuple[float, Dict[str, Any]]:
    base_sample = sample_base_motion(args, time_s, mpc_module)
    update_curobo_world_and_goal(controller, scene, base_sample, args, mpc_module)
    base_pose, base_twist, base_accel = base_prediction(
        args, time_s, base_velocity.shape[0], args.curobo_dt, mpc_module
    )
    velocity_np, acceleration_np = curobo_spatial_base_motion(
        base_pose, base_twist, base_accel, mpc_module
    )
    base_velocity.copy_(torch.as_tensor(velocity_np, device=base_velocity.device))
    base_acceleration.copy_(torch.as_tensor(acceleration_np, device=base_acceleration.device))

    current_state = make_joint_state(
        state.q,
        state.dq,
        (state.dq - state.prev_dq) / max(args.sim_dt, 1e-9),
        controller.joint_names,
        controller.device_cfg,
    )
    sync_cuda()
    start = time.perf_counter()
    result = controller.optimize_action_sequence(current_state)
    sync_cuda()
    solve_time = time.perf_counter() - start

    if result.action_sequence is None or result.action_sequence.position.shape[1] == 0:
        state.failed = True
        return solve_time, {"status": "no_action_sequence"}

    command_index = args.curobo_command_index
    if command_index < 0:
        command_index = result.action_sequence.position.shape[1] + command_index
    command_index = int(np.clip(command_index, 0, result.action_sequence.position.shape[1] - 1))
    next_q = result.action_sequence.position[:, command_index, :].detach().cpu().numpy().reshape(-1)[:7]
    next_dq = (
        result.action_sequence.velocity[:, command_index, :].detach().cpu().numpy().reshape(-1)[:7]
        if result.action_sequence.velocity is not None
        else (next_q - state.q) / max(args.sim_dt, 1e-9)
    )
    ddq = (next_dq - state.dq) / max(args.sim_dt, 1e-9)
    state.prev_dq = state.dq.copy()
    state.q = next_q
    state.dq = next_dq
    return solve_time, {
        "position_error_m": float(result.position_error.detach().cpu().reshape(-1)[0])
        if result.position_error is not None
        else float("nan"),
        "trajectory_length": int(result.action_sequence.position.shape[1]),
        "ddq": ddq,
    }


def evaluate_state(
    method_state: MethodState,
    scene: WorldScene,
    time_s: float,
    solve_time_s: float,
    step_info: Dict[str, Any],
    args: argparse.Namespace,
    mpc_module,
) -> Dict[str, Any]:
    config = mpc_module.load_config(args.mpc_config)
    fixed_model = mpc_module.load_panda_model(config.robot_model.urdf_path)
    floating_model = mpc_module.load_floating_panda_model(config.robot_model.urdf_path)
    frame_id = fixed_model.getFrameId(args.tool_frame)
    base_sample = sample_base_motion(args, time_s, mpc_module)
    ee_base = mpc_module.compute_fixed_fk_pose_xyzw(
        fixed_model, frame_id, method_state.q, method_state.dq
    )
    ee_world = matrix_to_pose_xyzw(
        pose_xyzw_to_matrix(base_sample.pose_xyzw, mpc_module)
        @ pose_xyzw_to_matrix(ee_base, mpc_module),
        mpc_module,
    )
    goal_world = pose_wxyz_to_xyzw(scene.goal_world_pose_wxyz)
    position_error_m = float(np.linalg.norm(ee_world[:3] - goal_world[:3]))

    obstacles_base = transform_world_obstacles_to_base(
        scene.obstacles_world, base_sample.pose_xyzw, mpc_module
    )
    collision_model = mpc_module.build_collision_model_with_obstacles(
        floating_model,
        config.robot_model.urdf_path,
        config.robot_model.package_dirs,
        obstacles_base,
        args.mpc_collision_links,
        args.mpc_ignore_unsupported_obstacles,
    )
    x = np.concatenate([method_state.q, method_state.dq])
    min_distance = mpc_module.min_collision_distance(floating_model, collision_model, [x])
    ddq = step_info.get("ddq", (method_state.dq - method_state.prev_dq) / max(args.sim_dt, 1e-9))
    tau = step_info.get(
        "tau",
        mpc_module.compute_floating_inverse_dynamics(
            floating_model,
            method_state.q,
            method_state.dq,
            ddq,
            base_sample.pose_xyzw,
            base_sample.twist,
            base_sample.accel,
        ),
    )
    method_state.reached = position_error_m <= scene.goal_tolerance_m
    return {
        "method": method_state.name,
        "scene": scene.name,
        "time_s": time_s,
        "solve_time_s": solve_time_s,
        "position_error_m": position_error_m,
        "goal_reached": int(method_state.reached),
        "min_collision_distance_m": min(float(min_distance), float(step_info.get("min_collision_distance_m", min_distance))),
        "collision": int(min_distance < args.mpc_collision_safety_margin),
        "max_abs_tau_nm": float(np.max(np.abs(tau))),
        "rms_tau_nm": float(math.sqrt(np.mean(tau * tau))),
        "mean_abs_power_w": float(np.mean(np.abs(tau * method_state.dq))),
        "torque_violation": int(np.any(np.abs(tau) > PANDA_TORQUE_LIMITS_NM)),
        "trajectory_length": int(step_info.get("trajectory_length", 1)),
        "status": step_info.get("status", "ok"),
    }


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for scene in sorted({row["scene"] for row in rows}):
        for method in sorted({row["method"] for row in rows if row["scene"] == scene}):
            method_rows = [row for row in rows if row["scene"] == scene and row["method"] == method]
            if not method_rows:
                continue
            finite_solve = [float(row["solve_time_s"]) for row in method_rows if math.isfinite(float(row["solve_time_s"]))]
            summaries.append(
                {
                    "scene": scene,
                    "method": method,
                    "steps": len(method_rows),
                    "goal_reached": int(any(int(row["goal_reached"]) for row in method_rows)),
                    "final_position_error_m": float(method_rows[-1]["position_error_m"]),
                    "min_collision_distance_m": float(
                        min(row["min_collision_distance_m"] for row in method_rows)
                    ),
                    "collision_count": int(sum(int(row["collision"]) for row in method_rows)),
                    "torque_violation_count": int(
                        sum(int(row["torque_violation"]) for row in method_rows)
                    ),
                    "mean_solve_time_s": float(np.mean(finite_solve)) if finite_solve else float("nan"),
                    "max_solve_time_s": float(np.max(finite_solve)) if finite_solve else float("nan"),
                    "max_abs_tau_nm": float(max(row["max_abs_tau_nm"] for row in method_rows)),
                    "mean_abs_power_w": float(np.mean([row["mean_abs_power_w"] for row in method_rows])),
                }
            )
    return summaries


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_scene(scene: WorldScene, args: argparse.Namespace, mpc_module) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    methods = set(args.methods)
    curobo_controller = None
    curobo_base_velocity = None
    curobo_base_acceleration = None
    states: Dict[str, MethodState] = {
        method: MethodState(method, scene.start_q.copy()) for method in methods
    }

    if "curobo" in methods:
        curobo_controller, curobo_base_velocity, curobo_base_acceleration = build_curobo_controller(
            scene, args, mpc_module
        )

    for step in range(args.num_steps):
        time_s = step * args.sim_dt
        for method in args.methods:
            state = states[method]
            if state.reached or state.failed:
                continue
            try:
                if method == "curobo":
                    solve_time, info = solve_curobo_step(
                        state,
                        curobo_controller,
                        curobo_base_velocity,
                        curobo_base_acceleration,
                        scene,
                        time_s,
                        args,
                        mpc_module,
                    )
                elif method == "mpc_python":
                    solve_time, info = solve_python_mpc_step(state, scene, time_s, args, mpc_module)
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as exc:
                state.failed = True
                solve_time = float("nan")
                info = {"status": f"exception:{type(exc).__name__}:{exc}"}
            row = evaluate_state(state, scene, time_s, solve_time, info, args, mpc_module)
            row["step"] = step
            rows.append(row)
        if args.stop_on_all_done and all(s.reached or s.failed for s in states.values()):
            break

    if curobo_controller is not None:
        curobo_controller.destroy()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reactive world-frame moving-base obstacle benchmark for cuRobo and Python MPC."
    )
    parser.add_argument("--methods", nargs="+", default=["curobo", "mpc_python"], choices=["curobo", "mpc_python"])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--tool-frame", type=str, default="panda_hand")
    parser.add_argument("--mass", type=float, default=3.0)
    parser.add_argument("--stop-on-all-done", action="store_true")
    parser.add_argument("--collision-activation-distance", type=float, default=0.03)
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.10)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--curobo-dt", type=float, default=0.025)
    parser.add_argument("--curobo-interpolation-steps", type=int, default=4)
    parser.add_argument("--curobo-warm-iters", type=int, default=80)
    parser.add_argument("--curobo-cold-iters", type=int, default=120)
    parser.add_argument("--curobo-command-index", type=int, default=-1)
    parser.add_argument("--curobo-run-ik", action="store_true")
    parser.add_argument("--mpc-config", type=Path, default=DEFAULT_MPC_CONFIG)
    parser.add_argument("--mpc-horizon", type=int, default=60)
    parser.add_argument("--mpc-dt", type=float, default=0.02)
    parser.add_argument("--mpc-iterations", type=int, default=5)
    parser.add_argument("--mpc-max-qp-iter", type=int, default=80)
    parser.add_argument("--mpc-command-state-index", type=int, default=1)
    parser.add_argument("--mpc-collision-safety-margin", type=float, default=0.04)
    parser.add_argument("--mpc-collision-links", nargs="+", default=list(DEFAULT_COLLISION_LINKS))
    parser.add_argument("--mpc-ignore-unsupported-obstacles", action="store_true")
    parser.add_argument("--base-frequency-hz", type=float, default=0.35)
    parser.add_argument("--base-translation-amp-m", nargs=3, type=float, default=[0.10, 0.06, 0.04])
    parser.add_argument("--base-rotation-amp-deg", nargs=3, type=float, default=[8.0, 6.0, 2.0])
    parser.add_argument("--base-phase-deg", nargs=6, type=float, default=[0.0, 90.0, 180.0, 0.0, 45.0, 90.0])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args.sim_dt <= 0.0:
        raise ValueError("--sim-dt must be positive")
    if "curobo" in args.methods and not torch.cuda.is_available():
        raise RuntimeError("cuRobo method requires CUDA; use --methods mpc_python on CPU-only sessions.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    setup_curobo_logger("error")
    mpc_module = load_mpc_module()
    scenes = selected_scenes(args.scenes)
    if not scenes:
        raise ValueError("No scenes selected")

    rows: List[Dict[str, Any]] = []
    for scene in scenes:
        rows.extend(run_scene(scene, args, mpc_module))
    summary_rows = summarize(rows)

    trial_path = args.output_prefix.with_name(args.output_prefix.name + "_trials.csv")
    summary_csv_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary_yaml_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.yml")
    write_csv(trial_path, rows)
    write_csv(summary_csv_path, summary_rows)
    write_yaml(summary_rows, str(summary_yaml_path))
    for row in summary_rows:
        print(
            "{scene} {method}: reached={goal_reached}, final_error={final_position_error_m:.4f}m, "
            "min_dist={min_collision_distance_m:.4f}m, mean_solve={mean_solve_time_s:.4f}s".format(**row)
        )
    print(f"Wrote trials to: {trial_path}")
    print(f"Wrote summary to: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
