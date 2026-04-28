# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Motion planning benchmark with randomized moving-base dynamics.

This script is intentionally separate from ``motion_plan_benchmark.py``. It runs
the same planning problems with three planner variants:

1. no_dynamics: cuRobo planning without inverse dynamics.
2. fixed_base_dynamics: inverse dynamics enabled with a fixed base.
3. base_motion_dynamics: inverse dynamics enabled with randomized base motion.

The randomized base motion is deterministic for a given seed and scene group.
It is injected before warmup into cuRobo RNEA in spatial
[angular xyz, linear xyz] order.
"""

from __future__ import annotations

# Standard Library
import argparse
import csv
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Avoid robometrics/geometrout numba cache failures in editable or relocated
# conda/mamba environments. Dataset loading is not a benchmarked hot path here.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Third Party
import numpy as np
import torch
from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from tqdm import tqdm

# cuRobo
from curobo._src.geom.types import SceneCfg
from curobo._src.motion import MotionPlanner, MotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.robot import RobotCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.benchmark_metrics import CuroboGroupMetrics, CuroboMetrics
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util_file import (
    get_robot_configs_path,
    get_scene_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class MethodSpec:
    name: str
    load_dynamics: bool
    use_base_motion: bool


@dataclass
class PlannerBundle:
    method: MethodSpec
    planner: MotionPlanner
    base_velocity: Optional[torch.Tensor] = None
    base_acceleration: Optional[torch.Tensor] = None
    base_horizon: int = 0
    base_dt: float = 0.01


METHODS = (
    MethodSpec("no_dynamics", load_dynamics=False, use_base_motion=False),
    MethodSpec("fixed_base_dynamics", load_dynamics=True, use_base_motion=False),
    MethodSpec("base_motion_dynamics", load_dynamics=True, use_base_motion=True),
)

PANDA_TORQUE_LIMITS_NM = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def scalar(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def check_problems(all_problems: Sequence[Dict[str, Any]], mesh: bool = False) -> int:
    max_cache = 0
    cache_key = "mesh" if mesh else "obb"
    for problem in all_problems:
        world = SceneCfg.create(deepcopy(problem["obstacles"]))
        if mesh:
            cache = world.get_mesh_world().get_cache_dict()
        else:
            cache = world.get_obb_world().get_cache_dict()
        max_cache = max(max_cache, int(cache[cache_key]))
    return max_cache


def build_world(problem: Dict[str, Any], mesh: bool):
    world_cfg = SceneCfg.create(deepcopy(problem["obstacles"]))
    if mesh:
        return world_cfg.get_mesh_world(merge_meshes=False)
    return world_cfg.get_obb_world()


def get_datasets(dataset: str):
    if dataset == "demo":
        return [("demo", demo_raw())]
    if dataset == "motion_benchmaker":
        return [("motion_benchmaker", motion_benchmaker_raw())]
    if dataset == "mpinets":
        return [("mpinets", mpinets_raw())]
    return [
        ("motion_benchmaker", motion_benchmaker_raw()),
        ("mpinets", mpinets_raw()),
    ]


def is_mpinets_dataset(problems: Dict[str, Any]) -> bool:
    return "dresser_task_oriented" in problems


def load_robot_cfg_dict(
    method: MethodSpec,
    mpinets: bool,
    collision_buffer: float,
) -> Dict[str, Any]:
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    if "robot_cfg" in robot_cfg:
        robot_cfg = robot_cfg["robot_cfg"]

    robot_cfg["kinematics"]["collision_sphere_buffer"] = collision_buffer
    if "attached_object" in robot_cfg["kinematics"]["collision_link_names"]:
        robot_cfg["kinematics"]["collision_link_names"].remove("attached_object")
    robot_cfg["kinematics"]["tool_frames"] = ["panda_hand"]

    robot_cfg["kinematics"]["lock_joints"] = {
        "panda_finger_joint1": 0.025,
        "panda_finger_joint2": 0.025,
    }
    robot_cfg["load_dynamics"] = method.load_dynamics
    return robot_cfg


def iter_solver_rollouts(planner: MotionPlanner):
    for solver in (planner.ik_solver, planner.trajopt_solver):
        core = solver.core
        yield from core.get_all_rollout_instances()
        yield from core.additional_metrics_rollouts.values()


def max_rollout_horizon(planner: MotionPlanner) -> int:
    return max(int(rollout.horizon) for rollout in iter_solver_rollouts(planner))


def infer_base_motion_dt(planner: MotionPlanner, fallback: float = 0.01) -> float:
    for rollout in planner.trajopt_solver.core.optimizer_rollouts:
        dt = float(rollout.dt)
        if dt > 0.0:
            return dt
    return fallback


def make_planner_bundle(
    method: MethodSpec,
    n_obstacles: int,
    mpinets: bool,
    args: argparse.Namespace,
) -> Tuple[PlannerBundle, Dict[str, Any]]:
    robot_cfg_dict = load_robot_cfg_dict(method, mpinets, collision_buffer=0.0)
    scene_cfg = SceneCfg.create(
        load_yaml(join_path(get_scene_configs_path(), "collision_table.yml"))
    ).get_obb_world()
    collision_cache = {"obb": n_obstacles}
    if args.mesh:
        scene_cfg = scene_cfg.get_mesh_world()
        collision_cache = {"mesh": n_obstacles}

    robot_cfg = RobotCfg.create(deepcopy(robot_cfg_dict), device_cfg=DeviceCfg())
    joint_limits = robot_cfg.kinematics.kinematics_config.joint_limits
    joint_limits.position[0, :] -= 0.2
    joint_limits.position[1, :] += 0.2

    collision_activation_distance = 0.0 if args.graph else args.collision_activation_distance
    trajopt_seeds = 4 if args.graph else args.trajopt_seeds

    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        scene_model=scene_cfg,
        ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
        ik_transition_model="ik/transition_ik.yml",
        metrics_rollout="metrics_base.yml",
        trajopt_optimizer_configs=[
            "trajopt/particle_trajopt.yml",
            "trajopt/lbfgs_bspline_trajopt.yml",
        ],
        trajopt_transition_model="trajopt/transition_bspline_trajopt.yml",
        use_cuda_graph=not args.disable_cuda_graph,
        num_ik_seeds=args.ik_seeds,
        num_trajopt_seeds=trajopt_seeds,
        collision_cache=collision_cache,
        store_debug=False,
        optimizer_collision_activation_distance=collision_activation_distance,
    )
    planner = MotionPlanner(planner_cfg)
    if method.load_dynamics:
        planner.update_links_inertial({"attached_object": {"mass": args.mass}})

    bundle = PlannerBundle(method=method, planner=planner)
    if method.use_base_motion:
        bundle.base_horizon = max_rollout_horizon(planner)
        bundle.base_dt = args.base_motion_dt or infer_base_motion_dt(planner)
        device = planner.device_cfg.device
        bundle.base_velocity = torch.zeros(
            (bundle.base_horizon, 6), device=device, dtype=torch.float32
        )
        bundle.base_acceleration = torch.zeros_like(bundle.base_velocity)
        planner.set_base_motion(bundle.base_velocity, bundle.base_acceleration)

    return bundle, robot_cfg_dict


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def make_random_base_motion(
    horizon: int,
    dt: float,
    seed: int,
    args: argparse.Namespace,
    gravity_world: np.ndarray = np.array([0.0, 0.0, -9.81]),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate smooth random base motion in cuRobo spatial order.

    The angular velocity/acceleration are small-angle roll/pitch/yaw rates.
    Linear acceleration includes the gravity-direction correction expected by
    cuRobo's moving-root RNEA interface.
    """
    rng = np.random.default_rng(seed)
    time_steps = np.arange(horizon, dtype=np.float64) * dt
    freq = rng.uniform(args.base_freq_min, args.base_freq_max, size=(2, 3))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(2, 3))

    angle_amp = np.deg2rad(args.base_angle_amp_deg) * rng.uniform(0.25, 1.0, size=3)
    angle_amp[2] *= args.base_yaw_scale
    linear_amp = args.base_linear_amp_m * rng.uniform(0.25, 1.0, size=3)

    angle_omega = 2.0 * math.pi * freq[0]
    linear_omega = 2.0 * math.pi * freq[1]

    angles = np.sin(time_steps[:, None] * angle_omega[None, :] + phase[0]) * angle_amp
    angular_velocity = (
        np.cos(time_steps[:, None] * angle_omega[None, :] + phase[0])
        * angle_amp
        * angle_omega
    )
    angular_acceleration = (
        -np.sin(time_steps[:, None] * angle_omega[None, :] + phase[0])
        * angle_amp
        * angle_omega**2
    )

    linear_velocity = (
        np.cos(time_steps[:, None] * linear_omega[None, :] + phase[1])
        * linear_amp
        * linear_omega
    )
    linear_acceleration = (
        -np.sin(time_steps[:, None] * linear_omega[None, :] + phase[1])
        * linear_amp
        * linear_omega**2
    )

    default_gravity_linear = -gravity_world
    base_velocity = np.zeros((horizon, 6), dtype=np.float32)
    base_acceleration = np.zeros((horizon, 6), dtype=np.float32)
    base_velocity[:, :3] = angular_velocity.astype(np.float32)
    base_velocity[:, 3:] = linear_velocity.astype(np.float32)
    base_acceleration[:, :3] = angular_acceleration.astype(np.float32)

    for i in range(horizon):
        rotation_world_base = euler_xyz_to_matrix(*angles[i])
        gravity_linear_base = -(rotation_world_base.T @ gravity_world)
        base_acceleration[i, 3:] = (
            linear_acceleration[i] + gravity_linear_base - default_gravity_linear
        ).astype(np.float32)

    stats = summarize_base_motion(base_velocity, base_acceleration)
    stats["base_motion_seed"] = float(seed)
    return base_velocity, base_acceleration, stats


def summarize_base_motion(
    base_velocity: np.ndarray,
    base_acceleration: np.ndarray,
) -> Dict[str, float]:
    return {
        "base_ang_vel_rms": float(np.sqrt(np.mean(base_velocity[:, :3] ** 2))),
        "base_lin_vel_rms": float(np.sqrt(np.mean(base_velocity[:, 3:] ** 2))),
        "base_ang_acc_rms": float(np.sqrt(np.mean(base_acceleration[:, :3] ** 2))),
        "base_lin_acc_rms": float(np.sqrt(np.mean(base_acceleration[:, 3:] ** 2))),
        "base_ang_vel_max": float(np.max(np.abs(base_velocity[:, :3]))),
        "base_lin_vel_max": float(np.max(np.abs(base_velocity[:, 3:]))),
        "base_ang_acc_max": float(np.max(np.abs(base_acceleration[:, :3]))),
        "base_lin_acc_max": float(np.max(np.abs(base_acceleration[:, 3:]))),
    }


def update_base_motion_buffer(
    bundle: PlannerBundle,
    base_velocity: np.ndarray,
    base_acceleration: np.ndarray,
) -> None:
    if bundle.base_velocity is None or bundle.base_acceleration is None:
        return
    bundle.base_velocity.copy_(
        torch.as_tensor(base_velocity, device=bundle.base_velocity.device, dtype=torch.float32)
    )
    bundle.base_acceleration.copy_(
        torch.as_tensor(
            base_acceleration,
            device=bundle.base_acceleration.device,
            dtype=torch.float32,
        )
    )


def get_dynamics_model(bundle: PlannerBundle):
    if not bundle.method.load_dynamics:
        return None
    return bundle.planner.trajopt_solver.core.metrics_rollout.transition_model.robot_dynamics


def get_torque_limits(planner: MotionPlanner, joint_names: Sequence[str]) -> torch.Tensor:
    # Match motion_plan_benchmark.py: Franka Panda effort limits for the seven arm joints.
    if len(joint_names) == len(PANDA_TORQUE_LIMITS_NM):
        return torch.as_tensor(
            PANDA_TORQUE_LIMITS_NM,
            device=planner.device_cfg.device,
            dtype=torch.float32,
        )
    joint_limits = planner.kinematics.kinematics_config.joint_limits
    effort = joint_limits.effort
    if effort is not None and joint_limits.joint_names is not None:
        indices = [joint_limits.joint_names.index(name) for name in joint_names]
        effort_limits = effort[:, indices].abs().max(dim=0).values
        return effort_limits.to(device=planner.device_cfg.device, dtype=torch.float32)
    raise ValueError(f"No torque limits available for joints: {joint_names}")


def slice_last_dim(value: Optional[torch.Tensor], indices: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.index_select(value, dim=-1, index=indices)


def align_trajectory_for_dynamics(
    dynamics_model,
    trajectory: JointState,
    target_joint_names: Optional[List[str]],
) -> JointState:
    target_dof = int(getattr(dynamics_model, "_n_dof", trajectory.position.shape[-1]))
    source_dof = int(trajectory.position.shape[-1])
    desired_joint_names = target_joint_names or []
    if source_dof == target_dof:
        if desired_joint_names and trajectory.joint_names == desired_joint_names:
            return trajectory
        if (
            desired_joint_names
            and trajectory.joint_names
            and all(name in trajectory.joint_names for name in desired_joint_names)
        ):
            return trajectory.reorder(desired_joint_names)
        if desired_joint_names:
            dynamics_trajectory = trajectory.clone()
            dynamics_trajectory.joint_names = desired_joint_names
            return dynamics_trajectory
        return trajectory

    if trajectory.joint_names:
        if len(desired_joint_names) != target_dof:
            desired_joint_names = [
                name for name in trajectory.joint_names if "finger" not in name
            ][:target_dof]
        available_targets = [
            name for name in desired_joint_names if name in trajectory.joint_names
        ]
        if len(available_targets) == target_dof:
            return trajectory.reorder(available_targets)

    if source_dof < target_dof:
        raise ValueError(
            f"Trajectory has {source_dof} joints, but dynamics model expects {target_dof}"
        )

    # Franka configs may include two gripper joints after the seven arm joints,
    # while the RNEA model only contains the arm. Keep the arm prefix as a
    # conservative fallback when joint names are unavailable.
    indices = torch.arange(target_dof, device=trajectory.device)
    joint_names = None
    if trajectory.joint_names:
        joint_names = [trajectory.joint_names[int(i)] for i in indices.detach().cpu()]
    return JointState(
        position=slice_last_dim(trajectory.position, indices),
        velocity=slice_last_dim(trajectory.velocity, indices),
        acceleration=slice_last_dim(trajectory.acceleration, indices),
        jerk=slice_last_dim(trajectory.jerk, indices),
        joint_names=joint_names,
        device_cfg=trajectory.device_cfg,
        dt=trajectory.dt,
        knot=trajectory.knot,
        knot_dt=trajectory.knot_dt,
        control_space=trajectory.control_space,
    )


def reshape_trajectory_to_bhd(trajectory: JointState) -> JointState:
    position = trajectory.position
    if position.ndim == 3:
        return trajectory
    if position.ndim == 1:
        shape = (1, 1, position.shape[-1])
    elif position.ndim == 2:
        shape = (1, position.shape[-2], position.shape[-1])
    else:
        shape = (-1, position.shape[-2], position.shape[-1])

    return JointState(
        position=trajectory.position.reshape(shape),
        velocity=trajectory.velocity.reshape(shape) if trajectory.velocity is not None else None,
        acceleration=(
            trajectory.acceleration.reshape(shape)
            if trajectory.acceleration is not None
            else None
        ),
        jerk=trajectory.jerk.reshape(shape) if trajectory.jerk is not None else None,
        joint_names=trajectory.joint_names,
        device_cfg=trajectory.device_cfg,
        dt=trajectory.dt,
        knot=trajectory.knot,
        knot_dt=trajectory.knot_dt,
        control_space=trajectory.control_space,
    )


def compute_dynamics_metrics(
    dynamics_model,
    trajectory: JointState,
    target_joint_names: Optional[List[str]],
    torque_limits: torch.Tensor,
) -> Dict[str, float]:
    """Compute trajectory dynamics metrics with motion_plan_benchmark.py semantics.

    This mirrors the original benchmark's ``compute_trajectory_energy``:
    energy is ``sum(abs(tau * qd)) * dt``, max torque is the maximum absolute
    torque over all joints/timesteps, and torque violation is checked per joint
    against Panda effort limits. The only difference is that this benchmark uses
    cuRobo's moving-base RNEA evaluator instead of fixed-base Pinocchio.
    """
    if dynamics_model is None or trajectory.velocity is None:
        return {
            "energy": 0.0,
            "max_torque": 0.0,
            "torque_violation": False,
            "torques": None,
            "power": None,
            "energy_j": float("nan"),
            "positive_energy_j": float("nan"),
            "max_abs_tau_nm": float("nan"),
            "rms_tau_nm": float("nan"),
            "mean_abs_power_w": float("nan"),
            "work_j": float("nan"),
            "peak_power_w": float("nan"),
            "max_tau_ratio": float("nan"),
            "max_abs_tau_per_joint_nm": [],
            "tau_limit_ratio_per_joint": [],
        }
    with torch.no_grad():
        dynamics_trajectory = align_trajectory_for_dynamics(
            dynamics_model, trajectory, target_joint_names
        )
        dynamics_trajectory = reshape_trajectory_to_bhd(dynamics_trajectory)
        tau = dynamics_model.compute_inverse_dynamics(dynamics_trajectory).detach()
        velocity = dynamics_trajectory.velocity.detach()
        torque_limits = torque_limits.to(device=tau.device, dtype=tau.dtype)
        power = tau * velocity
        dt = scalar(trajectory.dt, default=0.0)

        # Same core metrics as motion_plan_benchmark.compute_trajectory_energy().
        energy = torch.sum(torch.abs(power)) * dt
        max_abs_tau_per_joint = torch.amax(torch.abs(tau).reshape(-1, tau.shape[-1]), dim=0)
        max_abs_tau = torch.max(max_abs_tau_per_joint)
        torque_violation = bool(torch.any(max_abs_tau_per_joint > torque_limits))

        # Extra diagnostic fields derived from the same RNEA output.
        positive_energy = torch.sum(torch.clamp(power, min=0.0)) * dt
        work = torch.sum(power) * dt
        mean_abs_power = torch.mean(torch.abs(power))
        peak_power = torch.max(torch.abs(power))
        tau_limit_ratio = max_abs_tau_per_joint / torque_limits
        max_tau_ratio = torch.max(tau_limit_ratio)
        rms_tau = torch.sqrt(torch.mean(tau * tau))
    return {
        "energy": float(energy.item()),
        "max_torque": float(max_abs_tau.item()),
        "torque_violation": bool(torque_violation),
        "torques": tau,
        "power": power,
        "energy_j": float(energy.item()),
        "positive_energy_j": float(positive_energy.item()),
        "max_abs_tau_nm": float(max_abs_tau.item()),
        "rms_tau_nm": float(rms_tau.item()),
        "mean_abs_power_w": float(mean_abs_power.item()),
        "work_j": float(work.item()),
        "peak_power_w": float(peak_power.item()),
        "max_tau_ratio": float(max_tau_ratio.item()),
        "max_abs_tau_per_joint_nm": [
            float(value) for value in max_abs_tau_per_joint.detach().cpu().tolist()
        ],
        "tau_limit_ratio_per_joint": [
            float(value) for value in tau_limit_ratio.detach().cpu().tolist()
        ],
    }


def cspace_path_length(trajectory: JointState) -> float:
    position = trajectory.position
    if position.shape[-2] < 2:
        return 0.0
    diff = position[..., 1:, :] - position[..., :-1, :]
    return float(torch.sum(torch.linalg.norm(diff, dim=-1)).item())


def motion_time(trajectory: JointState, planner: MotionPlanner) -> float:
    dt = scalar(trajectory.dt, default=0.0)
    offset = 1
    if trajectory.control_space in ControlSpace.bspline_types():
        offset = 2 * planner.trajopt_solver.interpolation_steps
    return dt * max(0, int(trajectory.position.shape[-2]) - offset)


def max_abs_jerk(trajectory: JointState) -> float:
    if trajectory.jerk is None:
        return float("nan")
    return float(torch.max(torch.abs(trajectory.jerk)).item())


def quaternion_path_length(quaternion: torch.Tensor) -> float:
    if quaternion.shape[-2] < 2:
        return 0.0
    q0 = quaternion[..., :-1, :]
    q1 = quaternion[..., 1:, :]
    dot = torch.sum(q0 * q1, dim=-1).abs().clamp(max=1.0)
    return float(torch.sum(2.0 * torch.acos(dot)).item())


def align_trajectory_for_kinematics(planner: MotionPlanner, trajectory: JointState) -> JointState:
    target_joint_names = planner.joint_names
    target_dof = len(target_joint_names)
    source_dof = int(trajectory.position.shape[-1])

    if source_dof == target_dof:
        if trajectory.joint_names == target_joint_names:
            return trajectory
        if trajectory.joint_names and all(name in trajectory.joint_names for name in target_joint_names):
            return trajectory.reorder(target_joint_names)
        fk_trajectory = trajectory.clone()
        fk_trajectory.joint_names = target_joint_names
        return fk_trajectory

    if source_dof < target_dof:
        raise ValueError(
            f"Trajectory has {source_dof} joints, but kinematics expects {target_dof}"
        )

    if trajectory.joint_names and all(name in trajectory.joint_names for name in target_joint_names):
        return trajectory.reorder(target_joint_names)

    indices = torch.arange(target_dof, device=trajectory.device)
    return JointState(
        position=slice_last_dim(trajectory.position, indices),
        velocity=slice_last_dim(trajectory.velocity, indices),
        acceleration=slice_last_dim(trajectory.acceleration, indices),
        jerk=slice_last_dim(trajectory.jerk, indices),
        joint_names=target_joint_names,
        device_cfg=trajectory.device_cfg,
        dt=trajectory.dt,
        knot=trajectory.knot,
        knot_dt=trajectory.knot_dt,
        control_space=trajectory.control_space,
    )


def reshape_trajectory_for_kinematics(trajectory: JointState) -> JointState:
    return reshape_trajectory_to_bhd(trajectory)


def eef_path_lengths(planner: MotionPlanner, trajectory: JointState) -> Tuple[float, float]:
    try:
        fk_trajectory = align_trajectory_for_kinematics(planner, trajectory)
        fk_trajectory = reshape_trajectory_for_kinematics(fk_trajectory)
        fk_state = planner.compute_kinematics(fk_trajectory)
        tool_pose = fk_state.tool_poses.get_link_pose(planner.tool_frames[0])
        position = tool_pose.position.reshape(1, -1, 3)
        quaternion = tool_pose.quaternion.reshape(1, -1, 4)
        position_length = torch.sum(
            torch.linalg.norm(position[:, 1:, :] - position[:, :-1, :], dim=-1)
        )
        return float(position_length.item()), quaternion_path_length(quaternion)
    except Exception:
        return float("nan"), float("nan")


def make_goal(problem: Dict[str, Any], planner: MotionPlanner) -> Tuple[JointState, GoalToolPose]:
    q_start = problem["start"]
    pose = problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
    start_state = JointState.from_position(
        planner.device_cfg.to_device([q_start]),
        joint_names=planner.joint_names,
    )
    goal_pose = Pose.from_list(pose)
    goal_tool_poses = GoalToolPose.from_poses(
        {planner.tool_frames[0]: goal_pose},
        ordered_tool_frames=planner.tool_frames,
    )
    return start_state, goal_tool_poses


def run_one_plan(
    bundle: PlannerBundle,
    problem: Dict[str, Any],
    world,
    base_motion_eval_dynamics,
    base_stats: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    planner = bundle.planner
    planner.scene_collision_checker.clear_cache()
    planner.update_world(world)
    planner.reset_seed()

    start_state, goal_tool_poses = make_goal(problem, planner)
    cuda_sync()
    wall_start = time.perf_counter()
    result = planner.plan_pose(
        goal_tool_poses,
        start_state,
        max_attempts=args.max_attempts,
        enable_graph_attempt=args.enable_graph_attempt,
    )
    cuda_sync()
    wall_time = time.perf_counter() - wall_start

    row: Dict[str, Any] = {
        "method": bundle.method.name,
        "skip": 0,
        "success": 0,
        "collision": 0,
        "joint_limit_violation": 0,
        "self_collision": 0,
        "physical_violation": 0,
        "torque_violation": 0,
        "dynamics_success": 0,
        "payload_success": 0,
        "perception_success": 0,
        "perception_interpolated_success": 0,
        "eval_payload_mass_kg": args.mass,
        "wall_time_s": wall_time,
        "total_time_s": float("nan"),
        "solve_time_s": float("nan"),
        "time": float("nan"),
        "solve_time": float("nan"),
        "perception_time": 0.0,
        "position_error_mm": float("nan"),
        "orientation_error_deg": float("nan"),
        "position_error": float("nan"),
        "orientation_error": float("nan"),
        "motion_time_s": float("nan"),
        "motion_time": float("nan"),
        "attempts": 1,
        "trajectory_length": 1,
        "eef_position_path_length": float("nan"),
        "eef_orientation_path_length": float("nan"),
        "cspace_path_length_rad": float("nan"),
        "cspace_path_length": float("nan"),
        "max_abs_jerk": float("nan"),
        "jerk": float("nan"),
        "base_motion_eval_energy_j": float("nan"),
        "base_motion_eval_positive_energy_j": float("nan"),
        "base_motion_eval_max_abs_tau_nm": float("nan"),
        "base_motion_eval_rms_tau_nm": float("nan"),
        "base_motion_eval_mean_abs_power_w": float("nan"),
        "base_motion_eval_work_j": float("nan"),
        "base_motion_eval_peak_power_w": float("nan"),
        "base_motion_eval_max_tau_ratio": float("nan"),
        "moving_eval_energy_j": float("nan"),
        "moving_eval_positive_energy_j": float("nan"),
        "moving_eval_max_abs_tau_nm": float("nan"),
        "moving_eval_rms_tau_nm": float("nan"),
        "moving_eval_mean_abs_power_w": float("nan"),
        "moving_eval_work_j": float("nan"),
        "moving_eval_peak_power_w": float("nan"),
        "energy": float("nan"),
        "torque": float("nan"),
        "power": float("nan"),
        "work": float("nan"),
        "peak_power": float("nan"),
        "max_tau_ratio": float("nan"),
        "status": "failure",
    }
    for joint_name in planner.joint_names:
        row[f"max_abs_tau_{joint_name}_nm"] = float("nan")
        row[f"tau_limit_ratio_{joint_name}"] = float("nan")
    row.update(base_stats)

    if result is None:
        return row

    row["total_time_s"] = scalar(result.total_time)
    row["solve_time_s"] = scalar(result.solve_time)
    row["time"] = row["total_time_s"]
    row["solve_time"] = row["solve_time_s"]
    if not bool(result.success.item()):
        row["status"] = str(getattr(result, "status", "failure"))
        return row

    trajectory = result.js_solution
    torque_limits = get_torque_limits(planner, planner.joint_names)
    base_motion_metrics = compute_dynamics_metrics(
        base_motion_eval_dynamics, trajectory, planner.joint_names, torque_limits
    )
    torque_violation = int(base_motion_metrics["torque_violation"])
    dynamics_success = int(not torque_violation)
    position_error_mm = scalar(result.position_error) * 1000.0
    orientation_error_deg = scalar(result.rotation_error) * 180.0 / math.pi
    motion_time_s = motion_time(trajectory, planner)
    cspace_length = cspace_path_length(trajectory)
    jerk = max_abs_jerk(trajectory)
    eef_position_length, eef_orientation_length = eef_path_lengths(planner, trajectory)

    row.update(
        {
            "success": 1,
            "physical_violation": torque_violation,
            "torque_violation": torque_violation,
            "dynamics_success": dynamics_success,
            "payload_success": dynamics_success,
            "position_error_mm": position_error_mm,
            "orientation_error_deg": orientation_error_deg,
            "position_error": position_error_mm,
            "orientation_error": orientation_error_deg,
            "motion_time_s": motion_time_s,
            "motion_time": motion_time_s,
            "trajectory_length": int(trajectory.position.shape[-2]),
            "eef_position_path_length": eef_position_length,
            "eef_orientation_path_length": eef_orientation_length,
            "cspace_path_length_rad": cspace_length,
            "cspace_path_length": cspace_length,
            "max_abs_jerk": jerk,
            "jerk": jerk,
            "base_motion_eval_energy_j": base_motion_metrics["energy_j"],
            "base_motion_eval_positive_energy_j": base_motion_metrics["positive_energy_j"],
            "base_motion_eval_max_abs_tau_nm": base_motion_metrics["max_abs_tau_nm"],
            "base_motion_eval_rms_tau_nm": base_motion_metrics["rms_tau_nm"],
            "base_motion_eval_mean_abs_power_w": base_motion_metrics["mean_abs_power_w"],
            "base_motion_eval_work_j": base_motion_metrics["work_j"],
            "base_motion_eval_peak_power_w": base_motion_metrics["peak_power_w"],
            "base_motion_eval_max_tau_ratio": base_motion_metrics["max_tau_ratio"],
            "moving_eval_energy_j": base_motion_metrics["energy_j"],
            "moving_eval_positive_energy_j": base_motion_metrics["positive_energy_j"],
            "moving_eval_max_abs_tau_nm": base_motion_metrics["max_abs_tau_nm"],
            "moving_eval_rms_tau_nm": base_motion_metrics["rms_tau_nm"],
            "moving_eval_mean_abs_power_w": base_motion_metrics["mean_abs_power_w"],
            "moving_eval_work_j": base_motion_metrics["work_j"],
            "moving_eval_peak_power_w": base_motion_metrics["peak_power_w"],
            "energy": base_motion_metrics["energy_j"],
            "torque": base_motion_metrics["max_abs_tau_nm"],
            "power": base_motion_metrics["mean_abs_power_w"],
            "work": base_motion_metrics["work_j"],
            "peak_power": base_motion_metrics["peak_power_w"],
            "max_tau_ratio": base_motion_metrics["max_tau_ratio"],
            "status": "success",
        }
    )
    for joint_name, max_tau, tau_ratio in zip(
        planner.joint_names,
        base_motion_metrics["max_abs_tau_per_joint_nm"],
        base_motion_metrics["tau_limit_ratio_per_joint"],
    ):
        row[f"max_abs_tau_{joint_name}_nm"] = max_tau
        row[f"tau_limit_ratio_{joint_name}"] = tau_ratio
    return row


def finite_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def percentile_or_nan(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, percentile))


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows = []
    methods = sorted({row["method"] for row in rows})
    metric_keys = [
        "wall_time_s",
        "total_time_s",
        "solve_time_s",
        "time",
        "solve_time",
        "motion_time_s",
        "motion_time",
        "eef_position_path_length",
        "eef_orientation_path_length",
        "cspace_path_length_rad",
        "cspace_path_length",
        "max_abs_jerk",
        "jerk",
        "base_motion_eval_energy_j",
        "base_motion_eval_max_abs_tau_nm",
        "base_motion_eval_mean_abs_power_w",
        "base_motion_eval_work_j",
        "base_motion_eval_peak_power_w",
        "base_motion_eval_max_tau_ratio",
        "moving_eval_energy_j",
        "moving_eval_max_abs_tau_nm",
        "moving_eval_mean_abs_power_w",
        "moving_eval_work_j",
        "moving_eval_peak_power_w",
        "energy",
        "torque",
        "power",
        "work",
        "peak_power",
        "max_tau_ratio",
        "position_error_mm",
        "orientation_error_deg",
        "position_error",
        "orientation_error",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        success_rows = [row for row in method_rows if int(row["success"]) == 1]
        dynamics_success_rows = [
            row for row in method_rows if int(row.get("dynamics_success", 0)) == 1
        ]
        torque_violation_rows = [
            row for row in success_rows if int(row.get("torque_violation", 0)) == 1
        ]
        summary: Dict[str, Any] = {
            "method": method,
            "problems": len(method_rows),
            "successes": len(success_rows),
            "success_rate": len(success_rows) / max(1, len(method_rows)),
            "kinematic_successes": len(success_rows),
            "kinematic_success_rate": len(success_rows) / max(1, len(method_rows)),
            "dynamics_successes": len(dynamics_success_rows),
            "dynamics_success_rate": len(dynamics_success_rows) / max(1, len(method_rows)),
            "payload_successes": len(dynamics_success_rows),
            "payload_success_rate": len(dynamics_success_rows) / max(1, len(method_rows)),
            "torque_violations": len(torque_violation_rows),
            "torque_violation_rate": len(torque_violation_rows) / max(1, len(success_rows)),
        }
        for key in metric_keys:
            values = finite_values(success_rows, key)
            summary[f"{key}_mean"] = mean_or_nan(values)
            summary[f"{key}_p75"] = percentile_or_nan(values, 75.0)
            summary[f"{key}_p98"] = percentile_or_nan(values, 98.0)
        summary_rows.append(summary)
    return summary_rows


def finite_or_inf(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if math.isfinite(number) else float("inf")


def row_to_curobo_metrics(row: Dict[str, Any]) -> CuroboMetrics:
    return CuroboMetrics(
        skip=bool(int(row.get("skip", 0))),
        success=bool(int(row.get("success", 0))),
        collision=bool(int(row.get("collision", 0))),
        joint_limit_violation=bool(int(row.get("joint_limit_violation", 0))),
        self_collision=bool(int(row.get("self_collision", 0))),
        physical_violation=bool(int(row.get("physical_violation", 0))),
        payload_success=bool(int(row.get("dynamics_success", row.get("payload_success", 0)))),
        perception_success=bool(int(row.get("perception_success", 0))),
        perception_interpolated_success=bool(
            int(row.get("perception_interpolated_success", 0))
        ),
        position_error=finite_or_inf(row.get("position_error")),
        orientation_error=finite_or_inf(row.get("orientation_error")),
        eef_position_path_length=finite_or_inf(row.get("eef_position_path_length")),
        eef_orientation_path_length=finite_or_inf(row.get("eef_orientation_path_length")),
        cspace_path_length=finite_or_inf(row.get("cspace_path_length")),
        trajectory_length=max(1, int(row.get("trajectory_length", 1))),
        attempts=int(row.get("attempts", 1)),
        motion_time=finite_or_inf(row.get("motion_time")),
        solve_time=finite_or_inf(row.get("solve_time")),
        time=finite_or_inf(row.get("time")),
        perception_time=finite_or_inf(row.get("perception_time", 0.0)),
        jerk=finite_or_inf(row.get("jerk")),
        energy=finite_or_inf(row.get("energy")),
        torque=finite_or_inf(row.get("torque")),
        power=finite_or_inf(row.get("power")),
        work=finite_or_inf(row.get("work")),
        peak_power=finite_or_inf(row.get("peak_power")),
    )


def curobo_group_metrics(rows: List[Dict[str, Any]]) -> CuroboGroupMetrics:
    return CuroboGroupMetrics.from_list([row_to_curobo_metrics(row) for row in rows])


def print_group_line(group_name: str, method_name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    g_m = curobo_group_metrics(rows)
    dynamics_success_rate = (
        100.0
        * sum(int(row.get("dynamics_success", 0)) for row in rows)
        / max(1, len(rows))
    )
    print(
        group_name,
        method_name,
        f"{g_m.success:2.2f}",
        f"{dynamics_success_rate:2.2f}",
        f"{g_m.time.mean:2.2f}",
        f"{g_m.time.percent_98:2.2f}",
        f"{g_m.position_error.percent_98:2.4f}",
        f"{g_m.orientation_error.percent_98:2.4f}",
        f"{g_m.cspace_path_length.percent_98:2.2f}",
        f"{g_m.motion_time.percent_98:2.2f}",
    )
    print(g_m.attempts)


def statistic_yaml(statistic) -> Dict[str, float]:
    return {
        "mean": float(statistic.mean),
        "std": float(statistic.std),
        "median": float(statistic.median),
        "75th": float(statistic.percent_75),
        "98th": float(statistic.percent_98),
    }


def benchmark_table_data(g_m: CuroboGroupMetrics) -> Dict[str, Any]:
    return {
        "Kinematic Success": float(g_m.success),
        "Dynamics Success": float(g_m.payload_success),
        "Physical Violation": float(g_m.physical_violation_rate),
        "Planning Time": statistic_yaml(g_m.time),
        "Position Error (mm)": statistic_yaml(g_m.position_error),
        "Orientation Error (deg)": statistic_yaml(g_m.orientation_error),
        "Path Length (rad.)": statistic_yaml(g_m.cspace_path_length),
        "Motion Time(s)": statistic_yaml(g_m.motion_time),
        "Jerk": statistic_yaml(g_m.jerk),
        "Energy (J)": statistic_yaml(g_m.energy),
        "Torque (N·m)": statistic_yaml(g_m.torque),
        "Solve Time (s)": statistic_yaml(g_m.solve_time),
    }


def print_motion_plan_style_summary(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    method_names = [method.name for method in METHODS]
    all_tables: Dict[str, Any] = {}

    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    for method_name in method_names:
        method_rows = [row for row in rows if row["method"] == method_name]
        if not method_rows:
            continue
        g_m = curobo_group_metrics(method_rows)
        all_tables[method_name] = benchmark_table_data(g_m)

        if args.kpi:
            continue
        if tabulate is not None:
            table = [
                ["Kinematic Success %", f"{g_m.success:2.2f}"],
                ["Dynamics Success %", f"{g_m.payload_success:2.2f}"],
                ["Physical Violation %", f"{g_m.physical_violation_rate:2.2f}"],
                ["Plan Time (s)", g_m.time],
                ["Solve Time (s)", g_m.solve_time],
                ["Position Error (mm)", g_m.position_error],
                ["Path Length (rad.)", g_m.cspace_path_length],
                ["Motion Time(s)", g_m.motion_time],
                ["Jerk", g_m.jerk],
                ["Energy (J)", g_m.energy],
                ["Torque (N·m)", g_m.torque],
            ]
            print(method_name)
            print(tabulate(table, ["Metric", "Value"], tablefmt="grid"))
        else:
            print("######## FULL SET ############")
            print(method_name, f"{g_m.success:2.2f}", f"{g_m.payload_success:2.2f}")
            print("MT: ", g_m.motion_time)
            print("path-length: ", g_m.cspace_path_length)
            print("PT:", g_m.time)
            print("ST: ", g_m.solve_time)
            print("position error (mm): ", g_m.position_error)
            print("orientation error(%): ", g_m.orientation_error)
            print("jerk: ", g_m.jerk)

    if args.write_benchmark:
        out_path = join_path("benchmark/log", "table_" + args.file_name + ".yml")
        print(out_path)
        write_yaml(all_tables, out_path)

    if args.kpi:
        kpi_data = {}
        for method_name in method_names:
            method_rows = [row for row in rows if row["method"] == method_name]
            if not method_rows:
                continue
            g_m = curobo_group_metrics(method_rows)
            kpi_data[method_name] = {
                "Kinematic Success": g_m.success,
                "Dynamics Success": g_m.payload_success,
                "Planning Time": float(g_m.time.mean),
                "Planning Time Std": float(g_m.time.std),
                "Planning Time Median": float(g_m.time.median),
                "Planning Time 75th": float(g_m.time.percent_75),
                "Planning Time 98th": float(g_m.time.percent_98),
            }
        write_yaml(kpi_data, join_path(args.save_path, args.file_name + ".yml"))


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


def print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    for row in summary_rows:
        print(
            "{method}: success_rate={success_rate:.3f}, "
            "mean_wall={wall_time_s_mean:.4f}s, "
            "moving_energy={moving_eval_energy_j_mean:.3f}J, "
            "moving_max_tau={moving_eval_max_abs_tau_nm_mean:.3f}Nm".format(**row)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cuRobo planning with no dynamics, fixed-base dynamics, and random base motion dynamics.",
    )
    parser.add_argument(
        "--dataset",
        choices=["motion_benchmaker", "mpinets", "demo", "all"],
        default="motion_benchmaker",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="When True, runs only on small dataaset",
        default=False,
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("benchmark/log/base_motion_plan"))
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="path to save KPI file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="base_motion_plan",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--kpi",
        action="store_true",
        help="When True, saves minimal metrics",
        default=False,
    )
    parser.add_argument(
        "--write_benchmark",
        action="store_true",
        help="When True, writes benchmark summary table YAML",
        default=False,
    )
    parser.add_argument("--max-groups", type=int, default=0, help="0 means all groups.")
    parser.add_argument("--max-problems-per-group", type=int, default=0, help="0 means all problems.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--ik-seeds", type=int, default=32)
    parser.add_argument("--trajopt-seeds", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--enable-graph-attempt", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--collision-activation-distance", type=float, default=0.0025)
    parser.add_argument("--mass", type=float, default=3.0)
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--disable-cuda-graph", "--disable_cuda_graph", action="store_true")
    parser.add_argument("--base-motion-dt", type=float, default=None)
    parser.add_argument("--base-angle-amp-deg", type=float, default=15.0)
    parser.add_argument("--base-yaw-scale", type=float, default=0.5)
    parser.add_argument("--base-linear-amp-m", type=float, default=0.2)
    parser.add_argument("--base-freq-min", type=float, default=0.30)
    parser.add_argument("--base-freq-max", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.demo:
        args.dataset = "demo"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires CUDA. Activate the pink environment in a CUDA-enabled session."
        )
    if args.base_freq_min <= 0.0 or args.base_freq_max < args.base_freq_min:
        raise ValueError("Invalid base motion frequency range")

    setup_curobo_logger("error")
    set_seed(args.seed)
    if not args.kpi:
        print("*****RUN: 0")

    all_rows: List[Dict[str, Any]] = []
    datasets = get_datasets(args.dataset)

    for dataset_index, (dataset_name, problems) in enumerate(datasets):
        mpinets = is_mpinets_dataset(problems)
        group_items = list(problems.items())
        if args.max_groups > 0:
            group_items = group_items[: args.max_groups]

        for group_index, (group_name, scene_problems) in enumerate(tqdm(group_items)):
            n_obstacles = check_problems(scene_problems, mesh=args.mesh)
            bundles: Dict[str, PlannerBundle] = {}
            group_rows: List[Dict[str, Any]] = []
            try:
                for method in METHODS:
                    set_seed(args.seed)
                    bundle, _ = make_planner_bundle(method, n_obstacles, mpinets, args)
                    bundles[method.name] = bundle

                base_bundle = bundles["base_motion_dynamics"]
                base_seed = args.seed + 1000000 * dataset_index + 10000 * group_index
                base_velocity, base_acceleration, base_stats = make_random_base_motion(
                    base_bundle.base_horizon,
                    base_bundle.base_dt,
                    base_seed,
                    args,
                )
                update_base_motion_buffer(base_bundle, base_velocity, base_acceleration)

                # Warm up after the moving-base buffers are populated. cuRobo's
                # dynamics expansion cache and CUDA graphs can then reuse stable
                # base-motion tensor pointers for the entire scene group.
                for method in METHODS:
                    bundle = bundles[method.name]
                    bundle.planner.warmup(
                        enable_graph=not args.disable_cuda_graph,
                        num_warmup_iterations=args.warmup_iters,
                    )

                base_motion_eval_dynamics = get_dynamics_model(bundles["base_motion_dynamics"])

                solved_in_group = 0
                for problem_index, problem in enumerate(tqdm(scene_problems, leave=False)):
                    if problem["collision_buffer_ik"] < 0.0:
                        continue
                    if args.max_problems_per_group > 0 and solved_in_group >= args.max_problems_per_group:
                        break
                    solved_in_group += 1

                    for method in METHODS:
                        bundle = bundles[method.name]
                        world = build_world(problem, mesh=args.mesh)
                        row = run_one_plan(
                            bundle,
                            problem,
                            world,
                            base_motion_eval_dynamics,
                            base_stats,
                            args,
                        )
                        row.update(
                            {
                                "dataset": dataset_name,
                                "group": group_name,
                                "group_index": group_index,
                                "problem_index": problem_index,
                                "base_motion_dt": base_bundle.base_dt,
                                "base_motion_horizon": base_bundle.base_horizon,
                            }
                        )
                        all_rows.append(row)
                        group_rows.append(row)

                if not args.kpi:
                    for method in METHODS:
                        method_group_rows = [
                            row for row in group_rows if row["method"] == method.name
                        ]
                        print_group_line(group_name, method.name, method_group_rows)
            finally:
                for bundle in bundles.values():
                    bundle.planner.destroy()

    summary_rows = summarize_rows(all_rows)
    trial_path = args.output_prefix.with_name(args.output_prefix.name + "_trials.csv")
    summary_csv_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary_yaml_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.yml")
    write_csv(trial_path, all_rows)
    write_csv(summary_csv_path, summary_rows)
    write_yaml(summary_rows, str(summary_yaml_path))
    print_motion_plan_style_summary(all_rows, args)
    print(f"Wrote trials to: {trial_path}")
    print(f"Wrote summary to: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
