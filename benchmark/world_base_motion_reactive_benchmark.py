#!/usr/bin/env python3
"""World-frame moving-base obstacle benchmark.

Goals and obstacles stay fixed in the world frame. At every control step the
current base pose transforms them into the Panda base frame, then either cuRobo
reactive MPC or the pure-Python Crocoddyl/floating_mpc OCP solves one step.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import torch
import yaml

from curobo_floating_base._src.util.logging import setup_curobo_logger
from curobo_floating_base._src.util_file import get_robot_configs_path, join_path, load_yaml, write_yaml
from curobo_floating_base._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo_floating_base.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo_floating_base.types import DeviceCfg, GoalToolPose, JointState, Pose


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
PANDA_NMPC_SCRIPTS = WORKSPACE_ROOT / "src" / "panda_nmpc" / "scripts"
DEFAULT_CONFIG = SCRIPT_DIR / "config" / "world_base_motion_reactive.yaml"
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
CUROBO_INNER_ITERS = 25


def ns(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def load_config(path: Path) -> SimpleNamespace:
    data = yaml.safe_load(path.read_text()) or {}
    data["config_path"] = str(path.resolve())
    data["output_prefix"] = str(resolve_path(path.parent, data["output_prefix"]))
    data["mpc"]["config"] = str(resolve_path(path.parent, data["mpc"]["config"]))
    data["base_motion"] = ns(data["base_motion"])
    data["curobo"] = ns(data["curobo"])
    data["mpc"] = ns(data["mpc"])
    data["simulation"] = ns(data["simulation"])
    data["methods"] = list(data["methods"])
    data["scenes"] = list(data["scenes"])
    return ns(data)


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_mpc_module():
    if str(PANDA_NMPC_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PANDA_NMPC_SCRIPTS))
    import base_frame_numeric_sim_main as mpc_module

    return mpc_module


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return np.array([q[3], q[0], q[1], q[2]])


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return np.array([q[1], q[2], q[3], q[0]])


def pose_wxyz_to_xyzw(pose: Any) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return np.r_[pose[:3], wxyz_to_xyzw(pose[3:7])]


def pose_xyzw_to_wxyz(pose: Any) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return np.r_[pose[:3], xyzw_to_wxyz(pose[3:7])]


def pose_matrix(pose_xyzw: Any, mpc_module) -> np.ndarray:
    pose = np.asarray(pose_xyzw, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = mpc_module.quat_xyzw_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def matrix_pose(transform: np.ndarray, mpc_module) -> np.ndarray:
    return np.r_[transform[:3, 3], mpc_module.matrix_to_quat_xyzw(transform[:3, :3])]


def world_pose_to_base_wxyz(world_pose_wxyz: Any, base_pose_xyzw: Any, mpc_module) -> np.ndarray:
    world_t_base = pose_matrix(base_pose_xyzw, mpc_module)
    world_t_object = pose_matrix(pose_wxyz_to_xyzw(world_pose_wxyz), mpc_module)
    return pose_xyzw_to_wxyz(matrix_pose(np.linalg.inv(world_t_base) @ world_t_object, mpc_module))


def world_obstacles_to_base(obstacles_world: dict[str, Any], base_pose_xyzw: Any, mpc_module) -> dict[str, Any]:
    obstacles_base = deepcopy(obstacles_world)
    for obstacles in obstacles_base.values():
        for obstacle in obstacles.values():
            obstacle["pose"] = world_pose_to_base_wxyz(obstacle["pose"], base_pose_xyzw, mpc_module).tolist()
    return obstacles_base


def sample_base_motion(cfg: SimpleNamespace, t: float, mpc_module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion = cfg.base_motion
    if motion.frequency_hz <= 0.0:
        return np.r_[np.zeros(3), 0.0, 0.0, 0.0, 1.0], np.zeros(6), np.zeros(6)

    omega = 2.0 * math.pi * motion.frequency_hz
    phase = np.deg2rad(np.asarray(motion.phase_deg, dtype=float))
    xyz_amp = np.asarray(motion.translation_amp_m, dtype=float)
    rpy_amp = np.deg2rad(np.asarray(motion.rotation_amp_deg, dtype=float))

    s = np.sin(omega * t + phase)
    sd = omega * np.cos(omega * t + phase)
    sdd = -omega * omega * np.sin(omega * t + phase)

    xyz = xyz_amp * s[:3]
    xyz_d = xyz_amp * sd[:3]
    xyz_dd = xyz_amp * sdd[:3]
    rpy = rpy_amp * s[3:]
    rpy_d = rpy_amp * sd[3:]
    rpy_dd = rpy_amp * sdd[3:]

    rotation = mpc_module.euler_xyz_to_matrix(*rpy)
    linear_vel = rotation.T @ xyz_d
    angular_vel = rpy_d
    linear_acc = rotation.T @ xyz_dd - np.cross(angular_vel, linear_vel)
    angular_acc = rpy_dd
    pose = np.r_[xyz, mpc_module.matrix_to_quat_xyzw(rotation)]
    twist = np.r_[linear_vel, angular_vel]
    accel = np.r_[linear_acc, angular_acc]
    return pose, twist, accel


def base_prediction(cfg: SimpleNamespace, start_t: float, horizon: int, dt: float, mpc_module):
    poses, twists, accels = [], [], []
    for k in range(horizon):
        pose, twist, accel = sample_base_motion(cfg, start_t + k * dt, mpc_module)
        poses.append(pose)
        twists.append(twist)
        accels.append(accel)
    return np.asarray(poses), np.asarray(twists), np.asarray(accels)


def curobo_base_motion(base_pose: np.ndarray, base_twist: np.ndarray, base_accel: np.ndarray, mpc_module):
    gravity = np.array([0.0, 0.0, -9.81])
    vel = np.c_[base_twist[:, 3:6], base_twist[:, 0:3]].astype(np.float32)
    acc = np.zeros_like(vel)
    acc[:, :3] = base_accel[:, 3:6]
    for k, pose in enumerate(base_pose):
        rot = mpc_module.quat_xyzw_to_matrix(pose[3:7])
        gravity_base = -(rot.T @ gravity)
        acc[k, 3:6] = base_accel[k, :3] + gravity_base + gravity
    return vel, acc.astype(np.float32)


def time_sample_indices(target_dt: float, sample_dt: float, length: int, first_sample_time: float = 0.0) -> tuple[int, int, float]:
    if target_dt <= first_sample_time or length <= 1:
        return 0, 0, 0.0
    position = (target_dt - first_sample_time) / sample_dt
    lo = int(math.floor(position))
    if lo >= length - 1:
        return length - 1, length - 1, 0.0
    hi = lo + 1
    return lo, hi, float(position - lo)


def sample_numpy_sequence(sequence: list[np.ndarray], target_dt: float, sample_dt: float, first_sample_time: float = 0.0) -> np.ndarray:
    lo, hi, alpha = time_sample_indices(target_dt, sample_dt, len(sequence), first_sample_time)
    if lo == hi:
        return np.asarray(sequence[lo]).copy()
    return (1.0 - alpha) * np.asarray(sequence[lo]) + alpha * np.asarray(sequence[hi])


def sample_torch_sequence(sequence: Any, target_dt: float, sample_dt: float, first_sample_time: float) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    length = int(sequence.position.shape[1])
    lo, hi, alpha = time_sample_indices(target_dt, sample_dt, length, first_sample_time)
    q = sequence.position[:, lo, :] if lo == hi else (1.0 - alpha) * sequence.position[:, lo, :] + alpha * sequence.position[:, hi, :]
    dq = sequence.velocity[:, lo, :] if lo == hi else (1.0 - alpha) * sequence.velocity[:, lo, :] + alpha * sequence.velocity[:, hi, :]
    return q.detach().cpu().numpy().reshape(-1)[:7], dq.detach().cpu().numpy().reshape(-1)[:7], lo, hi, alpha


def robot_cfg(tool_frame: str) -> dict[str, Any]:
    data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    data = deepcopy(data.get("robot_cfg", data))
    kin = data["kinematics"]
    kin["tool_frames"] = [tool_frame]
    kin["lock_joints"] = {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
    if "attached_object" in kin.get("collision_link_names", []):
        kin["collision_link_names"].remove("attached_object")
    data["load_dynamics"] = True
    return data


def joint_state(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, names: list[str], device_cfg) -> JointState:
    state = JointState.from_position(
        torch.as_tensor(q, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1),
        joint_names=names,
    )
    state.velocity = torch.as_tensor(dq, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1)
    state.acceleration = torch.as_tensor(ddq, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1)
    return state


def goal_pose(goal_base_wxyz: np.ndarray, frames: list[str]) -> GoalToolPose:
    return GoalToolPose.from_poses(
        {frames[0]: Pose.from_list(goal_base_wxyz.tolist())},
        ordered_tool_frames=frames,
        num_goalset=1,
    )


def scene_cache(obstacles: dict[str, Any]) -> dict[str, int]:
    return {"obb": max(1, len(obstacles.get("cuboid", {})))}


def make_curobo(scene: dict[str, Any], cfg: SimpleNamespace, mpc_module):
    base_pose0, _, _ = sample_base_motion(cfg, 0.0, mpc_module)
    obstacles_base = world_obstacles_to_base(scene["obstacles_world"], base_pose0, mpc_module)
    cu = cfg.curobo
    controller_cfg = ModelPredictiveControlCfg.create(
        robot=robot_cfg(cfg.tool_frame),
        scene_model=obstacles_base,
        collision_cache=scene_cache(obstacles_base),
        use_cuda_graph=not cu.disable_cuda_graph,
        optimization_dt=cu.dt,
        interpolation_steps=cu.interpolation_steps,
        optimizer_collision_activation_distance=cu.collision_activation_distance,
        non_terminal_tool_pose_weight_factor=cu.non_terminal_tool_pose_weight_factor,
        position_tolerance=cu.position_tolerance,
        orientation_tolerance=cu.orientation_tolerance,
        warm_start_optimization_num_iters=cu.warm_iters,
        cold_start_optimization_num_iters=cu.cold_iters,
        device_cfg=DeviceCfg(),
    )
    controller = ModelPredictiveControl(controller_cfg)
    if not cu.track_orientation:
        controller.update_tool_pose_criteria(
            {frame: ToolPoseCriteria.track_position() for frame in controller.tool_frames}
        )
    controller.update_links_inertial({"attached_object": {"mass": cfg.mass}})

    horizon = max(2, int(controller.action_horizon))
    base_pose, base_twist, base_accel = base_prediction(cfg, 0.0, horizon, cu.dt, mpc_module)
    base_vel, base_acc = curobo_base_motion(base_pose, base_twist, base_accel, mpc_module)
    base_vel_t = torch.as_tensor(base_vel, device=controller.device_cfg.device)
    base_acc_t = torch.as_tensor(base_acc, device=controller.device_cfg.device)
    controller.core.set_base_motion(base_vel_t, base_acc_t)

    q0 = np.asarray(scene["start_q"], dtype=float)
    controller.setup(joint_state(q0, np.zeros(7), np.zeros(7), controller.joint_names, controller.device_cfg))
    goal_base = world_pose_to_base_wxyz(scene["goal_world_pose_wxyz"], base_pose0, mpc_module)
    controller.update_goal_tool_poses(goal_pose(goal_base, controller.tool_frames), run_ik=cu.run_ik, use_best_effort_ik=True)
    return controller, base_vel_t, base_acc_t


def curobo_step(state: dict[str, np.ndarray], controller, base_vel, base_acc, scene, cfg, t, mpc_module):
    base_pose0, _, _ = sample_base_motion(cfg, t, mpc_module)
    obstacles_base = world_obstacles_to_base(scene["obstacles_world"], base_pose0, mpc_module)
    for obstacles in obstacles_base.values():
        for name, obstacle in obstacles.items():
            controller.scene_collision_checker.update_obstacle_pose(name, Pose.from_list(obstacle["pose"]))
    goal_base = world_pose_to_base_wxyz(scene["goal_world_pose_wxyz"], base_pose0, mpc_module)
    controller.update_goal_tool_poses(goal_pose(goal_base, controller.tool_frames), run_ik=cfg.curobo.run_ik, use_best_effort_ik=True)

    base_pose, base_twist, base_accel = base_prediction(cfg, t, base_vel.shape[0], cfg.curobo.dt, mpc_module)
    vel_np, acc_np = curobo_base_motion(base_pose, base_twist, base_accel, mpc_module)
    base_vel.copy_(torch.as_tensor(vel_np, device=base_vel.device))
    base_acc.copy_(torch.as_tensor(acc_np, device=base_acc.device))

    ddq = (state["dq"] - state["prev_dq"]) / cfg.simulation.dt
    current = joint_state(state["q"], state["dq"], ddq, controller.joint_names, controller.device_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = controller.optimize_action_sequence(current)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    solve_time = time.perf_counter() - start

    seq = result.action_sequence
    if seq is None or seq.position.shape[1] == 0:
        return solve_time, {"status": "no_action_sequence"}
    command_dt = cfg.curobo.dt / cfg.curobo.interpolation_steps
    q_next, dq_next, lo, hi, alpha = sample_torch_sequence(seq, cfg.simulation.dt, command_dt, command_dt)
    state["prev_dq"] = state["dq"].copy()
    state["q"] = q_next
    state["dq"] = dq_next
    return solve_time, {
        "trajectory_length": int(seq.position.shape[1]),
        "command_index": lo,
        "command_index_hi": hi,
        "command_alpha": alpha,
        "command_time_s": cfg.simulation.dt,
        "ddq": (dq_next - state["prev_dq"]) / cfg.simulation.dt,
    }


def mpc_step(state: dict[str, np.ndarray], scene: dict[str, Any], cfg: SimpleNamespace, t: float, mpc_module):
    mcfg = mpc_module.load_config(Path(cfg.mpc.config))
    mcfg.planner.T = cfg.mpc.horizon
    mcfg.planner.dt_ocp = cfg.mpc.dt
    mcfg.planner.nb_iterations_max = cfg.mpc.iterations
    mcfg.planner.max_qp_iter = cfg.mpc.max_qp_iter
    mcfg.planner.ee_frame_name = cfg.tool_frame
    mcfg.planner.collision_safety_margin = cfg.mpc.collision_safety_margin

    base_pose, base_twist, base_accel = base_prediction(cfg, t, mcfg.planner.T + 1, mcfg.planner.dt_ocp, mpc_module)
    obstacles_base = world_obstacles_to_base(scene["obstacles_world"], base_pose[0], mpc_module)
    mcfg.simulation.target_pose_in_base = pose_wxyz_to_xyzw(
        world_pose_to_base_wxyz(scene["goal_world_pose_wxyz"], base_pose[0], mpc_module)
    )

    floating_model = mpc_module.load_floating_panda_model(mcfg.robot_model.urdf_path)
    collision_model = mpc_module.build_collision_model_with_obstacles(
        floating_model,
        mcfg.robot_model.urdf_path,
        mcfg.robot_model.package_dirs,
        obstacles_base,
        cfg.mpc.collision_links,
        cfg.mpc.ignore_unsupported_obstacles,
    )
    planner = mpc_module.BaseFrameReachingPy(floating_model, collision_model, mcfg.planner)
    x0 = np.r_[state["q"], state["dq"]]
    planner.ocp.problem.x0 = x0
    planner.set_base_motion_prediction(list(base_pose), list(base_twist), list(base_accel))
    planner.set_ee_ref_base_placement_list_constant_weights(mcfg.simulation.target_pose_in_base, np.zeros(6), True, 1.0)
    posture = np.zeros(14)
    posture[:7] = state["q"]
    planner.set_posture_ref(posture)

    xs = state.get("xs")
    us = state.get("us")
    if xs is not None and us is not None and len(xs) == len(us) + 1:
        xs_init = [x.copy() for x in xs[1:]] + [xs[-1].copy()]
        us_init = [u.copy() for u in us[1:]] + [us[-1].copy()]
        xs_init[0] = x0.copy()
    else:
        xs_init = [x0.copy() for _ in range(mcfg.planner.T + 1)]
        us_init = [
            mpc_module.compute_floating_inverse_dynamics(
                floating_model, state["q"], state["dq"], np.zeros(7), base_pose[i], base_twist[i], base_accel[i]
            )
            for i in range(mcfg.planner.T)
        ]

    start = time.perf_counter()
    planner.solve(xs_init, us_init)
    solve_time = time.perf_counter() - start
    state["xs"] = [np.asarray(x).copy() for x in planner.ocp.xs]
    state["us"] = [np.asarray(u).copy() for u in planner.ocp.us]

    x_next = sample_numpy_sequence(state["xs"], cfg.simulation.dt, cfg.mpc.dt)
    q_next = x_next[:7].copy()
    dq_next = x_next[7:].copy()
    ddq = (dq_next - state["dq"]) / cfg.simulation.dt
    tau = mpc_module.compute_floating_inverse_dynamics(floating_model, state["q"], state["dq"], ddq, base_pose[0], base_twist[0], base_accel[0])
    min_dist = mpc_module.min_collision_distance(floating_model, collision_model, state["xs"])
    state["prev_dq"] = state["dq"].copy()
    state["q"] = q_next
    state["dq"] = dq_next
    lo, hi, alpha = time_sample_indices(cfg.simulation.dt, cfg.mpc.dt, len(state["xs"]))
    return solve_time, {
        "tau": tau,
        "ddq": ddq,
        "min_collision_distance_m": min_dist,
        "trajectory_length": len(state["xs"]),
        "command_index": lo,
        "command_index_hi": hi,
        "command_alpha": alpha,
        "command_time_s": cfg.simulation.dt,
    }


def evaluate(method: str, state: dict[str, np.ndarray], info: dict[str, Any], scene: dict[str, Any], cfg: SimpleNamespace, t: float, solve_time: float, mpc_module):
    mcfg = mpc_module.load_config(Path(cfg.mpc.config))
    fixed_model = mpc_module.load_panda_model(mcfg.robot_model.urdf_path)
    floating_model = mpc_module.load_floating_panda_model(mcfg.robot_model.urdf_path)
    base_pose, base_twist, base_accel = sample_base_motion(cfg, t, mpc_module)
    ee_base = mpc_module.compute_fixed_fk_pose_xyzw(fixed_model, fixed_model.getFrameId(cfg.tool_frame), state["q"], state["dq"])
    ee_world = matrix_pose(pose_matrix(base_pose, mpc_module) @ pose_matrix(ee_base, mpc_module), mpc_module)
    pos_error = float(np.linalg.norm(ee_world[:3] - pose_wxyz_to_xyzw(scene["goal_world_pose_wxyz"])[:3]))

    obstacles_base = world_obstacles_to_base(scene["obstacles_world"], base_pose, mpc_module)
    collision_model = mpc_module.build_collision_model_with_obstacles(
        floating_model, mcfg.robot_model.urdf_path, mcfg.robot_model.package_dirs, obstacles_base,
        cfg.mpc.collision_links, cfg.mpc.ignore_unsupported_obstacles,
    )
    x = np.r_[state["q"], state["dq"]]
    min_dist = min(mpc_module.min_collision_distance(floating_model, collision_model, [x]), info.get("min_collision_distance_m", float("inf")))
    ddq = info.get("ddq", (state["dq"] - state["prev_dq"]) / cfg.simulation.dt)
    tau = info.get("tau", mpc_module.compute_floating_inverse_dynamics(floating_model, state["q"], state["dq"], ddq, base_pose, base_twist, base_accel))
    return {
        "method": method,
        "scene": scene["name"],
        "time_s": t,
        "solve_time_s": solve_time,
        "position_error_m": pos_error,
        "goal_reached": int(pos_error <= scene["goal_tolerance_m"]),
        "min_collision_distance_m": float(min_dist),
        "collision": int(min_dist < cfg.mpc.collision_safety_margin),
        "max_abs_tau_nm": float(np.max(np.abs(tau))),
        "rms_tau_nm": float(np.sqrt(np.mean(tau * tau))),
        "mean_abs_power_w": float(np.mean(np.abs(tau * state["dq"]))),
        "torque_violation": int(np.any(np.abs(tau) > TORQUE_LIMITS)),
        "trajectory_length": int(info.get("trajectory_length", 1)),
        "command_index": int(info.get("command_index", 0)),
        "command_index_hi": int(info.get("command_index_hi", 0)),
        "command_alpha": float(info.get("command_alpha", 0.0)),
        "command_time_s": float(info.get("command_time_s", 0.0)),
        "status": info.get("status", "ok"),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scene in sorted({r["scene"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["scene"] == scene}):
            rs = [r for r in rows if r["scene"] == scene and r["method"] == method]
            solve = [r["solve_time_s"] for r in rs if math.isfinite(r["solve_time_s"])]
            out.append({
                "scene": scene,
                "method": method,
                "steps": len(rs),
                "goal_reached": int(any(r["goal_reached"] for r in rs)),
                "final_position_error_m": rs[-1]["position_error_m"],
                "min_collision_distance_m": min(r["min_collision_distance_m"] for r in rs),
                "collision_count": sum(r["collision"] for r in rs),
                "torque_violation_count": sum(r["torque_violation"] for r in rs),
                "mean_solve_time_s": float(np.mean(solve)) if solve else float("nan"),
                "max_abs_tau_nm": max(r["max_abs_tau_nm"] for r in rs),
            })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def align_curobo_iters(cfg: SimpleNamespace) -> None:
    for name in ("warm_iters", "cold_iters"):
        value = int(getattr(cfg.curobo, name))
        rem = value % CUROBO_INNER_ITERS
        if rem:
            setattr(cfg.curobo, name, value + CUROBO_INNER_ITERS - rem)


def run_scene(scene: dict[str, Any], cfg: SimpleNamespace, mpc_module) -> list[dict[str, Any]]:
    controllers = {}
    if "curobo" in cfg.methods:
        controller, base_vel, base_acc = make_curobo(scene, cfg, mpc_module)
        controllers["curobo"] = (controller, base_vel, base_acc)

    states = {m: {"q": np.asarray(scene["start_q"], dtype=float), "dq": np.zeros(7), "prev_dq": np.zeros(7)} for m in cfg.methods}
    rows = []
    for step in range(cfg.simulation.num_steps):
        t = step * cfg.simulation.dt
        for method in cfg.methods:
            if method == "curobo":
                solve_time, info = curobo_step(states[method], *controllers["curobo"], scene, cfg, t, mpc_module)
            elif method == "mpc_python":
                solve_time, info = mpc_step(states[method], scene, cfg, t, mpc_module)
            else:
                raise ValueError(f"Unknown method: {method}")
            row = evaluate(method, states[method], info, scene, cfg, t, solve_time, mpc_module)
            row["step"] = step
            rows.append(row)
            if cfg.simulation.stop_on_goal and row["goal_reached"]:
                break
    if "curobo" in controllers:
        controllers["curobo"][0].destroy()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--methods", nargs="+", choices=["curobo", "mpc_python"], default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.methods is not None:
        cfg.methods = args.methods
    if args.num_steps is not None:
        cfg.simulation.num_steps = args.num_steps
    align_curobo_iters(cfg)

    if "curobo" in cfg.methods and not torch.cuda.is_available():
        raise RuntimeError("cuRobo needs CUDA; use --methods mpc_python on CPU-only sessions.")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    setup_curobo_logger("error")
    mpc_module = load_mpc_module()

    rows = []
    for scene in cfg.scenes:
        rows.extend(run_scene(scene, cfg, mpc_module))
    summary = summarize(rows)

    prefix = Path(cfg.output_prefix)
    trials_path = prefix.with_name(prefix.name + "_trials.csv")
    summary_csv_path = prefix.with_name(prefix.name + "_summary.csv")
    summary_yml_path = prefix.with_name(prefix.name + "_summary.yml")
    write_csv(trials_path, rows)
    write_csv(summary_csv_path, summary)
    write_yaml(summary, str(summary_yml_path))

    for row in summary:
        print(
            f"{row['scene']} {row['method']}: reached={row['goal_reached']}, "
            f"final_error={row['final_position_error_m']:.4f}m, "
            f"min_dist={row['min_collision_distance_m']:.4f}m, "
            f"mean_solve={row['mean_solve_time_s']:.4f}s"
        )
    print(f"Wrote trials to: {trials_path}")
    print(f"Wrote summary to: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
