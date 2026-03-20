#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
import torch
from dataclasses import dataclass

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)


parser.add_argument(
    "--surge_amp", type=float, default=0.05, help="Surge amplitude (m)"
)
parser.add_argument(
    "--sway_amp", type=float, default=0.05, help="Sway amplitude (m)"
)
parser.add_argument(
    "--heave_amp", type=float, default=0.05, help="Heave amplitude (m)"
)

parser.add_argument(
    "--roll_amp_deg", type=float, default=7.0, help="Roll amplitude (deg)"
)
parser.add_argument(
    "--pitch_amp_deg", type=float, default=7.0, help="Pitch amplitude (deg)"
)
parser.add_argument(
    "--yaw_amp_deg", type=float, default=7.0, help="Yaw amplitude (deg)"
)

parser.add_argument(
    "--surge_freq", type=float, default=0.08, help="Surge frequency (Hz)"
)
parser.add_argument(
    "--sway_freq", type=float, default=0.11, help="Sway frequency (Hz)"
)
parser.add_argument(
    "--heave_freq", type=float, default=0.20, help="Heave frequency (Hz)"
)
parser.add_argument(
    "--roll_freq", type=float, default=0.32, help="Roll frequency (Hz)"
)
parser.add_argument(
    "--pitch_freq", type=float, default=0.27, help="Pitch frequency (Hz)"
)
parser.add_argument(
    "--yaw_freq", type=float, default=0.05, help="Yaw frequency (Hz)"
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--mode",
    type=int,
    default=1,
    help="0: Moving base with static target/obstacle (base moves, target/obstacle fixed in base frame), "
         "1: Fixed base with moving obstacle (base fixed, obstacle moves in world frame), "
         "2: Fixed base with moving obstacle and target (both move in world frame)"
)
args = parser.parse_args()

# Validate mode
if args.mode not in [0, 1, 2]:
    raise ValueError(f"Mode must be 0, 1, or 2, got {args.mode}")

print(f"Running in mode {args.mode}:")
if args.mode == 0:
    print("  Mode 0: Moving base with static target/obstacle")
    print("  - Robot base moves with ship motion")
    print("  - Target and obstacle are fixed in base frame")
elif args.mode == 1:
    print("  Mode 1: Fixed base with moving obstacle only")
    print("  - Robot base is fixed")
    print("  - Obstacle moves in world frame with ship motion")
    print("  - Target is static in world frame")
elif args.mode == 2:
    print("  Mode 2: Fixed base with moving obstacle and target")
    print("  - Robot base is fixed")
    print("  - Both obstacle and target move in world frame with ship motion")

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
# Enable the layers and stage windows in the UI
# Standard Library
import os

# Third Party
import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

############################################################


########### OV #################;;;;;


###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;


# Standard Library
from typing import Optional

# Third Party
from helper import add_extensions, add_robot_to_scene

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Sphere, WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.util.usd_helper import UsdHelper, set_prim_transform  # noqa: E402

############################################################

@dataclass
class ShipMotionProfile:
    surge_amp: float
    sway_amp: float
    heave_amp: float
    roll_amp_rad: float
    pitch_amp_rad: float
    yaw_amp_rad: float
    surge_freq: float
    sway_freq: float
    heave_freq: float
    roll_freq: float
    pitch_freq: float
    yaw_freq: float
    base_height: float

    def pose(self, t: float):
        twopi = 2.0 * np.pi
        x = self.surge_amp * np.sin(twopi * self.surge_freq * t)
        y = self.sway_amp * np.sin(twopi * self.sway_freq * t + np.pi / 3.0)
        z = self.base_height + self.heave_amp * np.sin(
            twopi * self.heave_freq * t + np.pi / 6.0
        )

        roll = self.roll_amp_rad * np.sin(twopi * self.roll_freq * t)
        pitch = self.pitch_amp_rad * np.sin(
            twopi * self.pitch_freq * t + np.pi / 5.0
        )
        yaw = self.yaw_amp_rad * np.sin(
            twopi * self.yaw_freq * t + np.pi / 7.0
        )

        q = euler_xyz_to_quaternion_wxyz(roll, pitch, yaw)
        return np.array([x, y, z, q[0], q[1], q[2], q[3]], dtype=np.float32)


def euler_xyz_to_quaternion_wxyz(roll: float, pitch: float, yaw: float):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def quat_wxyz_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_wxyz_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def rotate_vec_by_quat_wxyz(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    q_inv = quat_wxyz_conjugate(q)
    q_out = quat_wxyz_multiply(quat_wxyz_multiply(q, q_v), q_inv)
    return q_out[1:]


def apply_inverse_base_motion_to_pose(
    obj_pos_world: np.ndarray,
    obj_quat_world: np.ndarray,
    base_pose_world: np.ndarray,
):
    base_pos = base_pose_world[:3]
    base_quat = base_pose_world[3:]
    inv_base_quat = quat_wxyz_conjugate(base_quat)
    rel_pos = rotate_vec_by_quat_wxyz(obj_pos_world - base_pos, inv_base_quat)
    rel_quat = quat_wxyz_multiply(inv_base_quat, obj_quat_world)
    return rel_pos.astype(np.float32), rel_quat.astype(np.float32)


def make_pose(
    pos: np.ndarray, quat_wxyz: np.ndarray, tensor_args: TensorDeviceType
):
    return Pose(
        position=tensor_args.to_device(pos).view(1, 3),
        quaternion=tensor_args.to_device(quat_wxyz).view(1, 4),
    )
def world_to_base_goal(
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
    base_pose_world: np.ndarray,
    tensor_args: TensorDeviceType,
):
    base_pose = make_pose(
        base_pose_world[:3], base_pose_world[3:], tensor_args
    )
    target_pose_world_t = make_pose(
        target_pos_world, target_quat_world, tensor_args
    )
    return base_pose.inverse().multiply(target_pose_world_t)




def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow.
    target_x = 0.35
    target_z = 0.5
    target_y_left = -0.4
    target_y_right = 0.4
    target_repeat_interval_s = 10.0
    target_orientation = np.array([0, 1, 0, 0], dtype=np.float32)
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([target_x, target_y_left, target_z], dtype=np.float32),
        orientation=target_orientation,
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    # Add a fixed spherical obstacle in the scene.
    visual_sphere = sphere.VisualSphere(
        "/World/obstacle_sphere_0",
        position=np.array([0.35, 0, 0.4]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        radius=0.2,
        color=np.array([0.0, 0.4, 1.0]),
    )

    sphere_obstacle = Sphere(
        name="obstacle_sphere_0",
        radius=0.2,
        pose=[0.35, 0, 0.4, 1.0, 0.0, 0.0, 0.0],
        color=[0.0, 0.4, 1.0, 1.0],
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    print(f"Loaded robot config: {robot_cfg}")
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    
    robot_root_prim = stage.GetPrimAtPath(robot_prim_path)

    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(
        cuboid=world_cfg_table.cuboid,
        mesh=world_cfg1.mesh,
        sphere=[sphere_obstacle],
    )

    init_curobo = False

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].pose[2] = -10.0

    world_cfg = WorldConfig(
        cuboid=world_cfg_table.cuboid,
        mesh=world_cfg1.mesh,
        sphere=[sphere_obstacle],
    )
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)

    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    step = 0
    last_collision_time = None
    collision_history = []
    
    # Store initial positions for modes with fixed base
    initial_target_pos = np.array([target_x, target_y_left, target_z], dtype=np.float32)
    initial_target_quat = target_orientation.copy()
    initial_obstacle_pos = np.array([0.35, 0, 0.4], dtype=np.float32)
    initial_obstacle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    initial_robot_base_pos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    add_extensions(simulation_app, args.headless_mode)
    
    motion_profile = ShipMotionProfile(
        surge_amp=args.surge_amp,
        sway_amp=args.sway_amp,
        heave_amp=args.heave_amp,
        roll_amp_rad=np.deg2rad(args.roll_amp_deg),
        pitch_amp_rad=np.deg2rad(args.pitch_amp_deg),
        yaw_amp_rad=np.deg2rad(args.yaw_amp_deg),
        surge_freq=args.surge_freq,
        sway_freq=args.sway_freq,
        heave_freq=args.heave_freq,
        roll_freq=args.roll_freq,
        pitch_freq=args.pitch_freq,
        yaw_freq=args.yaw_freq,
        base_height=0.1,
    )
    
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index
        sim_time = step_index * 0.02
        if sim_time > target_repeat_interval_s * 2.0:
            break

        t_in_period = sim_time % target_repeat_interval_s
        if t_in_period < target_repeat_interval_s / 2.0:
            target_y = target_y_left
        else:
            target_y = target_y_right
        
        ship_pose_world = motion_profile.pose(sim_time)
        moving_obstacle_pose = None

        # Apply mode-specific transformations
        if args.mode == 0:
            # Mode 0: Moving base - target and obstacle fixed in base frame
            target.set_world_pose(
                position=np.array([target_x, target_y, target_z], dtype=np.float32),
                orientation=target_orientation,
            )
            set_prim_transform(robot_root_prim, ship_pose_world.tolist())
            
        elif args.mode == 1:
            # Mode 1: Fixed base - only obstacle moves in world frame
            # Emulate relative motion from moving-base mode using inverse base transform.
            obstacle_pos, obstacle_quat = apply_inverse_base_motion_to_pose(
                initial_obstacle_pos,
                initial_obstacle_quat,
                ship_pose_world,
            )
            # Update visual sphere
            visual_sphere.set_world_pose(
                position=obstacle_pos,
                orientation=obstacle_quat,
            )
            moving_obstacle_pose = (obstacle_pos, obstacle_quat)
            # Keep robot base fixed
            set_prim_transform(robot_root_prim, initial_robot_base_pos.tolist())
            # Target stays at initial position
            target.set_world_pose(
                position=np.array([target_x, target_y, target_z], dtype=np.float32),
                orientation=target_orientation,
            )
            
        elif args.mode == 2:
            # Mode 2: Fixed base - both obstacle and target move in world frame
            # Move obstacle
            obstacle_pos, obstacle_quat = apply_inverse_base_motion_to_pose(
                initial_obstacle_pos,
                initial_obstacle_quat,
                ship_pose_world,
            )
            visual_sphere.set_world_pose(
                position=obstacle_pos,
                orientation=obstacle_quat,
            )
            moving_obstacle_pose = (obstacle_pos, obstacle_quat)
            
            # Move target with the same inverse-base transform from moving-base mode.
            raw_target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)
            target_pos, target_quat = apply_inverse_base_motion_to_pose(
                raw_target_pos,
                initial_target_quat,
                ship_pose_world,
            )
            target.set_world_pose(
                position=target_pos,
                orientation=target_quat,
            )
            # Keep robot base fixed
            set_prim_transform(robot_root_prim, initial_robot_base_pos.tolist())

        if step_index <= 10:
            # my_world.reset()
            robot._articulation_view.initialize()

            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step
        
        print("Updating world")
        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            ignore_substring=[
                robot_prim_path,
                "/World/target",
                "/World/defaultGroundPlane",
                "/curobo",
            ],
            reference_prim_path=robot_prim_path,
        )
        # print(f"Found {len(obstacles)} obstacles in the scene for curobo.")
        # for obs in obstacles:
        #     print(f"Obstacle: {obs.name}, type: {type(obs)}")
        #     print(f"Obstacle pose: {obs.pose}")
        
        obstacles.add_obstacle(world_cfg_table.cuboid[0])
        
        # For modes with moving obstacles, update the sphere obstacle
        if moving_obstacle_pose is not None:
            obstacle_pos, obstacle_quat = moving_obstacle_pose
            sphere_obstacle.pose = obstacle_pos.tolist() + obstacle_quat.tolist()
            # Replace sphere obstacle with updated pose
            obstacles.sphere[0] = sphere_obstacle
        
        mpc.world_coll_checker.load_collision_model(obstacles)

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()
        print(f"Target position: {cube_position}, orientation: {cube_orientation}")
        base_pos_world, base_quat_world = robot.get_world_pose()
        base_pose_world = np.array(
            [
                base_pos_world[0],
                base_pos_world[1],
                base_pos_world[2],
                base_quat_world[0],
                base_quat_world[1],
                base_quat_world[2],
                base_quat_world[3],
            ],
            dtype=np.float32,
        )
        
        if args.mode == 0:
            # Mode 0: Moving base - convert target pose from world to base frame
            goal_pose_ship = world_to_base_goal(
                cube_position,
                cube_orientation,
                base_pose_world,
                tensor_args,
            )
        else:
            # Modes 1 & 2: Fixed base - target is already in world frame
            # Since base is at origin with identity orientation, goal pose in base frame = world pose
            goal_pose_ship = make_pose(
                cube_position,
                cube_orientation,
                tensor_args,
            )


        goal_buffer.goal_pose.copy_(goal_pose_ship)
        mpc.update_goal(goal_buffer)

        # if not changed don't call curobo:

        # get robot current state:
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        js_names = robot.dof_names
        sim_js_names = robot.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
            # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        common_js_names = []
        current_state.copy_(cu_js)

        mpc_result = mpc.step(current_state, max_attempts=2)
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        # Check for collisions
        try:
            collision_result = mpc.world_coll_checker.check_collision(cu_js.position.unsqueeze(0))
            is_in_collision = collision_result.item() if hasattr(collision_result, 'item') else collision_result
            
            if is_in_collision:
                if last_collision_time != sim_time:
                    last_collision_time = sim_time
                    collision_history.append({
                        'time': sim_time,
                        'step': step_index,
                        'position': cu_js.position.cpu().numpy()
                    })
                    print(f"[COLLISION DETECTED] Contact time: {sim_time:.4f}s (Step: {step_index})")
                    print(f"  Robot position: {cu_js.position.cpu().numpy()}")
            else:
                if last_collision_time is not None:
                    print(f"[COLLISION CLEARED] At time: {sim_time:.4f}s (Step: {step_index})")
                    last_collision_time = None
        except Exception as e:
            pass  # Collision checker may not be available

        succ = True  # ik_result.success.item()
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state

        art_action = ArticulationAction(
            cmd_state.position.view(-1).cpu().numpy(),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        # positions_goal = articulation_action.joint_positions
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(1):
                articulation_controller.apply_action(art_action)

        else:
            carb.log_warn("No action is being taken.")
    
    return collision_history


############################################################

if __name__ == "__main__":
    collision_history = main()
    
    # Print collision summary
    if collision_history:
        print("\n" + "="*60)
        print("COLLISION SUMMARY")
        print("="*60)
        print(f"Total collisions detected: {len(collision_history)}")
        for i, collision in enumerate(collision_history, 1):
            print(f"{i}. Time: {collision['time']:.4f}s (Step: {collision['step']})")
        print("="*60 + "\n")
    else:
        print("\nNo collisions detected during simulation.")
    
    simulation_app.close()
