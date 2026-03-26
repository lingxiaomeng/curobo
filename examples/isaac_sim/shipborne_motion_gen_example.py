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

_ = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse
import time

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

parser.add_argument("--surge_amp", type=float, default=0.05, help="Surge amplitude (m)")
parser.add_argument("--sway_amp", type=float, default=0.05, help="Sway amplitude (m)")
parser.add_argument("--heave_amp", type=float, default=0.05, help="Heave amplitude (m)")

parser.add_argument("--roll_amp_deg", type=float, default=10.0, help="Roll amplitude (deg)")
parser.add_argument("--pitch_amp_deg", type=float, default=10.0, help="Pitch amplitude (deg)")
parser.add_argument("--yaw_amp_deg", type=float, default=7.0, help="Yaw amplitude (deg)")

parser.add_argument("--surge_freq", type=float, default=0.08, help="Surge frequency (Hz)")
parser.add_argument("--sway_freq", type=float, default=0.11, help="Sway frequency (Hz)")
parser.add_argument("--heave_freq", type=float, default=0.20, help="Heave frequency (Hz)")
parser.add_argument("--roll_freq", type=float, default=0.32, help="Roll frequency (Hz)")
parser.add_argument("--pitch_freq", type=float, default=0.27, help="Pitch frequency (Hz)")
parser.add_argument("--yaw_freq", type=float, default=0.05, help="Yaw frequency (Hz)")

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--mode",
    type=int,
    default=0,
    help="0: Moving base with static target/obstacle (base moves, target/obstacle fixed in world frame), "
    "1: Fixed base with fixed obstacle and target, "
    "2: Fixed base with moving obstacle and target (both move in world frame)",
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=2,
    help="MotionGen planning attempts per simulation step",
)
parser.add_argument(
    "--replan_time_scale",
    type=float,
    default=1.0,
    help="Scale factor for adaptive replan interval computed from measured planning time.",
)
args = parser.parse_args()

if args.mode not in [0, 1, 2]:
    raise ValueError(f"Mode must be 0, 1, or 2, got {args.mode}")

print(f"Running in mode {args.mode}:")
if args.mode == 0:
    print("  Mode 0: Moving base with static target/obstacle")
    print("  - Robot base moves with ship motion")
    print("  - Target and obstacle are fixed in world frame")
elif args.mode == 1:
    print("  Mode 1: Fixed base with fixed obstacle and target")
    print("  - Robot base is fixed")
    print("  - Obstacle is fixed in world frame")
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
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer
from curobo.geom.types import Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


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
        z = self.base_height + self.heave_amp * np.sin(twopi * self.heave_freq * t + np.pi / 6.0)

        roll = self.roll_amp_rad * np.sin(twopi * self.roll_freq * t)
        pitch = self.pitch_amp_rad * np.sin(twopi * self.pitch_freq * t + np.pi / 5.0)
        yaw = self.yaw_amp_rad * np.sin(twopi * self.yaw_freq * t + np.pi / 7.0)

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


def make_pose(pos: np.ndarray, quat_wxyz: np.ndarray, tensor_args: TensorDeviceType):
    return Pose(
        position=tensor_args.to_device(pos).view(1, 3),
        quaternion=tensor_args.to_device(quat_wxyz).view(1, 4),
    )


def get_robot_obstacle_collision_metrics(motion_gen: MotionGen, joint_state: JointState):
    """Return distance-like and hard-constraint collision metrics.

    collision_distance is from world distance query.
    collision_constraint is a hard indicator (>0 means in collision).
    """
    kin_state = motion_gen.compute_kinematics(joint_state)
    robot_spheres = kin_state.robot_spheres
    if robot_spheres is None:
        return 0.0, 0.0
    if len(robot_spheres.shape) == 3:
        robot_spheres = robot_spheres.unsqueeze(1)

    coll_query_buffer = CollisionQueryBuffer()
    coll_query_buffer.update_buffer_shape(
        robot_spheres.shape,
        motion_gen.tensor_args,
        motion_gen.world_coll_checker.collision_types,
    )

    unit_weight = robot_spheres.new_tensor([1.0])
    zero_activation_distance = robot_spheres.new_tensor([0.0])

    coll_distance = motion_gen.world_coll_checker.get_sphere_distance(
        robot_spheres,
        coll_query_buffer,
        unit_weight,
        zero_activation_distance,
        env_query_idx=None,
        return_loss=False,
        sum_collisions=True,
    )
    coll_constraint = motion_gen.world_coll_checker.get_sphere_collision(
        robot_spheres,
        coll_query_buffer,
        unit_weight,
        zero_activation_distance,
        env_query_idx=None,
        return_loss=False,
    )
    return float(torch.max(coll_distance).item()), float(torch.max(coll_constraint).item())


def main():
    total_steps = 0
    sim_dt = 0.02  # should match motion_gen_config.trajopt_dt

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    my_world.scene.add_default_ground_plane()

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
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    usd_help = UsdHelper()
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    robot_root_prim = stage.GetPrimAtPath(robot_prim_path)
    articulation_controller = None

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    table_pose_world = np.array(world_cfg_table.cuboid[0].pose, dtype=np.float32)

    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.0

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh, sphere=[sphere_obstacle])

    # Dynamic ship motion requires reactive replanning every step.
    trajopt_dt = 0.04
    optimize_dt = False
    trajopt_tsteps = 40
    trim_steps = [1, None]
    interpolation_dt = trajopt_dt
    enable_finetune_trajopt = False
    max_attempts = args.max_attempts

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("Warming up MotionGen...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=1,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=1.0,
    )

    usd_help.load_stage(my_world.stage)
    init_world = False

    plan_success_count = 0
    plan_fail_count = 0
    collision_constraint_threshold = 0.0
    collision_history = []
    cmd_plan = None
    cmd_idx = 0
    next_replan_step = 0
    idx_list = [robot.get_dof_index(x) for x in j_names]
    spheres = None
    past_cmd = None

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

    ship_pose_world_0 = motion_profile.pose(0.0)
    T_world_ship_0 = make_pose(ship_pose_world_0[:3], ship_pose_world_0[3:], tensor_args)
    T_world_obstacle_initial = make_pose(initial_obstacle_pos, initial_obstacle_quat, tensor_args)
    T_ship0_obstacle = T_world_ship_0.inverse().multiply(T_world_obstacle_initial)

    while simulation_app.is_running():

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        total_steps += 1

        # Hold dynamics during startup so base/joint initialization settles first.
        sim_time = max(0.0, (step_index - 20) * sim_dt)
        if sim_time > target_repeat_interval_s * 3.0:
            break

        t_in_period = sim_time % target_repeat_interval_s
        target_y = target_y_left if t_in_period < target_repeat_interval_s / 2.0 else target_y_right

        ship_pose_world = motion_profile.pose(sim_time)
        T_world_ship_motion = make_pose(ship_pose_world[:3], ship_pose_world[3:], tensor_args)

        T_world_base = make_pose(initial_robot_base_pos[:3], initial_robot_base_pos[3:], tensor_args)

        T_world_target_ref = make_pose(
            np.array([target_x, target_y, target_z], dtype=np.float32),
            target_orientation,
            tensor_args,
        )

        if args.mode == 0:
            target.set_world_pose(
                position=np.array([target_x, target_y, target_z], dtype=np.float32),
                orientation=target_orientation,
            )
            set_prim_transform(robot_root_prim, ship_pose_world.tolist())
            T_world_base = T_world_ship_motion

        elif args.mode == 1:
            T_world_obstacle_fixed = T_ship0_obstacle
            obstacle_pos_world = T_world_obstacle_fixed.position.cpu().numpy().flatten()
            obstacle_quat_world = T_world_obstacle_fixed.quaternion.cpu().numpy().flatten()
            visual_sphere.set_world_pose(position=obstacle_pos_world, orientation=obstacle_quat_world)

            set_prim_transform(robot_root_prim, initial_robot_base_pos.tolist())
            target.set_world_pose(
                position=np.array([target_x, target_y, target_z], dtype=np.float32),
                orientation=target_orientation,
            )

        elif args.mode == 2:
            T_world_obstacle_moving = T_world_ship_motion.multiply(T_ship0_obstacle)
            obstacle_pos_world = T_world_obstacle_moving.position.cpu().numpy().flatten()
            obstacle_quat_world = T_world_obstacle_moving.quaternion.cpu().numpy().flatten()
            visual_sphere.set_world_pose(position=obstacle_pos_world, orientation=obstacle_quat_world)

            T_ship0_target = T_world_ship_0.inverse().multiply(T_world_target_ref)
            T_world_target_moving = T_world_ship_motion.multiply(T_ship0_target)
            target_pos_world = T_world_target_moving.position.cpu().numpy().flatten()
            target_quat_world = T_world_target_moving.quaternion.cpu().numpy().flatten()
            target.set_world_pose(position=target_pos_world, orientation=target_quat_world)

            set_prim_transform(robot_root_prim, initial_robot_base_pos.tolist())

        if step_index <= 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for _ in range(len(idx_list))]),
                joint_indices=idx_list,
            )
        if step_index < 20:
            continue

        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            ignore_substring=[
                robot_prim_path,
                "/World/target",
                "/World/obstacle_sphere_0",
                "/World/defaultGroundPlane",
                "/curobo",
            ],
            reference_prim_path=robot_prim_path,
        )

        obstacle_pos_world_vis, obstacle_quat_world_vis = visual_sphere.get_world_pose()
        T_world_obstacle = make_pose(obstacle_pos_world_vis, obstacle_quat_world_vis, tensor_args)
        T_ship_obstacle = T_world_base.inverse().multiply(T_world_obstacle)
        obstacle_pos = T_ship_obstacle.position.cpu().numpy().flatten()
        obstacle_quat = T_ship_obstacle.quaternion.cpu().numpy().flatten()
        sphere_obstacle.pose = np.concatenate([obstacle_pos, obstacle_quat]).tolist()
        obstacles.add_obstacle(sphere_obstacle)

        T_world_table = make_pose(table_pose_world[:3], table_pose_world[3:], tensor_args)
        T_ship_table = T_world_base.inverse().multiply(T_world_table)
        world_cfg_table.cuboid[0].pose = np.concatenate(
            [
                T_ship_table.position.cpu().numpy().flatten(),
                T_ship_table.quaternion.cpu().numpy().flatten(),
            ]
        ).tolist()
        obstacles.add_obstacle(world_cfg_table.cuboid[0])

        collision_world = obstacles.get_collision_check_world(mesh_process=False)
        motion_gen.update_world(collision_world)

        cube_position, cube_orientation = target.get_world_pose()
        T_world_target = make_pose(cube_position, cube_orientation, tensor_args)
        T_ship_target = T_world_base.inverse().multiply(T_world_target)

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue

        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        if past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        collision_distance, collision_constraint = get_robot_obstacle_collision_metrics(
            motion_gen, cu_js
        )
        if collision_constraint > collision_constraint_threshold:
            collision_event = {
                "time": sim_time,
                "step": int(step_index),
                "collision_distance": collision_distance,
                "collision_constraint": collision_constraint,
            }
            collision_history.append(collision_event)
            print(
                "[COLLISION] "
                f"t={sim_time:.3f}s, step={int(step_index)}, "
                f"distance={collision_distance:.6f}, "
                f"constraint={collision_constraint:.6f}"
            )

        active_plan = cmd_plan is not None and len(cmd_plan.position) > 0 and cmd_idx < len(cmd_plan.position)
        should_replan = (not active_plan) or (step_index >= next_replan_step)

        if should_replan:
            plan_t0 = time.perf_counter()
            result = motion_gen.plan_single(cu_js.unsqueeze(0), T_ship_target, plan_config)
            plan_time_s = max(0.0, time.perf_counter() - plan_t0)
            adaptive_replan_steps = max(
                1,
                int(np.ceil((plan_time_s * args.replan_time_scale) / sim_dt)),
            )
            next_replan_step = step_index + adaptive_replan_steps

            if not result.success.item():
                plan_fail_count += 1
                carb.log_warn("MotionGen failed to converge: " + str(result.status))
            else:
                plan_success_count += 1

                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)

                common_js_names = []
                idx_list = []
                for joint_name in sim_js_names:
                    if joint_name in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(joint_name))
                        common_js_names.append(joint_name)
                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 1 if len(cmd_plan.position) > 1 else 0

        if cmd_plan is not None and len(cmd_plan.position) > 0:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()

            art_action = ArticulationAction(
                cmd_state.position.view(-1).cpu().numpy(),
                cmd_state.velocity.view(-1).cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)

            cmd_idx += 1
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None

    return total_steps, plan_success_count, plan_fail_count, collision_history


if __name__ == "__main__":
    total_steps, plan_success_count, plan_fail_count, collision_history = main()

    print("\n" + "=" * 60)
    print("MOTIONGEN SHIPBORNE SUMMARY")
    print("=" * 60)
    print(f"Total steps simulated: {total_steps}")
    print(f"Successful replans: {plan_success_count}")
    print(f"Failed replans: {plan_fail_count}")
    if total_steps > 0:
        print(f"Success ratio: {plan_success_count / total_steps:.4f} plans/step")
    print("=" * 60 + "\n")

    if collision_history:
        print("\n" + "=" * 60)
        print("COLLISION SUMMARY")
        print("=" * 60)
        for i, collision in enumerate(collision_history, 1):
            print(
                f"{i}. Time: {collision['time']:.4f}s "
                f"(Step: {collision['step']}), "
                f"Distance: {collision['collision_distance']:.6f}, "
                f"Constraint: {collision['collision_constraint']:.6f}"
            )
        print(f"Total collisions detected: {len(collision_history)}")
        print("=" * 60 + "\n")
    else:
        print("\nNo collisions detected during simulation.")

    simulation_app.close()
