# SPDX-License-Identifier: Apache-2.0
"""Moving-base scene visualizer for motion-planning results.

This module is intentionally independent from the benchmark runner.  It renders
obstacles, the robot link skeleton, and end-effector / joint trajectory in the
world frame while both the robot and scene obstacles are attached to the moving
base frame (for example, a ship deck).

Typical use in ``base_motion_plan_benchmark.py`` after a successful plan::

    from base_motion_visualizer import (
        make_base_motion_poses_from_velocity,
        extract_robot_trace_from_curobo,
        render_moving_base_scene,
    )

    base_poses = make_base_motion_poses_from_velocity(base_velocity, base_bundle.base_dt)
    robot_trace = extract_robot_trace_from_curobo(bundle.planner, result.js_solution)
    render_moving_base_scene(
        problem=problem,
        base_poses_world=base_poses,
        robot_trace_base=robot_trace,
        output_html="debug_scene.html",
        title=f"{dataset_name}/{group_name}/{problem_index}",
    )

Coordinate convention
---------------------
- ``problem`` obstacles and ``robot_trace_base`` are expressed in the robot/base
  frame, matching cuRobo planning datasets.
- ``base_poses_world[t]`` maps a point from base frame to world frame at time t.
- The renderer transforms both obstacles and robot points by the same base pose,
  so the complete scene moves together with the base.
"""

from __future__ import annotations

import io
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import numpy as np


ArrayLike = Any


@dataclass
class RobotTrace:
    """Robot link points and optional end-effector points in the base frame.

    Attributes:
        link_points_base: Array with shape ``[T, L, 3]``. Each row is an ordered
            chain of link points; consecutive points are drawn as line segments.
        ee_points_base: Optional array with shape ``[T, 3]``. If omitted, the last
            link point is used as the end-effector trajectory.
        link_names: Optional names for the link points.
    """

    link_points_base: np.ndarray
    ee_points_base: Optional[np.ndarray] = None
    link_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.link_points_base = np.asarray(self.link_points_base, dtype=float)
        if self.link_points_base.ndim != 3 or self.link_points_base.shape[-1] != 3:
            raise ValueError("link_points_base must have shape [T, L, 3]")
        if self.ee_points_base is not None:
            self.ee_points_base = np.asarray(self.ee_points_base, dtype=float)
            if self.ee_points_base.shape != (self.link_points_base.shape[0], 3):
                raise ValueError("ee_points_base must have shape [T, 3]")


@dataclass
class RobotMeshLink:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    local_pose: np.ndarray


@dataclass
class RobotMeshTrace:
    """Robot visual meshes and per-link poses in the base frame."""

    links: List[RobotMeshLink]
    link_poses_base: np.ndarray

    def __post_init__(self) -> None:
        self.link_poses_base = np.asarray(self.link_poses_base, dtype=float)
        if self.link_poses_base.ndim != 4 or self.link_poses_base.shape[-2:] != (4, 4):
            raise ValueError("link_poses_base must have shape [T, L, 4, 4]")
        if self.link_poses_base.shape[1] != len(self.links):
            raise ValueError("link pose count must match mesh link count")


@dataclass
class BoxObstacle:
    name: str
    dims: np.ndarray
    pose_base: np.ndarray


@dataclass
class SphereObstacle:
    name: str
    radius: float
    center_base: np.ndarray


@dataclass
class CylinderObstacle:
    name: str
    radius: float
    height: float
    pose_base: np.ndarray


# -----------------------------------------------------------------------------
# Rigid transforms
# -----------------------------------------------------------------------------


def _as_numpy(x: Any) -> np.ndarray:
    """Detach torch-like arrays and return a NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def make_transform(position: Sequence[float], rotation: Optional[np.ndarray] = None) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, 3] = np.asarray(position, dtype=float)
    if rotation is not None:
        t[:3, :3] = np.asarray(rotation, dtype=float)
    return t


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    flat = points.reshape(-1, 3)
    out = flat @ transform[:3, :3].T + transform[:3, 3]
    return out.reshape(points.shape)


def transform_pose(transform_world_base: np.ndarray, pose_base: np.ndarray) -> np.ndarray:
    return transform_world_base @ pose_base


def pose_from_position_quaternion(position: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    return make_transform(position, quat_wxyz_to_matrix(quaternion))


def make_base_motion_poses_from_velocity(
    base_velocity: ArrayLike,
    dt: float,
    initial_pose_world: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Approximate base poses by integrating spatial velocity.

    Args:
        base_velocity: Array of shape ``[T, 6]`` in cuRobo spatial order
            ``[angular xyz, linear xyz]``.
        dt: Integration timestep.
        initial_pose_world: Optional 4x4 pose. Identity is used by default.

    Returns:
        ``[T, 4, 4]`` transforms mapping base-frame points into world frame.

    Notes:
        This is a visualization integrator, not a dynamics integrator.  If your
        simulator already has exact base poses, pass those directly to
        ``render_moving_base_scene`` instead.
    """
    velocity = _as_numpy(base_velocity)
    if velocity.ndim != 2 or velocity.shape[1] != 6:
        raise ValueError("base_velocity must have shape [T, 6]")
    pose = np.eye(4, dtype=float) if initial_pose_world is None else np.array(initial_pose_world, dtype=float)
    poses = np.repeat(pose[None, :, :], velocity.shape[0], axis=0)
    angles = np.zeros(3, dtype=float)
    position = pose[:3, 3].copy()
    for i in range(velocity.shape[0]):
        if i > 0:
            angles += velocity[i - 1, :3] * dt
            position += velocity[i - 1, 3:] * dt
        poses[i] = make_transform(position, euler_xyz_to_matrix(*angles))
    return poses


def make_random_base_motion_poses(
    horizon: int,
    dt: float,
    seed: int,
    base_freq_min: float,
    base_freq_max: float,
    base_angle_amp_deg: float,
    base_yaw_scale: float,
    base_linear_amp_m: float,
) -> np.ndarray:
    """Generate exact visualization poses for the benchmark's sinusoidal base motion.

    This mirrors the random roll/pitch/yaw and translation profile used by the
    benchmark's moving-base generator, but returns poses directly for rendering.
    """
    rng = np.random.default_rng(seed)
    time_steps = np.arange(horizon, dtype=np.float64) * dt
    freq = rng.uniform(base_freq_min, base_freq_max, size=(2, 3))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(2, 3))
    angle_amp = np.deg2rad(base_angle_amp_deg) * rng.uniform(0.25, 1.0, size=3)
    angle_amp[2] *= base_yaw_scale
    linear_amp = base_linear_amp_m * rng.uniform(0.25, 1.0, size=3)
    angle_omega = 2.0 * math.pi * freq[0]
    linear_omega = 2.0 * math.pi * freq[1]
    angles = np.sin(time_steps[:, None] * angle_omega[None, :] + phase[0]) * angle_amp
    translation = np.sin(time_steps[:, None] * linear_omega[None, :] + phase[1]) * linear_amp
    poses = np.zeros((horizon, 4, 4), dtype=float)
    for i in range(horizon):
        poses[i] = make_transform(translation[i], euler_xyz_to_matrix(*angles[i]))
    return poses


# -----------------------------------------------------------------------------
# Obstacle parsing
# -----------------------------------------------------------------------------


def _pose_from_list(pose: Sequence[float]) -> np.ndarray:
    """Accept [x,y,z,qw,qx,qy,qz], [x,y,z], or a 4x4 matrix."""
    arr = np.asarray(pose, dtype=float)
    if arr.shape == (4, 4):
        return arr
    if arr.size >= 7:
        return make_transform(arr[:3], quat_wxyz_to_matrix(arr[3:7]))
    if arr.size >= 3:
        return make_transform(arr[:3])
    return np.eye(4, dtype=float)


def _iter_named_objects(container: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if isinstance(container, Mapping):
        for name, value in container.items():
            if isinstance(value, Mapping):
                yield str(value.get("name", name)), value
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        for i, value in enumerate(container):
            if isinstance(value, Mapping):
                yield str(value.get("name", f"obj_{i}")), value


def parse_obstacles(problem_or_obstacles: Mapping[str, Any]) -> Tuple[List[BoxObstacle], List[SphereObstacle], List[CylinderObstacle]]:
    """Parse common cuRobo / robometrics obstacle dictionaries.

    Supported keys include ``cuboid``, ``obb``, ``box``, ``sphere``, and
    ``cylinder``.  Unknown mesh-like objects are skipped unless they include a
    bounding-box-style ``dims`` field.
    """
    obstacles = problem_or_obstacles.get("obstacles", problem_or_obstacles)
    boxes: List[BoxObstacle] = []
    spheres: List[SphereObstacle] = []
    cylinders: List[CylinderObstacle] = []

    for key in ("cuboid", "obb", "box", "boxes"):
        for name, obj in _iter_named_objects(obstacles.get(key, {})):
            dims = obj.get("dims", obj.get("scale", obj.get("size", None)))
            pose = obj.get("pose", obj.get("position", obj.get("center", [0, 0, 0])))
            if dims is not None:
                boxes.append(BoxObstacle(name=name, dims=np.asarray(dims, dtype=float)[:3], pose_base=_pose_from_list(pose)))

    for key in ("mesh", "meshes"):
        for name, obj in _iter_named_objects(obstacles.get(key, {})):
            dims = obj.get("dims", obj.get("scale", None))
            pose = obj.get("pose", obj.get("position", [0, 0, 0]))
            if dims is not None:
                boxes.append(BoxObstacle(name=f"{name}_bbox", dims=np.asarray(dims, dtype=float)[:3], pose_base=_pose_from_list(pose)))

    for key in ("sphere", "spheres"):
        for name, obj in _iter_named_objects(obstacles.get(key, {})):
            radius = obj.get("radius", obj.get("r", None))
            pose = obj.get("pose", obj.get("position", obj.get("center", [0, 0, 0])))
            if radius is not None:
                center = _pose_from_list(pose)[:3, 3]
                spheres.append(SphereObstacle(name=name, radius=float(radius), center_base=center))

    for key in ("cylinder", "cylinders"):
        for name, obj in _iter_named_objects(obstacles.get(key, {})):
            radius = obj.get("radius", obj.get("r", None))
            height = obj.get("height", obj.get("h", None))
            pose = obj.get("pose", obj.get("position", [0, 0, 0]))
            if radius is not None and height is not None:
                cylinders.append(CylinderObstacle(name=name, radius=float(radius), height=float(height), pose_base=_pose_from_list(pose)))
    return boxes, spheres, cylinders


# -----------------------------------------------------------------------------
# cuRobo extraction helper
# -----------------------------------------------------------------------------


def extract_robot_trace_from_curobo(
    planner: Any,
    trajectory: Any,
    link_names: Optional[Sequence[str]] = None,
) -> RobotTrace:
    """Extract link-point trajectories from a cuRobo planner and JointState.

    The function uses public-ish objects available in cuRobo releases but keeps
    fallbacks because exact FK container names vary across versions.
    """
    # Prefer benchmark helper if caller imported this module next to the script.
    try:
        from base_motion_plan_benchmark import align_trajectory_for_kinematics, reshape_trajectory_to_bhd  # type: ignore

        fk_trajectory = align_trajectory_for_kinematics(planner, trajectory)
        fk_trajectory = reshape_trajectory_to_bhd(fk_trajectory)
    except Exception:
        fk_trajectory = trajectory
        pos = getattr(fk_trajectory, "position", None)
        if pos is not None and getattr(pos, "ndim", 0) == 2:
            # Rebuild the minimal JointState shape by mutating clone if possible.
            try:
                fk_trajectory = fk_trajectory.clone()
                fk_trajectory.position = fk_trajectory.position.reshape(1, fk_trajectory.position.shape[-2], fk_trajectory.position.shape[-1])
            except Exception:
                pass

    with _maybe_torch_no_grad():
        fk_state = planner.compute_kinematics(fk_trajectory)

    if link_names is None:
        sphere_trace = _extract_robot_sphere_trace(fk_state, planner)
        if sphere_trace is not None:
            return sphere_trace
        link_names = _available_fk_link_names(fk_state)
    else:
        available = set(_available_fk_link_names(fk_state))
        if available:
            link_names = [name for name in link_names if str(name) in available]

    points: List[np.ndarray] = []
    kept_names: List[str] = []
    for name in link_names:
        pose = _get_link_pose(fk_state, str(name))
        if pose is None:
            continue
        position = _as_numpy(getattr(pose, "position", pose))
        position = position.reshape(-1, 3)
        points.append(position)
        kept_names.append(str(name))

    if not points:
        # Last-resort: use tool frame only, which still gives a trajectory line.
        tool_frames = list(getattr(planner, "tool_frames", []) or [])
        for name in tool_frames:
            pose = _get_link_pose(getattr(fk_state, "tool_poses", fk_state), str(name))
            if pose is not None:
                position = _as_numpy(getattr(pose, "position", pose)).reshape(-1, 3)
                return RobotTrace(link_points_base=position[:, None, :], ee_points_base=position, link_names=[str(name)])
        raise RuntimeError("Could not extract any link poses from planner.compute_kinematics().")

    # Ensure all link arrays have the same T.  Truncate to the shortest valid trace.
    min_t = min(p.shape[0] for p in points)
    link_points = np.stack([p[:min_t] for p in points], axis=1)
    ee_points = link_points[:, -1, :]
    return RobotTrace(link_points_base=link_points, ee_points_base=ee_points, link_names=kept_names)


class _maybe_torch_no_grad:
    def __enter__(self):
        try:
            import torch

            self._ctx = torch.no_grad()
            return self._ctx.__enter__()
        except Exception:
            self._ctx = None
            return None

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


def _get_link_pose(fk_state: Any, link_name: str) -> Optional[Any]:
    for source in (fk_state, getattr(fk_state, "link_poses", None), getattr(fk_state, "tool_poses", None)):
        if source is None:
            continue
        source_links = getattr(source, "tool_frames", None)
        if source_links is not None and link_name not in source_links:
            continue
        if hasattr(source, "get_link_pose"):
            try:
                return source.get_link_pose(link_name)
            except Exception:
                pass
        if isinstance(source, Mapping) and link_name in source:
            return source[link_name]
        if hasattr(source, link_name):
            return getattr(source, link_name)
    return None


def _available_fk_link_names(fk_state: Any) -> List[str]:
    names = list(getattr(fk_state, "tool_frames", []) or [])
    tool_poses = getattr(fk_state, "tool_poses", None)
    names.extend(list(getattr(tool_poses, "tool_frames", []) or []))
    link_poses = getattr(fk_state, "link_poses", None)
    names.extend(list(getattr(link_poses, "tool_frames", []) or []))
    if isinstance(link_poses, Mapping):
        names.extend([str(name) for name in link_poses.keys()])
    return list(dict.fromkeys(str(name) for name in names))


def _extract_robot_sphere_trace(fk_state: Any, planner: Any) -> Optional[RobotTrace]:
    robot_spheres = getattr(fk_state, "robot_spheres", None)
    geometry = getattr(fk_state, "robot_collision_geometry", None)
    if robot_spheres is None or geometry is None:
        return None

    spheres = _as_numpy(robot_spheres)
    if spheres.ndim == 4:
        spheres = spheres.reshape(-1, spheres.shape[-2], spheres.shape[-1])
    elif spheres.ndim != 3:
        return None
    if spheres.shape[-1] < 4:
        return None

    link_sphere_idx_map = getattr(geometry, "link_sphere_idx_map", None)
    if link_sphere_idx_map is None:
        return None
    sphere_link_ids = _as_numpy(link_sphere_idx_map).astype(int).reshape(-1)
    if sphere_link_ids.shape[0] != spheres.shape[1]:
        return None

    kinematics = getattr(planner, "kinematics", None)
    kin_cfg = getattr(kinematics, "kinematics_config", None)
    if kin_cfg is None:
        kin_cfg = getattr(getattr(kinematics, "config", None), "kinematics_config", None)
    if kin_cfg is None:
        return None

    link_name_to_idx = getattr(kin_cfg, "link_name_to_idx_map", None) or {}
    mesh_link_names = list(getattr(kin_cfg, "mesh_link_names", []) or [])
    if not mesh_link_names and link_name_to_idx:
        mesh_link_names = list(link_name_to_idx.keys())

    points: List[np.ndarray] = []
    kept_names: List[str] = []
    for name in mesh_link_names:
        link_idx = link_name_to_idx.get(name)
        if link_idx is None:
            continue
        mask = sphere_link_ids == int(link_idx)
        if not np.any(mask):
            continue
        link_spheres = spheres[:, mask, :]
        valid = link_spheres[..., 3] > 0.0
        if not np.any(valid):
            continue
        weight = valid[..., None].astype(float)
        count = np.maximum(np.sum(weight, axis=1), 1.0)
        center = np.sum(link_spheres[..., :3] * weight, axis=1) / count
        points.append(center)
        kept_names.append(str(name))

    if not points:
        return None

    link_points = np.stack(points, axis=1)
    ee_points = None
    tool_frames = _available_fk_link_names(fk_state)
    for name in tool_frames:
        pose = _get_link_pose(fk_state, name)
        if pose is not None:
            ee_points = _as_numpy(getattr(pose, "position", pose)).reshape(-1, 3)
            ee_points = ee_points[: link_points.shape[0]]
            break
    if ee_points is None:
        ee_points = link_points[:, -1, :]
    return RobotTrace(link_points_base=link_points, ee_points_base=ee_points, link_names=kept_names)


def extract_robot_mesh_trace_from_curobo(
    planner: Any,
    trajectory: Any,
    link_names: Optional[Sequence[str]] = None,
) -> Optional[RobotMeshTrace]:
    """Extract visual mesh link poses for a cuRobo trajectory.

    The planning model usually stores only the end-effector tool pose.  For
    rendering, build a temporary FK model that stores poses for mesh links while
    leaving the planner untouched.
    """
    try:
        from curobo_floating_base._src.robot.kinematics.kinematics import Kinematics
        from curobo_floating_base._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from base_motion_plan_benchmark import align_trajectory_for_kinematics, reshape_trajectory_to_bhd  # type: ignore

        fk_trajectory = reshape_trajectory_to_bhd(
            align_trajectory_for_kinematics(planner, trajectory)
        )
        planner_kinematics = getattr(planner, "kinematics", None)
        planner_cfg = getattr(planner_kinematics, "config", None)
        generator_cfg = getattr(planner_cfg, "generator_config", None)
        kin_params = getattr(planner_cfg, "kinematics_config", None)
        if generator_cfg is None or kin_params is None:
            return None

        mesh_link_names = list(link_names or getattr(kin_params, "mesh_link_names", []) or [])
        mesh_link_names = [name for name in mesh_link_names if name != "attached_object"]
        if not mesh_link_names:
            return None

        visual_generator_cfg = deepcopy(generator_cfg)
        visual_generator_cfg.tool_frames = mesh_link_names
        visual_generator_cfg.load_tool_frames_with_mesh = False
        visual_cfg = KinematicsCfg.from_config(visual_generator_cfg)
        visual_kinematics = Kinematics(
            visual_cfg,
            compute_jacobian=False,
            compute_spheres=False,
            compute_com=False,
        )
        visual_state = visual_kinematics.compute_kinematics(fk_trajectory)
        tool_poses = visual_state.tool_poses
        if tool_poses is None:
            return None

        meshes = visual_kinematics.get_robot_link_meshes()
        links: List[RobotMeshLink] = []
        pose_indices: List[int] = []
        for i, (name, mesh) in enumerate(zip(mesh_link_names, meshes)):
            if mesh is None:
                continue
            trimesh_mesh = mesh.get_trimesh_mesh(process=False, transform_with_pose=False)
            vertices = np.asarray(trimesh_mesh.vertices, dtype=float)
            faces = np.asarray(trimesh_mesh.faces, dtype=int)
            if vertices.size == 0 or faces.size == 0:
                continue
            vertices, faces = _simplify_plotly_mesh(trimesh_mesh, max_faces=1200)
            links.append(
                RobotMeshLink(
                    name=str(name),
                    vertices=vertices.reshape(-1, 3),
                    faces=faces.reshape(-1, 3),
                    local_pose=_pose_from_list(mesh.pose or [0, 0, 0, 1, 0, 0, 0]),
                )
            )
            pose_indices.append(i)

        if not links:
            return None

        positions = np.take(_as_numpy(tool_poses.position)[0], pose_indices, axis=1)
        quaternions = np.take(_as_numpy(tool_poses.quaternion)[0], pose_indices, axis=1)
        link_poses = np.zeros((positions.shape[0], len(links), 4, 4), dtype=float)
        for t in range(positions.shape[0]):
            for li in range(len(links)):
                link_poses[t, li] = pose_from_position_quaternion(
                    positions[t, li],
                    quaternions[t, li],
                )
        return RobotMeshTrace(links=links, link_poses_base=link_poses)
    except Exception:
        return None


def _simplify_plotly_mesh(
    mesh: Any,
    max_faces: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep Plotly animation HTML small while preserving a closed surface."""
    if len(mesh.faces) > max_faces and hasattr(mesh, "simplify_quadric_decimation"):
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception:
            try:
                mesh = mesh.convex_hull
            except Exception:
                pass
    vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=int).reshape(-1, 3)
    if faces.shape[0] > max_faces:
        try:
            hull = mesh.convex_hull
            vertices = np.asarray(hull.vertices, dtype=float).reshape(-1, 3)
            faces = np.asarray(hull.faces, dtype=int).reshape(-1, 3)
        except Exception:
            face_ids = np.linspace(0, faces.shape[0] - 1, max_faces * 2, dtype=int)
            faces = faces[np.unique(face_ids)[:max_faces]]
    used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    compact_vertices = vertices[used_vertices]
    compact_faces = inverse.reshape(-1, 3)
    return compact_vertices, compact_faces


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def render_moving_base_scene(
    problem: Mapping[str, Any],
    base_poses_world: ArrayLike,
    robot_trace_base: RobotTrace,
    robot_mesh_trace_base: Optional[RobotMeshTrace] = None,
    output_html: str | Path = "moving_base_scene.html",
    output_gif: Optional[str | Path] = None,
    every_n: int = 1,
    frame_duration_ms: int = 33,
    gif_width: int = 1280,
    gif_height: int = 900,
    gif_scale: float = 1.0,
    gif_every_n: int = 1,
    gif_max_frames: int = 0,
    title: str = "Moving-base robot scene",
    show_frames: bool = True,
    show_trajectory: bool = True,
    auto_open: bool = False,
) -> Path:
    """Render moving-base obstacles, robot mesh, and trajectory with fixed view ranges.

    Args:
        problem: Dataset problem or raw obstacle dictionary.
        base_poses_world: ``[T, 4, 4]`` transforms mapping base-frame points into world.
        robot_trace_base: Robot link points in base frame.
        output_html: Destination HTML path.
        output_gif: Optional GIF path. Requires ``kaleido`` and ``imageio``.
        every_n: Downsample animation frames for large trajectories.
        frame_duration_ms: Plotly animation frame duration in milliseconds.
        gif_width: Exported GIF frame width in pixels.
        gif_height: Exported GIF frame height in pixels.
        gif_scale: Plotly static image scale for each exported GIF frame.
        gif_every_n: Downsample exported GIF frames after HTML frames are built.
        gif_max_frames: Maximum exported GIF frames; ``0`` keeps all selected frames.
        title: Figure title.
        show_frames: Draw small base axes for sampled frames.
        show_trajectory: Draw end-effector trajectory relative to the moving base.
        auto_open: Ask Plotly to open the browser.
    """
    import plotly.graph_objects as go

    base_poses = _as_numpy(base_poses_world)
    if base_poses.ndim != 3 or base_poses.shape[1:] != (4, 4):
        raise ValueError("base_poses_world must have shape [T, 4, 4]")
    trace = robot_trace_base
    n = min(base_poses.shape[0], trace.link_points_base.shape[0])
    if n == 0:
        raise ValueError("empty trajectory")
    base_poses = base_poses[:n]
    link_base = trace.link_points_base[:n]
    ee_base = trace.ee_points_base[:n] if trace.ee_points_base is not None else link_base[:, -1, :]
    mesh_trace = robot_mesh_trace_base
    if mesh_trace is not None:
        n = min(n, mesh_trace.link_poses_base.shape[0])
        base_poses = base_poses[:n]
        link_base = link_base[:n]
        ee_base = ee_base[:n]
        mesh_trace.link_poses_base = mesh_trace.link_poses_base[:n]
    sample_ids = np.arange(0, n, max(1, int(every_n)))
    if sample_ids[-1] != n - 1:
        sample_ids = np.append(sample_ids, n - 1)

    boxes, spheres, cylinders = parse_obstacles(problem)
    start_point_base = ee_base[0]
    goal_point_base = _goal_position_base(problem)
    scene_ranges = _scene_ranges(
        base_poses,
        boxes,
        spheres,
        cylinders,
        link_base,
        ee_base,
        goal_point_base,
        mesh_trace,
    )

    def frame_data(i: int) -> List[Any]:
        data: List[Any] = []
        # Obstacles and robot are expressed in base frame; animate by world<-base pose.
        for box in boxes:
            data.append(_box_trace(transform_pose(base_poses[i], box.pose_base), box.dims, name=box.name))
        for sphere in spheres:
            center = transform_points(base_poses[i], sphere.center_base)
            data.append(_sphere_trace(center, sphere.radius, name=sphere.name))
        for cyl in cylinders:
            data.append(_cylinder_trace(transform_pose(base_poses[i], cyl.pose_base), cyl.radius, cyl.height, name=cyl.name))
        if mesh_trace is not None:
            data.extend(_robot_mesh_traces(mesh_trace, base_poses[i], i))
        else:
            link_world = transform_points(base_poses[i], link_base[i])
            data.append(
                go.Scatter3d(
                    x=link_world[:, 0], y=link_world[:, 1], z=link_world[:, 2],
                    mode="lines+markers", line=dict(width=8), marker=dict(size=4),
                    name="robot_links",
                )
            )
        start_world = transform_points(base_poses[i], start_point_base)
        data.append(_point_trace(start_world, name="start", color="#2ca02c", size=6, symbol="circle"))
        if goal_point_base is not None:
            goal_world = transform_points(base_poses[i], goal_point_base)
            data.append(_point_trace(goal_world, name="goal", color="#d62728", size=7, symbol="diamond"))
        if show_frames:
            data.extend(_basis_traces(base_poses[i], scale=0.12, prefix="base"))
        if show_trajectory:
            ee_path_current_base = transform_points(base_poses[i], ee_base[: i + 1])
            data.append(
                go.Scatter3d(
                    x=ee_path_current_base[:, 0],
                    y=ee_path_current_base[:, 1],
                    z=ee_path_current_base[:, 2],
                    mode="lines",
                    line=dict(width=5),
                    name="ee_trajectory_base",
                )
            )
        return data

    frames = [go.Frame(data=frame_data(int(i)), name=str(int(i))) for i in sample_ids]
    fig = go.Figure(data=frame_data(int(sample_ids[0])), frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="cube",
            xaxis_title="world x [m]",
            yaxis_title="world y [m]",
            zaxis_title="world z [m]",
            xaxis=dict(range=scene_ranges[0], autorange=False),
            yaxis=dict(range=scene_ranges[1], autorange=False),
            zaxis=dict(range=scene_ranges[2], autorange=False),
            camera=dict(eye=dict(x=1.45, y=-1.75, z=1.15), up=dict(x=0, y=0, z=1)),
        ),
        uirevision="fixed_base_scene",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": int(frame_duration_ms), "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[dict(method="animate", args=[[str(int(i))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=str(int(i))) for i in sample_ids],
                currentvalue={"prefix": "step: "},
            )
        ],
    )
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn", auto_open=auto_open)
    if output_gif is not None:
        gif_frames, duration_scale = _select_gif_frames(frames, gif_every_n, gif_max_frames)
        _write_animation_gif(
            fig,
            gif_frames,
            output_gif,
            frame_duration_ms=max(1, int(round(float(frame_duration_ms) * duration_scale))),
            width=gif_width,
            height=gif_height,
            scale=gif_scale,
        )
    return output


def _select_gif_frames(
    frames: Sequence[Any],
    every_n: int,
    max_frames: int,
) -> Tuple[List[Any], float]:
    if not frames:
        return [], 1.0
    stride = max(1, int(every_n))
    selected = list(frames[::stride])
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])

    if max_frames and max_frames > 1 and len(selected) > max_frames:
        ids = np.linspace(0, len(selected) - 1, int(max_frames), dtype=int)
        selected = [selected[int(i)] for i in np.unique(ids)]
        if selected[-1] is not frames[-1]:
            selected[-1] = frames[-1]

    duration_scale = max(1.0, float(len(frames)) / float(max(1, len(selected))))
    return selected, duration_scale


def _write_animation_gif(
    fig: Any,
    frames: Sequence[Any],
    output_gif: str | Path,
    frame_duration_ms: int,
    width: int,
    height: int,
    scale: float,
) -> Path:
    import imageio.v2 as imageio
    import plotly.graph_objects as go

    output = Path(output_gif)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(float(frame_duration_ms) / 1000.0, 1.0 / 1000.0)

    layout = go.Layout(fig.layout)
    layout.updatemenus = None
    layout.sliders = None
    with imageio.get_writer(str(output), mode="I", duration=duration_s, loop=0) as writer:
        for frame in frames:
            frame_fig = go.Figure(data=frame.data, layout=layout)
            png = frame_fig.to_image(
                format="png",
                width=int(width),
                height=int(height),
                scale=float(scale),
            )
            writer.append_data(imageio.imread(io.BytesIO(png)))
    return output


def _box_corners(dims: np.ndarray) -> np.ndarray:
    hx, hy, hz = np.asarray(dims, dtype=float)[:3] / 2.0
    return np.array(
        [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )


def _box_trace(pose_world: np.ndarray, dims: np.ndarray, name: str) -> Any:
    import plotly.graph_objects as go

    pts = transform_points(pose_world, _box_corners(dims))
    faces = np.asarray(
        [
            [0, 1, 3], [0, 3, 2],
            [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4],
            [1, 5, 7], [1, 7, 3],
        ],
        dtype=int,
    )
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#9aa0a6", opacity=0.38, flatshading=True,
        name=f"obs:{name}",
    )


def _sphere_trace(center: np.ndarray, radius: float, name: str) -> Any:
    import plotly.graph_objects as go

    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 9)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        opacity=0.38,
        showscale=False,
        colorscale=[[0, "#9aa0a6"], [1, "#9aa0a6"]],
        name=f"obs:{name}",
    )


def _cylinder_trace(pose_world: np.ndarray, radius: float, height: float, name: str) -> Any:
    import plotly.graph_objects as go

    segments = 32
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    bottom = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, -height / 2.0)],
        axis=-1,
    )
    top = bottom.copy()
    top[:, 2] = height / 2.0
    local = np.vstack([bottom, top, [[0.0, 0.0, -height / 2.0], [0.0, 0.0, height / 2.0]]])
    world = transform_points(pose_world, local)
    bottom_center = 2 * segments
    top_center = bottom_center + 1
    faces: List[List[int]] = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([idx, nxt, segments + nxt])
        faces.append([idx, segments + nxt, segments + idx])
        faces.append([bottom_center, nxt, idx])
        faces.append([top_center, segments + idx, segments + nxt])
    faces_arr = np.asarray(faces, dtype=int)
    return go.Mesh3d(
        x=world[:, 0], y=world[:, 1], z=world[:, 2],
        i=faces_arr[:, 0], j=faces_arr[:, 1], k=faces_arr[:, 2],
        color="#9aa0a6", opacity=0.38, flatshading=True,
        name=f"obs:{name}",
    )


def _point_trace(point: np.ndarray, name: str, color: str, size: int, symbol: str) -> Any:
    import plotly.graph_objects as go

    point = np.asarray(point, dtype=float).reshape(3)
    return go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]],
        mode="markers",
        marker=dict(size=size, color=color, symbol=symbol),
        name=name,
    )


def _goal_position_base(problem: Mapping[str, Any]) -> Optional[np.ndarray]:
    goal_pose = problem.get("goal_pose", {})
    if isinstance(goal_pose, Mapping) and "position_xyz" in goal_pose:
        return np.asarray(goal_pose["position_xyz"], dtype=float)[:3]
    return None


def _scene_ranges(
    base_poses: np.ndarray,
    boxes: Sequence[BoxObstacle],
    spheres: Sequence[SphereObstacle],
    cylinders: Sequence[CylinderObstacle],
    link_base: np.ndarray,
    ee_base: np.ndarray,
    goal_point_base: Optional[np.ndarray],
    mesh_trace: Optional[RobotMeshTrace],
) -> Tuple[List[float], List[float], List[float]]:
    points: List[np.ndarray] = []
    base_poses = np.asarray(base_poses, dtype=float)
    link_base = np.asarray(link_base, dtype=float)
    ee_base = np.asarray(ee_base, dtype=float)
    sample_ids = np.unique(
        np.linspace(0, base_poses.shape[0] - 1, min(12, base_poses.shape[0]), dtype=int)
    )
    for frame_index in sample_ids:
        i = int(frame_index)
        points.append(transform_points(base_poses[i], link_base[i]))
        points.append(transform_points(base_poses[i], ee_base[: i + 1]))
        if goal_point_base is not None:
            points.append(transform_points(base_poses[i], goal_point_base).reshape(1, 3))

    for box in boxes:
        corners_base = transform_points(box.pose_base, _box_corners(box.dims))
        for frame_index in sample_ids:
            points.append(transform_points(base_poses[int(frame_index)], corners_base))
    for sphere in spheres:
        center = np.asarray(sphere.center_base, dtype=float)
        radius = float(sphere.radius)
        bounds_base = np.asarray(
            [
                center + [radius, 0.0, 0.0],
                center - [radius, 0.0, 0.0],
                center + [0.0, radius, 0.0],
                center - [0.0, radius, 0.0],
                center + [0.0, 0.0, radius],
                center - [0.0, 0.0, radius],
            ],
            dtype=float,
        )
        for frame_index in sample_ids:
            points.append(transform_points(base_poses[int(frame_index)], bounds_base))
    for cyl in cylinders:
        bounds_base = transform_points(cyl.pose_base, _cylinder_bounds(cyl.radius, cyl.height))
        for frame_index in sample_ids:
            points.append(transform_points(base_poses[int(frame_index)], bounds_base))

    if mesh_trace is not None:
        for frame_index in sample_ids:
            for link_index, link in enumerate(mesh_trace.links):
                i = int(min(frame_index, mesh_trace.link_poses_base.shape[0] - 1))
                pose_world = base_poses[i] @ mesh_trace.link_poses_base[i, link_index] @ link.local_pose
                points.append(transform_points(pose_world, _mesh_bounds(link.vertices)))

    all_points = np.concatenate(points, axis=0)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.55, 0.35)
    return (
        [float(center[0] - radius), float(center[0] + radius)],
        [float(center[1] - radius), float(center[1] + radius)],
        [float(center[2] - radius), float(center[2] + radius)],
    )


def _mesh_bounds(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return np.asarray(
        [[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])],
        dtype=float,
    )


def _cylinder_bounds(radius: float, height: float) -> np.ndarray:
    return np.asarray(
        [
            [radius, radius, height / 2.0],
            [radius, radius, -height / 2.0],
            [radius, -radius, height / 2.0],
            [radius, -radius, -height / 2.0],
            [-radius, radius, height / 2.0],
            [-radius, radius, -height / 2.0],
            [-radius, -radius, height / 2.0],
            [-radius, -radius, -height / 2.0],
        ],
        dtype=float,
    )


def _robot_mesh_traces(
    mesh_trace: RobotMeshTrace,
    base_pose_world: np.ndarray,
    frame_index: int,
) -> List[Any]:
    import plotly.graph_objects as go

    traces: List[Any] = []
    for link_index, link in enumerate(mesh_trace.links):
        link_pose_world = base_pose_world @ mesh_trace.link_poses_base[frame_index, link_index]
        visual_pose_world = link_pose_world @ link.local_pose
        vertices_world = transform_points(visual_pose_world, link.vertices)
        traces.append(
            go.Mesh3d(
                x=vertices_world[:, 0],
                y=vertices_world[:, 1],
                z=vertices_world[:, 2],
                i=link.faces[:, 0],
                j=link.faces[:, 1],
                k=link.faces[:, 2],
                color="#4c78a8",
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.55, diffuse=0.75, specular=0.25, roughness=0.45),
                name=f"robot:{link.name}",
                showscale=False,
            )
        )
    return traces


def _basis_traces(pose_world: np.ndarray, scale: float, prefix: str) -> List[Any]:
    import plotly.graph_objects as go

    origin = pose_world[:3, 3]
    axes = pose_world[:3, :3] * scale
    names = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    return [
        go.Scatter3d(
            x=[origin[0], origin[0] + axes[0, j]],
            y=[origin[1], origin[1] + axes[1, j]],
            z=[origin[2], origin[2] + axes[2, j]],
            mode="lines",
            line=dict(width=5),
            name=names[j],
            showlegend=False,
        )
        for j in range(3)
    ]


__all__ = [
    "RobotTrace",
    "RobotMeshTrace",
    "make_base_motion_poses_from_velocity",
    "make_random_base_motion_poses",
    "extract_robot_trace_from_curobo",
    "extract_robot_mesh_trace_from_curobo",
    "parse_obstacles",
    "render_moving_base_scene",
]
