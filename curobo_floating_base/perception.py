# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perception utilities module.

This module provides sensor processing utilities for robot perception tasks,
including robot segmentation and volumetric mapping from depth images.

Example (robot segmentation):
    ```python
    from curobo_floating_base.perception import RobotSegmenter
    from curobo_floating_base.kinematics import KinematicsCfg
    from curobo_floating_base.types import CameraObservation, JointState

    # Create segmenter
    segmenter = RobotSegmenter.from_robot_file(
        robot_file="franka.yml",
        distance_threshold=0.05,
    )

    # Segment robot from depth image
    camera_obs = CameraObservation(
        depth_image=depth_tensor,
        intrinsics=camera_intrinsics,
        pose=camera_pose,
    )
    joint_state = JointState(position=current_joint_angles)

    mask, filtered_depth = segmenter.get_robot_mask(camera_obs, joint_state)
    # mask: binary mask of robot pixels
    # filtered_depth: depth image with robot removed
    ```

Example (volumetric mapping):
    ```python
    from curobo_floating_base.perception import FilterDepth, Mapper, MapperCfg

    mapper = Mapper(MapperCfg(voxel_size=0.02, ...))
    depth_filter = FilterDepth(image_shape=depth.shape[-2:])
    filtered, valid = depth_filter(depth.unsqueeze(0))
    mapper.integrate(camera_obs)
    ```

Example (robot pose estimation):
    ```python
    from curobo_floating_base.perception import (
        DetectorCfg, PoseDetector, RobotMesh, SDFDetectorCfg, SDFPoseDetector,
    )

    mesh = RobotMesh.from_kinematics(kinematics, joint_state)
    detector = PoseDetector(DetectorCfg(...), mesh)
    pose = detector.detect(pointcloud)
    ```
"""

from curobo_floating_base._src.perception.filter_depth import FilterDepth
from curobo_floating_base._src.perception.mapper import Mapper, MapperCfg
from curobo_floating_base._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo_floating_base._src.perception.pose_estimation.pose_detector import PoseDetector
from curobo_floating_base._src.perception.pose_estimation.pose_detector_cfg import DetectorCfg
from curobo_floating_base._src.perception.pose_estimation.sdf_pose_detector import SDFPoseDetector
from curobo_floating_base._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo_floating_base._src.perception.robot_segmenter import RobotSegmenter

__all__ = [
    "DetectorCfg",
    "FilterDepth",
    "Mapper",
    "MapperCfg",
    "PoseDetector",
    "RobotMesh",
    "RobotSegmenter",
    "SDFDetectorCfg",
    "SDFPoseDetector",
]
