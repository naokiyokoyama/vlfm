# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.composite_fow import CompositeFOWMixin
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes


class ObstacleMap(CompositeFOWMixin, BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _map_dtype: np.dtype = np.dtype(bool)
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self.frontiers: np.ndarray = np.array([])
        self._frontiers_px: np.ndarray = np.array([])
        self._frontier_segments: List[np.ndarray] = []
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    @property
    def explored_area(self) -> np.ndarray:
        return self.full_mask

    def reset(self) -> None:
        super().reset()
        self._navigable_map.fill(0)
        self.frontiers = np.array([])
        self._frontiers_px = np.array([])
        self._frontier_segments = []

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            point_cloud_episodic_frame = transform_points(
                tf_camera_to_episodic, point_cloud_camera_frame
            )
            obstacle_cloud = filter_points_by_height(
                point_cloud_episodic_frame, self._min_height, self._max_height
            )

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3].reshape(1, 2)
        agent_pixel_location = self._xy_to_px(agent_xy_location).reshape(2)[::-1]
        self.add_fow(
            fow=reveal_fog_of_war(
                top_down_map=self._navigable_map.astype(np.uint8),
                current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
                current_point=agent_pixel_location,
                current_angle=-extract_yaw(tf_camera_to_episodic),
                fov=np.rad2deg(topdown_fov),
                max_line_len=max_depth * self.pixels_per_meter,
            ),
            fow_position=agent_pixel_location,
            fow_yaw=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            fov_length_px=max_depth * self.pixels_per_meter,
            obstacle_map=self._navigable_map.astype(np.uint8),
        )

        # Compute frontier locations
        self._update_frontiers()

    def _update_frontiers(self) -> None:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        self._frontiers_px, self._frontier_segments = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            self.explored_area.astype(np.uint8),
            self._area_thresh_in_pixels,
        )
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_height(
    points: np.ndarray, min_height: float, max_height: float
) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]
