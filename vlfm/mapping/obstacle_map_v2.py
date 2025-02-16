from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from frontier_exploration.utils.frontier_filtering import (
    CallCounter,
    FrontierFilter,
    FrontierFilterData,
)
from frontier_exploration.utils.segment_monitor import get_action

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import extract_yaw


class ObstacleMapV2(ObstacleMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_filtering = True
        self.selected_waypoint: np.ndarray = np.full((3,), np.nan)
        self._bad_idx_to_good_idx: Dict[int, int] = {}
        self.frontier_filter: Optional[FrontierFilter] = None
        self.call_count: int = 0
        self.rgb_images: List[np.ndarray] = []
        self.frontier_rgb_waypoints: List[FrontierRGBWaypoint] = []

    def initialize_filter(self, fov, visibility_dist_in_pixels):
        self.frontier_filter = FrontierFilter(fov, visibility_dist_in_pixels)

    def reset(self) -> None:
        super().reset()
        self.selected_waypoint = np.full((3,), np.nan)
        if self.frontier_filter is not None:
            self.frontier_filter.reset()
        self._bad_idx_to_good_idx = {}
        self.call_count = 0
        self.rgb_images = []
        self.frontier_rgb_waypoints = []

    @property
    def _frontier_segments_yx(self) -> List[np.ndarray]:
        curr_f_segments = []
        for fs in self._frontier_segments:
            # Swap the x and y coordinates
            new_fs = np.zeros((fs.shape[0], 2), dtype=np.int32)
            new_fs[:, 0] = fs[:, 1]
            new_fs[:, 1] = fs[:, 0]
            curr_f_segments.append(new_fs)
        return curr_f_segments

    @property
    def frontiers_px(self):
        return self._frontiers_px

    @CallCounter
    def update_map(
        self,
        rgb: np.ndarray,
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
        super().update_map(
            depth,
            tf_camera_to_episodic,
            min_depth,
            max_depth,
            fx,
            fy,
            topdown_fov,
            explore,
            update_obstacles,
        )
        self.rgb_images.append(rgb)
        agent_xy_location = tf_camera_to_episodic[:2, 3].reshape(1, 2)
        agent_pixel_location = self._xy_to_px(agent_xy_location).reshape(2)[::-1]

        if self.frontier_filter is None:
            self.initialize_filter(
                np.degrees(topdown_fov), int(max_depth * self.pixels_per_meter * 2)
            )

        result: FrontierFilterData = (
            self.frontier_filter.score_and_filter_frontiers(
                curr_f_segments=self._frontier_segments_yx,
                curr_cam_yaw=-extract_yaw(tf_camera_to_episodic),
                curr_cam_pos=agent_pixel_location,
                top_down_map=self._navigable_map.astype(np.uint8),
                curr_timestep_id=self.call_count,
                filter=self._use_filtering,
            )
        )

        d = result.filtered if self._use_filtering else result.unfiltered
        self._bad_idx_to_good_idx = d.bad_idx_to_good_idx

        self.frontier_rgb_waypoints = [
            FrontierRGBWaypoint(
                rgb=self.rgb_images[t_step], waypoint=self.frontiers[f_idx]
            )
            for f_idx, t_step in d.good_indices_to_timestep.items()
        ]

    def get_action(
        self,
        tf_camera_to_episodic: np.ndarray,
        topdown_fov: float,
        max_depth: float,
        turn_angle: float = np.radians(30),
    ) -> np.ndarray:
        agent_xy_location = tf_camera_to_episodic[:2, 3].reshape(1, 2)
        agent_pixel_location = tuple(self._xy_to_px(agent_xy_location).reshape(2))
        action = get_action(
            frontier_segments=self._frontier_segments_yx,
            obstacle_map=self._navigable_map.astype(np.uint8),
            camera_pos=agent_pixel_location[::-1],
            camera_yaw=-extract_yaw(tf_camera_to_episodic),
            fov=np.degrees(topdown_fov),
            max_line_len=int(max_depth * self.pixels_per_meter),
            turn_angle=turn_angle,
        )
        return action

    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = super().visualize()
        vis_img = cv2.flip(vis_img, 0)

        # Draw the frontier segments in blue, and the selected frontier in pink;
        # also draw the frontier midpoint in white. Alter the midpoint's color based on
        # whether it is active or not (light gray or black)
        for idx, waypoint in enumerate(self._frontiers_px):
            boundary_color = (
                (255, 0, 255)  # Pink
                if np.array_equal(self.frontiers[idx], self.selected_waypoint)
                else (0, 0, 255)  # Blue
            )

            if idx in self._bad_idx_to_good_idx:
                frontier_midpoint_color = (200, 200, 200)
            else:
                frontier_midpoint_color = (0, 0, 0)

            if np.array_equal(self.frontiers[idx], self.selected_waypoint):
                frontier_midpoint_color = (255, 0, 255)

            # Draw frontier segments forming its boundary
            p_px = self._frontier_segments[idx].reshape((-1, 1, 2))
            cv2.polylines(
                vis_img, [p_px], isClosed=False, color=boundary_color, thickness=2
            )

            # Draw frontier midpoint
            cv2.circle(
                vis_img,
                waypoint.astype(np.int32),
                5,
                (255, 255, 255),
                -1,
            )
            cv2.circle(
                vis_img,
                waypoint.astype(np.int32),
                5,
                frontier_midpoint_color,
                2,
            )

        # Visualize which frontiers were filtered out by which active frontier
        for bad_idx, good_idx in self._bad_idx_to_good_idx.items():
            try:
                bad_waypoint = self._frontiers_px[bad_idx].astype(np.int32)
                good_waypoint = self._frontiers_px[good_idx].astype(np.int32)
            except IndexError:
                continue

            cv2.line(vis_img, bad_waypoint, good_waypoint, (0, 0, 255), 1)

        vis_img = cv2.flip(vis_img, 0)

        return vis_img


@dataclass(frozen=True)
class FrontierRGBWaypoint:
    rgb: np.ndarray
    waypoint: np.ndarray
