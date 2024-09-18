from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

from frontier_exploration.utils.frontier_filtering import FrontierInfo, filter_frontiers
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import extract_yaw


class ObstacleMapV2(ObstacleMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontier_infos: List[Tuple[int, FrontierInfo]] = []
        self._f_position_to_f_info: Dict[Tuple[int, int], FrontierInfo] = {}

    def reset(self) -> None:
        super().reset()
        self.frontier_infos = []
        self._f_position_to_f_info = {}

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
        yaw = extract_yaw(tf_camera_to_episodic)
        agent_xy_location = tf_camera_to_episodic[:2, 3].reshape(1, 2)
        agent_pixel_location = tuple(self._xy_to_px(agent_xy_location).reshape(2))

        frontier_infos = []
        # keys_to_del = set(self._f_position_to_f_info.keys())

        for f_px in self._frontiers_px:
            if tuple(f_px) not in self._f_position_to_f_info:
                f_info = FrontierInfo(
                    camera_position_px=agent_pixel_location,
                    frontier_position_px=tuple(f_px),
                    single_fog_of_war=self._new_explored_area,
                    agent_pose=[0, 0, 0, yaw],  # only yaw; we have camera_position_px
                    frontier_position=None,
                    rgb_img=rgb,
                )
                self._f_position_to_f_info[f_px] = f_info
            else:
                f_info = self._f_position_to_f_info[f_px]
                # keys_to_del.remove(f_px)

            frontier_infos.append(f_info)

        # for k in keys_to_del:
        #     del self._f_position_to_f_info[k]

        boundary_contour = cv2.findContours(
            self.explored_area, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        inds_to_keep = filter_frontiers(frontier_infos, boundary_contour)

        self.frontier_infos = [(i, self.frontier_infos[i]) for i in inds_to_keep]
