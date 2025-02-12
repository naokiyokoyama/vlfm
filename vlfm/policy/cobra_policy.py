import os
import random
import string
import time
import warnings
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import requests
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from torch import Tensor

from vlfm.mapping.obstacle_map_v2 import FrontierRGBWaypoint
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.habitat_policies import HabitatMixin, TorchActionIDs

warnings.filterwarnings("ignore")


class CobraPolicy(BaseObjectNavPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._first_transmission: bool = True
        self._obstacle_map._use_filtering = os.environ["USE_FILTERING"] == "1"
        self._episode_identifier: str = "".join(
            random.choices(string.ascii_letters, k=8)
        )
        self._cobra_port = os.environ.get("COBRA_PORT", "5000")
        wait_for_server(f"http://127.0.0.1:{self._cobra_port}/health", timeout=500)

    def _reset(self) -> None:
        super()._reset()
        self._first_transmission = True
        self._episode_identifier = "".join(random.choices(string.ascii_letters, k=8))

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        if len(self._obstacle_map.frontiers) == 0:
            # Undefined behavior. Return STOP action.
            return torch.tensor(0, dtype=torch.long).reshape(1, 1)

        tf_camera_to_episodic = self._observations_cache["object_map_rgbd"][0][2]
        max_depth = self._observations_cache["object_map_rgbd"][0][4]
        pointnav_action = self._obstacle_map.get_action(
            tf_camera_to_episodic, self._camera_fov, max_depth
        )
        if pointnav_action is not None:
            self._send_only_current()
            if pointnav_action:
                return TorchActionIDs.TURN_LEFT
            else:
                return TorchActionIDs.TURN_RIGHT

        # Retrieve video
        if not self._first_transmission or os.environ.get("NO_EXPLORATION", "0") == "1":
            video = None
        else:
            # video = self._observations_cache["video"]
            scene_id, ep_id = os.environ["scene_ep_id"].split(",")
            scene_id = os.path.basename(scene_id).split(".")[0]
            video_path = (
                "/mnt/scale_vln_data/exploration_episodes_v3/"
                f"{scene_id}/episode_{ep_id}/"
                "exploration_imgs_0/exploration.mp4"
            )
            video = video_to_numpy(video_path)
            print(f"Video path: {video_path}")
            print(f"Video shape: {video.shape}")

        # Retrieve the images of all the frontiers from the obstacle map
        frontier_rgb_waypoints: List[FrontierRGBWaypoint] = (
            self._obstacle_map.frontier_rgb_waypoints
        )
        frontier_images = [f.rgb for f in frontier_rgb_waypoints]
        all_images = np.stack(
            [self._observations_cache["object_map_rgbd"][0][0]] + frontier_images
        )

        pred_idx = self._send_request(all_images, video)
        self._obstacle_map.selected_f_idx = pred_idx

        return self._pointnav(frontier_rgb_waypoints[pred_idx].waypoint, stop=False)

    def _send_request(
        self, images: np.ndarray, video: Optional[np.ndarray] = None
    ) -> int:
        predicted_choice_index = cobra_request(
            episode_identifier=self._episode_identifier,
            object_category=self._target_object,
            images=images,
            video=video,
            server_url=f"http://127.0.0.1:{self._cobra_port}",
        )
        self._first_transmission = False

        return predicted_choice_index

    def _send_only_current(self) -> None:
        curr = self._observations_cache["object_map_rgbd"][0][0]
        images = np.stack([curr, curr])
        self._send_request(images)


def video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    cap.release()
    video_array = np.array(frames)

    return video_array


@baseline_registry.register_policy
class HabitatCobraPolicy(HabitatMixin, CobraPolicy):
    def _initialize(self: HabitatMixin) -> None:
        self._send_only_current()
        return super()._initialize()


def cobra_request(
    episode_identifier: str,
    object_category: str,
    images: np.ndarray,
    video: Optional[np.ndarray] = None,
    server_url: str = "http://localhost:5000",
) -> int:
    """
    Send input data to the CobraObjNav server and return the selected choice as an
    integer. Uses numpy serialization for efficiency.

    Args:
        episode_identifier (str): A unique identifier for the episode.
        object_category (str): The category of the object to find.
        images (np.ndarray): A numpy array of shape (N, H, W, C) representing the choice
                             images.
        video (np.ndarray, optional): A numpy array of shape (M, H, W, C) representing
                                      the video frames.
                                      If None, an empty array will be sent.
        server_url (str, optional): The URL of the server. Defaults to
                                    "http://localhost:5000/select_choice".

    Returns:
        int: The index of the selected choice.

    Raises:
        requests.RequestException: If there's an error in the HTTP request.
        ValueError: If the server returns an unexpected response.
    """

    if not hasattr(cobra_request, "server_is_ready"):
        cobra_request.server_is_ready = wait_for_server(
            f"{server_url}/health", timeout=500
        )

    shared_memories = []

    # Serialize numpy arrays
    def memory_pointer(data) -> str:
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        # Create a NumPy array backed by shared memory
        shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shared_array[:] = data[:]  # Copy the data to shared memory
        shared_memories.append(shm)
        return shm.name

    # Prepare the payload
    payload = {
        "episode_identifier": episode_identifier,
        "object_category": object_category,
        "images": memory_pointer(images),
        "images_shape": images.shape,
        "video": memory_pointer(video) if video is not None else None,
        "video_shape": video.shape if video is not None else None,
    }

    try:
        # Send POST request to the server
        response = requests.post(f"{server_url}/select_choice", json=payload)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON response
        result = response.json()

        # Check if the 'choice' key exists in the response
        if "choice" not in result:
            raise ValueError("Unexpected response from server: 'choice' not found")

        # Return the choice as an integer
        return int(result["choice"])

    except requests.RequestException as e:
        # Handle any errors that occurred during the request
        print(f"Error communicating with the server: {e}")
        raise

    except ValueError as e:
        # Handle unexpected response format
        print(f"Error parsing server response: {e}")
        raise
    finally:
        # Clean up shared memory
        for shm in shared_memories:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


def wait_for_server(url, timeout=120, interval=1):
    """
    Wait for the server to become ready.

    :param url: The URL of the server's health check endpoint
    :param timeout: Maximum time to wait (in seconds)
    :param interval: Time between attempts (in seconds)
    :return: True if the server is ready, False if it timed out
    """
    start_time = time.time()
    print(f"Waiting for server at {url} to become ready...")
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Server at {url} is ready!")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
        seconds_remaining = timeout - (time.time() - start_time)
        print(
            f"Waiting for server to become ready... {int(seconds_remaining)}s remaining"
        )
    print(f"Server did not become ready within {timeout} seconds")
    return False
