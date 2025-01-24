# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

# The following imports require habitat to be installed, and despite not being used by
# this script itself, will register several classes and make them discoverable by Hydra.
# This run.py script is expected to only be used when habitat is installed, thus they
# are hidden here instead of in an __init__.py file. This avoids import errors when used
# in an environment without habitat, such as when doing real-world deployment. noqa is
# used to suppress the unused import and unsorted import warnings by ruff.
import frontier_exploration  # noqa
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)

import time
import warnings
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Union

import cv2, random, string
import numpy as np
import requests
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from torch import Tensor

from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.habitat_policies import HabitatMixin

warnings.filterwarnings("ignore")


def cobra_request(
    episode_identifier: str,
    object_category: str,
    images: np.ndarray,
    video_hash: str,
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
        video_hash (str): A hash string for the video.
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
        cobra_request.server_is_ready = wait_for_server(f"{server_url}/health")

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
        "video_hash": video_hash,
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
        print(f"Waiting for server to become ready... {int(seconds_remaining)}s remaining")
    print(f"Server did not become ready within {timeout} seconds")
    return False


class CobraPolicy(BaseObjectNavPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._video_hash: str = ""
        self._first_transmission: bool = True
        self._obstacle_map._use_filtering = False
        self._episode_identifier: str = ''.join(
            random.choices(string.ascii_letters, k=8)
        )

    def _reset(self) -> None:
        super()._reset()
        # self._done_initializing = True  # Always True for CobraPolicy
        self._video_hash = str(int(time.time() * 10000))
        self._first_transmission = True
        self._episode_identifier = ''.join(random.choices(string.ascii_letters, k=8))

    def act(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        ret = super().act(*args, **kwargs)
        return ret

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        if len(self._obstacle_map.frontier_infos) == 0:
            # Undefined behavior. Return STOP action.
            return torch.tensor(0, dtype=torch.long).reshape(1, 1)
        images = np.stack([f[1].rgb_img for f in self._obstacle_map.frontier_infos])
        if len(images) < 2:
            print(f"Not enough frontier groups to choose from ({len(images)}).")
            predicted_choice_index = 0
        else:
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

            cobra_port = os.environ.get("COBRA_PORT", "5000")
            predicted_choice_index = cobra_request(
                episode_identifier=self._episode_identifier,
                object_category=self._target_object,
                images=images,
                video_hash=self._video_hash,
                video=video,
                server_url=f"http://127.0.0.1:{cobra_port}",
            )
            self._first_transmission = False

        predicted_frontier_index, _ = self._obstacle_map.frontier_infos[predicted_choice_index]
        best_frontier = self._obstacle_map.frontiers[predicted_frontier_index]
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action


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

def count_json_files(directory_path):
    json_count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            json_count += 1
    return json_count
@baseline_registry.register_policy
class HabitatCobraPolicy(HabitatMixin, CobraPolicy):
    pass


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/vlfm_objectnav_hm3d",
)
def main(cfg: DictConfig) -> None:
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)

    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
