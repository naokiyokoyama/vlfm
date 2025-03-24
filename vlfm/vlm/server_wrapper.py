# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import contextlib
import fcntl
import os
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        return jsonify(model.process_payload(payload))

    @app.route(f"/{name}/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy"}), 200

    app.run(host="0.0.0.0", port=port)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    return _send_request(url, **kwargs)


def _send_request(url: str, **kwargs: Any) -> dict:
    # Create a payload dict which is a clone of kwargs but all np.array values are
    # converted to strings
    payload = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
        else:
            payload[k] = v

    # Set the headers
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code == 200:
        return resp.json()

    raise requests.RequestException(
        f"Request failed with status code {resp.status_code}"
    )


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


@contextlib.contextmanager
def cuda_lock(lock_file: str = "/tmp/cuda_lock.lock", timeout=60):
    """
    A context manager that acquires a lock before running CUDA operations.

    Args:
        lock_file: Path to the lock file
        timeout: Maximum time to wait for the lock in seconds

    Yields:
        None

    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout period
    """
    start_time = time.time()

    # Create the lock file if it doesn't exist
    if not os.path.exists(lock_file):
        with open(lock_file, "w"):
            pass

    # Try to acquire the lock
    lock_fd = open(lock_file, "r+")
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError:  # Another process has the lock
            if time.time() - start_time > timeout:
                lock_fd.close()
                raise TimeoutError(
                    f"Could not acquire CUDA lock within {timeout} seconds"
                )
            time.sleep(0.1)  # Wait a bit before retrying

    try:
        # Do CUDA operations with the lock held
        yield
    finally:
        # Release the lock
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


class CUDALockMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        prefix = os.environ.get("SLURM_JOBID", "0")
        lock_file = f"/tmp/{prefix}_cuda_lock.lock"

        torch.cuda.empty_cache()
        with cuda_lock(lock_file=lock_file, timeout=10):
            return super().inference(*args, **kwargs)
