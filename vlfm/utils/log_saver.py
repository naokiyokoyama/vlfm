# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
import time
from typing import Dict, Union


def log_episode(episode_id: Union[str, int], scene_id: str, data: Dict) -> None:
    if data.get("ignore", 0) > 0:
        return

    log_dir = os.environ["ZSOS_LOG_DIR"]
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    base = f"{episode_id}_{scene_id}.json"
    filename = os.path.join(log_dir, base)

    # Skip if the filename already exists AND it isn't empty
    if not (os.path.exists(filename) and is_json_completed(filename)):
        print(f"Logging episode {int(episode_id):04d} to {filename}")
        with open(filename, "w") as f:
            json.dump(
                {"episode_id": episode_id, "scene_id": scene_id, **data}, f, indent=4
            )

def is_json_completed(json_file: str) -> bool:
    try:
        if os.path.getsize(json_file) == 0:
            return False
        with open(json_file, 'r') as f:
            content = f.read().strip()
            if content == "" or content == "{}" or "spl" not in content:
                return False
    except:
        return False
    return True

def is_evaluated(
    episode_id: Union[str, int],
    scene_id: str,
    ignore_stale: bool = False,
    assert_complete: bool = False,
) -> bool:
    log_dir = os.environ["ZSOS_LOG_DIR"]
    base = f"{episode_id}_{scene_id}.json"
    filename = os.path.join(log_dir, base)

    # Return false if the directory doesn't exist
    if not os.path.exists(log_dir):
        return False

    if not ignore_stale:
        # Delete any empty files that are older than 3 minutes
        try:
            for f in os.listdir(log_dir):
                try:
                    if os.path.getsize(os.path.join(log_dir, f)) == 0 and (
                        time.time() - os.path.getmtime(os.path.join(log_dir, f)) > 3 * 60
                    ):
                        os.remove(os.path.join(log_dir, f))
                except:
                    pass
        except:
            pass

    if assert_complete:
        if not os.path.exists(filename):
            return False

        with open(filename) as f:
            data = f.read()
        return "success" in data and "spl" in data

    return os.path.exists(filename)
