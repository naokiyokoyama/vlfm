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
import ovon  # noqa: F401
from habitat import get_config  # noqa
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import HabitatBaselinesRLConfig
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig
from habitat.config import read_write

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.cobra_policy  # noqa: F401
import vlfm.policy.cobra_tour_sensor  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.policy.open_vocab_goal_sensor  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401

cs = ConfigStore.instance()
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=HabitatBaselinesRLConfig(),
)


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)


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

    if "tour_sensor" in cfg.habitat.task.lab_sensors and "MAX_LENGTH" in os.environ:
        with read_write(cfg):
            cfg.habitat.task.lab_sensors.tour_sensor.max_tour_length = int(
                os.environ["MAX_LENGTH"]
            )

    assert cfg.habitat_baselines.evaluate, "Only evaluation is supported."
    execute_exp(cfg, "eval")


if __name__ == "__main__":
    main()
