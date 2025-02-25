from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat import Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from vlfm.policy.cobra_tour_sensor import StringArrayConverter


@registry.register_sensor
class OpenVocabGoalSensor(Sensor):
    cls_uuid: str = ObjectGoalSensor.cls_uuid

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        # Initialize the string converter with max length for object category names
        self.converter = StringArrayConverter(max_length=80)
        super().__init__(sim, config, *args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        # Return a space matching our converter's array shape
        return spaces.Box(
            low=0,
            high=255,  # Maximum ASCII value
            shape=self.converter.shape,
            dtype=np.uint8,
        )

    def get_observation(self, *args: Any, episode, **kwargs: Any) -> np.ndarray:
        """Convert episode's object category into a fixed-length numpy array."""
        # Get the object category from the episode
        object_category = episode.object_category

        # Convert to array and return
        return self.converter.string_to_array(object_category)


@dataclass
class OpenVocabGoalSensorConfig(LabSensorConfig):
    type: str = OpenVocabGoalSensor.__name__


# Register the config
cs = ConfigStore.instance()
cs.store(
    package="habitat.task.lab_sensors.open_vocab_goal_sensor",
    group="habitat/task/lab_sensors",
    name="open_vocab_goal_sensor",
    node=OpenVocabGoalSensorConfig,
)
