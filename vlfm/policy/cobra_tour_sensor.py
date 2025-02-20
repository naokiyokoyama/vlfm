from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat import Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pathlib import Path



class StringArrayConverter:
    def __init__(self, max_length):
        """
        Initialize the converter with maximum string length.
        Will create a 1D array of that length.

        Args:
            max_length (int): Maximum length of input strings
        """
        self.max_length = max_length
        self.shape = (max_length,)

    def string_to_array(self, input_string):
        """Convert a string to a 1D numpy array."""
        if len(input_string) > self.max_length:
            raise ValueError(
                f"Input string length {len(input_string)} exceeds maximum length"
                f" {self.max_length}: {input_string}"
            )

        ascii_values = [ord(c) for c in input_string]
        padded_values = ascii_values + [0] * (self.max_length - len(ascii_values))
        return np.array(padded_values, dtype=np.uint8)

    def array_to_string(self, array):
        """Convert a numpy array back to the original string."""
        if array.shape != self.shape:
            raise ValueError(
                f"Input array shape {array.shape} does not match expected shape"
                f" {self.shape}"
            )

        chars = [chr(int(val)) for val in array if val > 0]
        return "".join(chars)


@registry.register_sensor
class TourSensor(Sensor):
    cls_uuid: str = "tour_sensor"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        # Initialize the string converter with a reasonable max length
        # Assuming scene_id and episode_id won't exceed 30 chars total when combined
        self.converter = StringArrayConverter(max_length=30)
        self.max_tour_length = config.max_tour_length

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
        """Convert episode identifier into a fixed-length numpy array."""
        # Create the identifier string
        scene_id = Path(episode.scene_id).stem.replace(".basis", "")
        identifier = f"{scene_id}_{episode.episode_id}_{self.max_tour_length}"

        # Convert to array and return
        return self.converter.string_to_array(identifier)


@dataclass
class TourSensorConfig(LabSensorConfig):
    type: str = TourSensor.__name__
    max_tour_length: int = 300


# Register the config
cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{TourSensor.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{TourSensor.cls_uuid}",
    node=TourSensorConfig,
)
