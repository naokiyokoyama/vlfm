from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from gym import Space, spaces
from habitat import Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


class StringArrayConverter:
    def __init__(self, max_length: int) -> None:
        """
        Initialize the converter with maximum string length.
        Will create a 1D array of that length.

        Args:
            max_length: Maximum length of input strings to be converted
                        to fixed-size arrays.

        Attributes:
            max_length: Maximum length of strings that can be converted.
            shape: Shape tuple of the output numpy array.
        """
        self.max_length: int = max_length
        self.shape: Tuple[int, ...] = (max_length,)

    def string_to_array(self, input_string: str) -> np.ndarray:
        """
        Convert a string to a 1D numpy array of ASCII values.

        Args:
            input_string: String to convert to a numpy array.

        Returns:
            np.ndarray: 1D array of ASCII values with padding zeros.

        Raises:
            ValueError: If input string length exceeds maximum length.
        """
        if len(input_string) > self.max_length:
            raise ValueError(
                f"Input string length {len(input_string)} exceeds maximum length"
                f" {self.max_length}: {input_string}"
            )

        ascii_values: List[int] = [ord(c) for c in input_string]
        padded_values: List[int] = ascii_values + [0] * (
            self.max_length - len(ascii_values)
        )
        return np.array(padded_values, dtype=np.uint8)

    def array_to_string(self, array: np.ndarray) -> str:
        """
        Convert a numpy array back to the original string.

        Args:
            array: Numpy array of ASCII values to convert back to string.

        Returns:
            str: The reconstructed string from the array.

        Raises:
            ValueError: If the array shape doesn't match the expected shape.
        """
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
        self, sim: HabitatSim, config: DictConfig, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize the TourSensor which provides episode identifiers as fixed-length arrays.

        Args:
            sim: The simulator instance.
            config: Configuration parameters including max_tour_length.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Initialize the string converter with a reasonable max length
        # Assuming scene_id and episode_id won't exceed 30 chars total when combined
        self.converter: StringArrayConverter = StringArrayConverter(max_length=30)
        self.max_tour_length: int = config.max_tour_length

        super().__init__(sim, config, *args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        """
        Get the unique identifier for this sensor.

        Returns:
            str: The sensor's UUID.
        """
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        """
        Get the sensor type.

        Returns:
            SensorTypes: The type of this sensor (TENSOR).
        """
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        """
        Define the observation space for this sensor.

        Returns:
            Space: Box space matching the converter's array shape.
        """
        # Return a space matching our converter's array shape
        return spaces.Box(
            low=0,
            high=255,  # Maximum ASCII value
            shape=self.converter.shape,
            dtype=np.uint8,
        )

    def get_observation(self, *args: Any, episode: Any, **kwargs: Any) -> np.ndarray:
        """
        Convert episode identifier into a fixed-length numpy array.

        Args:
            episode: The current episode object containing scene_id and episode_id.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Array representation of the episode identifier.
        """
        # Create the identifier string
        scene_id = Path(episode.scene_id).stem.replace(".basis", "")
        identifier = f"{scene_id}_{episode.episode_id}_{self.max_tour_length}"

        # Convert to array and return
        return self.converter.string_to_array(identifier)


@dataclass
class TourSensorConfig(LabSensorConfig):
    """
    Configuration class for the TourSensor.

    Attributes:
        type: The type of sensor (should match the class name).
        max_tour_length: Maximum length of tour sequences.
    """

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
