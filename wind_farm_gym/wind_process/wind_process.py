from abc import ABC, abstractmethod
from typing import Dict


class WindProcess(ABC):
    """
    WindProcess is an abstract class representing a discrete-time (stochastic) process for wind simulation.

    A wind process implements three methods:
    `step`: returns the next time step value; the value can be of any type, but the WindProcess expects a dictionary
    of atmospheric measurements, for example, {'wind_speed': 7.0, 'wind_direction': 270.0}
    `reset`: resets the process so that it can start anew from the beginning
    `close`: finalizes the wind process; this method is called from WindFarmEnv.close()
    """

    @abstractmethod
    def step(self) -> Dict[str, float]:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass
