from __future__ import annotations

import abc
import threading
import time
from typing import TYPE_CHECKING

import lcm
import numpy as np

from lcm_types.itmessage import vector_t

if TYPE_CHECKING:
    from src.deployment_area.polygon import PolygonWrapper
    from src.robots.robot_modules.robot_simulation import RobotSimulation


class StateMonitor(metaclass=abc.ABCMeta):
    """Represents a type of sensor that monitors the state of a robot, e.g. its position or rotation."""

    def __init__(self, ts_control: float = 0.2, **kwargs) -> None:
        self.ts_control = ts_control
        return

    @abc.abstractmethod
    def get_current_position(self) -> np.ndarray:
        """Get the current position of the robot."""
        raise NotImplementedError

    @abc.abstractmethod
    def _monitor_state(self) -> None:
        """Implements the monitoring function that is continuously called by the monitor thread."""
        raise NotImplementedError()

    def start(self, start_area: PolygonWrapper) -> None:
        if hasattr(self, "monitor_thread") and not self.monitor_thread.is_alive():
            self.is_monitoring = True
            self.monitor_thread.start()
        if hasattr(self, "simulated_robot"):
            self.simulated_robot.initialize_robot_state(start_area)  # type: ignore
            self.simulated_robot.start()  # type: ignore
        return

    def stop(self) -> None:
        if hasattr(self, "simulated_robot"):
            self.simulated_robot.stop()  # type: ignore
        if hasattr(self, "monitor_thread") and self.monitor_thread.is_alive():
            self.is_monitoring = False
            # create a new thread for the next run
            self.monitor_thread = threading.Thread(
                target=self._monitor_state, daemon=True
            )
        return


class SimulatedSensor(StateMonitor):
    """Simulates the behaviour of a position sensor that is directly attached to the robot."""

    def __init__(
        self, simulated_robot: RobotSimulation, ts_control: float = 0.2, **kwargs
    ) -> None:
        super().__init__(ts_control)
        self.simulated_robot = simulated_robot
        return

    def get_current_position(self) -> np.ndarray:
        self._monitor_state()
        return self.current_position

    def _monitor_state(self) -> None:
        """Get the robot state from the simulated robot."""
        self.current_position = self.simulated_robot.get_robot_state()[-2:]
        return


class SimulatedOptitrack(StateMonitor):
    """Simulates the behaviour of an optitrack (camera) system that monitors the robot
    positions and communicates them via LCM."""

    def __init__(
        self,
        robot_id: int,
        simulated_robot: RobotSimulation,
        ts_control: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(ts_control)
        self.simulated_robot = simulated_robot
        self.monitored_robot_id = robot_id

        self.pause_between_monitoring = self.ts_control / 5
        self.monitor_thread = threading.Thread(target=self._monitor_state, daemon=True)
        # Uses sequential message numbers to order messages received via UDP.
        self.seq_number_pos = 0
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
        return

    def get_current_position(self) -> np.ndarray:
        """This is not necessary, when the classes are used correctly, since the position is sent via LCM."""
        return self.current_position

    def _monitor_state(self) -> None:
        """Get the robot state from the simulated robot and send it via LCM to the state monitor."""
        while self.is_monitoring:
            self.current_position = self.simulated_robot.get_robot_state()[-2:]
            self._send_position(self.current_position)
            time.sleep(self.pause_between_monitoring)

    def _send_position(self, position: np.ndarray) -> None:
        """Sends a LCM message containing the id of the monitored robot and its current state.

        Args:
            position (np.array): Robot position in the inertial coordinate system.
        """
        msg_content = np.zeros(shape=(6,))
        msg_content[:2] = position
        self.seq_number_pos += 1

        state_msg = vector_t()
        state_msg.length = 6
        state_msg.id_sender = self.monitored_robot_id
        state_msg.seq_number = self.seq_number_pos
        state_msg.value = list(msg_content)
        # print(state_msg.value)
        self.lc.publish(f"/robot{self.monitored_robot_id}/euler", state_msg.encode())
        return
