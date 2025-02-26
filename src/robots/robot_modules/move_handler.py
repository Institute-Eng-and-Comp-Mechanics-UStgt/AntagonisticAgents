from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import lcm
import numpy as np
from lcm_types.itmessage import vector_t

if TYPE_CHECKING:
    from src.robots.robot_modules.robot_simulation import RobotSimulation


class MoveHandler(metaclass=abc.ABCMeta):
    """Sends the current velocity to the robot and saves the velocity history."""

    def __init__(self, ts_control: float = 0.2, **kwargs) -> None:
        self.ts_control = ts_control
        return

    @abc.abstractmethod
    def trigger_movement(self, vel: np.ndarray) -> None:
        """Trigger a movement of target velocity vel in the corresponding robot.

        Args:
            vel (np.array): Target velocity.
        """
        raise NotImplementedError()

    def start(self) -> None:
        self.velocity_history = []
        return

    def stop(self) -> None:
        return

    def reset(self) -> None:
        self.velocity_history = []
        return


class SimulationMoveHandler(MoveHandler):
    """Holds a reference to a robot simulation monitored by a state_monitor and triggers the movement of this robot simulation.
    Wait for a duration of ts_control while the robot moves."""

    def __init__(
        self, simulated_robot: RobotSimulation, ts_control: float = 0.2, **kwargs
    ) -> None:
        super().__init__(ts_control)
        self.simulated_robot = simulated_robot

    def trigger_movement(self, vel: np.ndarray) -> None:
        """Passes the target velocity to the robot simulation.

        Args:
            vel (np.array): Target velocity.
        """
        self.simulated_robot.trigger_movement(vel)
        # self.velocity_history.append(vel)
        return


class LCM_MoveHandler(MoveHandler):
    """Uses LCM to trigger the movement of the robot with the given id."""

    def __init__(
        self, communication_id: int, ts_control: float = 0.2, ttl: int = 0, **kwargs
    ) -> None:
        super().__init__(ts_control)

        # The communication id refers to the robot id used for communication. When using hardware robots, the communication id might differ from the robot id.
        self.communication_id = communication_id
        self.lc = lcm.LCM(f"udpm://239.255.76.67:7667?ttl={ttl}")
        # Uses sequential message numbers to order messages received via UDP.
        self.seq_number_u = 0

    def trigger_movement(self, vel: np.ndarray) -> None:
        """Send the target velocity via LCM and wait for a duration of ts_control while the robot moves.

        Args:
            vel (np.array): Target velocity.
        """
        # for _ in range(3):
        self._send_velocity(vel)
        self.velocity_history.append(vel)
        return

    def _send_velocity(self, vel: np.ndarray) -> None:
        """Generates and sends the velocity message via LCM.

        Args:
            vel (np.array): Velocity
        """
        u = [0.0, 0.0, 0.0]
        u[:2] = vel
        self.seq_number_u += 1

        control_input_msg = vector_t()
        control_input_msg.length = 3
        control_input_msg.id_sender = self.communication_id
        control_input_msg.seq_number = self.seq_number_u
        control_input_msg.value = u
        self.lc.publish(f"/robot{self.communication_id}/u", control_input_msg.encode())

        return
