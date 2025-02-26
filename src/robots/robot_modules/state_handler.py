from __future__ import annotations

import abc
import threading
from typing import TYPE_CHECKING

import lcm
import numpy as np
from lcm_types.itmessage import vector_t
from src.robots.robot_modules.state_monitor import *

if TYPE_CHECKING:
    from src.robots.robot_modules.state_handler import RobotState
# pyright: reportAttributeAccessIssue=false


class RobotState:

    def __init__(
        self,
        time_stamp: float,
        position: np.ndarray,
        intermediate_positions: list[np.ndarray] = [],
        **kwargs,
    ) -> None:
        self.time_stamp = time_stamp
        self.position = position
        self.intermediate_positions = []  # intermediate_positions
        return

    def add_intermediate_state(self, intermediate_state: RobotState) -> None:
        # print(intermediate_state.position)
        # print(len(self.intermediate_positions))
        self.intermediate_positions.append(intermediate_state.position)
        return


class StateHandler(metaclass=abc.ABCMeta):
    """Keeps track of the robot's current state and state history.
    The state can contain the robot's position, velocity, energy level, etc.
    """

    def __init__(self, ts_control: float, **kwargs) -> None:
        self.ts_control = ts_control

    def get_state_history(self) -> list:
        if len(self.state_history) == 0:
            self.record_state(intermediate_state=False)
        return self.state_history

    @abc.abstractmethod
    def get_robot_state(self) -> RobotState:
        raise NotImplementedError()

    @abc.abstractmethod
    def _update_state(self) -> None:
        """Receive the robot's current state and save it to the state history."""
        raise NotImplementedError()

    @abc.abstractmethod
    def record_state(self, intermediate_state: bool) -> None:
        """Record the current state and the corresponding time stamp.

        Parameters
        ----------
        intermediate_state : bool
            Set to True if the state should only be saved in the detailed state history that is used, e.g., for animations.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def start(self) -> None:

        self.state = None
        self.state_history = []
        self.start_time = time.time()

        if hasattr(self, "update_thread") and not self.update_thread.is_alive():
            print("state_handler: start listening")
            self.is_updating = True
            self.update_thread.start()

        return

    def stop(self) -> None:
        if hasattr(self, "state_monitor"):
            self.state_monitor.stop()

        if hasattr(self, "update_thread") and self.update_thread.is_alive():
            self.is_updating = False
            # create a new thread for the next run
            self.update_thread = threading.Thread(
                target=self._update_state, daemon=True
            )
        return

    def reset(self) -> None:
        self.state = None
        self.state_history = []
        self.start_time = time.time()
        return


class BasicStateHandler(StateHandler):
    """Has access to a state monitor that monitors the robot's state."""

    def __init__(
        self, state_monitor: SimulatedSensor, ts_control: float = 0.2, **kwargs
    ) -> None:
        super().__init__(ts_control)
        self.state_monitor = state_monitor

    def record_state(self, intermediate_state: bool) -> None:
        self._update_state()
        if intermediate_state:
            # print('h', self.state_history[-1].position)
            # print('s', self.state.position)
            # print(" ")
            self.state_history[-1].add_intermediate_state(self.state)
        else:
            self.state_history.append(self.state)
        return

    def get_robot_state(self) -> RobotState:
        self._update_state()
        return self.state

    def _update_state(self) -> None:
        """Get the current state from the state monitor."""
        current_position = self.state_monitor.get_current_position()
        time_stamp = np.round(time.time() - self.start_time, decimals=1)
        self.state = RobotState(time_stamp, current_position)
        return


class FeignedStateHandler(StateHandler):
    def __init__(self, ts_control: float = 0.2, **kwargs) -> None:
        super().__init__(ts_control)

    def record_state(self, intermediate_state: bool) -> None:
        return

    def get_robot_state(self) -> RobotState:
        return self.state

    def _update_state(self) -> None:
        return

    def record_feigned_state(self, current_position: np.ndarray) -> None:
        """Record the given (feigned) position as a state.

        Parameters
        ----------
        current_position : np.ndarray
            Feigned position of the robot.
        """
        time_stamp = np.round(time.time() - self.start_time, decimals=1)
        self.state = RobotState(time_stamp, current_position)
        self.state_history.append(self.state)
        return


class LCM_StateHandler(StateHandler):
    """Uses LCM to receive the robot state sent by a state monitor."""

    def __init__(
        self,
        id: int,
        communication_id: int,
        ts_control: float = 0.2,
        ttl: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(ts_control)

        self.id = id
        # The communication id refers to the robot id used for communication. When using hardware robots, the communication id might differ from the robot id.
        self.communication_id = communication_id
        self.pause_between_updating = self.ts_control / 3
        self.update_thread = threading.Thread(target=self._update_state, daemon=True)

        # Uses sequential message numbers to order messages received via UDP.
        self.seq_number_pos = 0
        self.lc = lcm.LCM(f"udpm://239.255.76.67:7667?ttl={ttl}")
        lcs = self.lc.subscribe(
            f"/robot{self.communication_id}/euler", self._lcm_handler
        )
        lcs.set_queue_capacity(1)

    def record_state(self, intermediate_state: bool) -> None:
        if intermediate_state:
            self.state_history[-1].add_intermediate_state(self.state)
        else:
            self.state_history.append(self.state)
        return

    def get_robot_state(self) -> RobotState:
        while self.state is None:
            print(
                f"waiting for position data on robot with communication id {self.communication_id}"
            )
            time.sleep(1)
        return self.state

    def _update_state(self) -> None:
        """Listen to LCM messages that contain the robot state."""
        while self.is_updating:
            # timeout in ms
            self.lc.handle_timeout(int(3 * self.ts_control * 1000))
            time.sleep(self.pause_between_updating)
        print("state_handler: stop listening")

    def _lcm_handler(self, _, state: vector_t) -> None:
        """Read the LCM message containing the robot state that was sent by a state monitor.

        Args:
            state (vector_t): message
        """
        msg = vector_t.decode(state)
        if msg.seq_number > self.seq_number_pos:
            self.seq_number_pos = msg.seq_number
            current_position = np.array(msg.value)[:2]
            time_stamp = np.round(time.time() - self.start_time, decimals=1)
            self.state = RobotState(time_stamp, current_position)
        return
