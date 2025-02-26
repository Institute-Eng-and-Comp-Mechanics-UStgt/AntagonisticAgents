from __future__ import annotations

import abc
import asyncio
import threading
import time
from typing import TYPE_CHECKING

import lcm
import numpy as np

from lcm_types.itmessage import vector_t

if TYPE_CHECKING:
    from src.robots.robot_modules.state_monitor import StateMonitor
    from src.robots.robot_swarm import RobotSwarm


class SwarmCommunicationHandler(metaclass=abc.ABCMeta):
    """Communicates with other entities of the robot swarm, e.g. in order to send and receive robot states."""

    def __init__(self, **kwargs) -> None:
        return

    @abc.abstractmethod
    async def gather_swarm_info(
        self, info: np.ndarray, robot_swarm: RobotSwarm
    ) -> np.ndarray:
        """Gather information about the swarm positions.

        Returns:
            np.array: positions of the swarm agents
        """
        raise NotImplementedError()

    def start(self) -> None:

        self.swarm_info = {}

        if hasattr(self, "listen_thread") and not self.listen_thread.is_alive():
            print("communication_handler: start listening")
            self.is_listening = True
            self.listen_thread.start()

        return

    def stop(self) -> None:
        if hasattr(self, "listen_thread") and self.listen_thread.is_alive():
            self.is_listening = False
            # create a new thread for the next run
            self.listen_thread = threading.Thread(target=self._listen, daemon=True)  # type: ignore
        return

    def reset(self) -> None:
        self.swarm_info = {}
        return


class BasicCommunicationHandler(SwarmCommunicationHandler):
    """Uses the reference to the robot swarm to gather information about the swarm state."""

    def __init__(self, state_monitor: StateMonitor, **kwargs) -> None:
        super().__init__()
        self.state_monitor = state_monitor

    async def gather_swarm_info(
        self, info: np.ndarray, robot_swarm: RobotSwarm
    ) -> np.ndarray:
        """Collects information about the swarm state from the robot swarm.

        Args:
            info (np.array): information about the robot's current position that is sent to the swarm, unused in this implementation
            robot_swarm (RobotSwarm): robot swarm

        Returns:
            np.ndarray: most recent information received by the agents within the swarm
        """
        return robot_swarm.get_positions()


class LCM_CommunicationHandler(SwarmCommunicationHandler):
    """Uses LCM to send and receive messages to and from other entities in the robot swarm."""

    def __init__(
        self, communication_id: int, ts_communicate: float = 3, ttl: int = 0, **kwargs
    ) -> None:
        super().__init__()

        # The communication id refers to the robot id used for communication. When using hardware robots, the communication id might differ from the robot id.
        self.communication_id = communication_id
        self.ts_communicate = ts_communicate

        self.listen_thread = threading.Thread(target=self._listen, daemon=True)
        self.wait_for_swarm_info = 0.1
        self.lc = lcm.LCM(f"udpm://239.255.76.67:7667?ttl={ttl}")
        # Uses sequential message numbers to order messages received via UDP.
        self.seq_number = 0
        lcs = self.lc.subscribe(f"/robot/euler", self._lcm_handler)
        lcs.set_queue_capacity(30)

    async def gather_swarm_info(
        self, info: np.ndarray, robot_swarm: RobotSwarm
    ) -> np.ndarray:
        """Sends information about its own state to the swarm. Waits for a short period of time while receiving the messages from other agents, before returning the swarm information.

        Args:
            info (np.array): information about the robot's current position that is sent to the swarm
            robot_swarm (RobotSwarm): robot swarm, unused in this implementation

        Returns:
            np.ndarray: most recent information received by other agents within the swarm
        """
        self._communicate_to_swarm(info)
        # wait for swarm info to arrive
        await asyncio.sleep(self.wait_for_swarm_info)
        return np.vstack([self.swarm_info[k] for k in self.swarm_info.keys()])

    def _communicate_to_swarm(self, position: np.ndarray) -> None:
        """Send a LCM message containing the robot state to the swarm.

        Args:
            position (np.array): robot position
        """
        msg_content = list(np.zeros(shape=(6,)))
        msg_content[:2] = position
        self.seq_number += 1

        state_msg = vector_t()
        state_msg.length = 6
        state_msg.id_sender = self.communication_id
        state_msg.seq_number = self.seq_number
        state_msg.value = msg_content

        self.lc.publish(f"/robot/euler", state_msg.encode())

    def _listen(self) -> None:
        while self.is_listening:
            self.lc.handle_timeout(int(2 * self.ts_communicate * 1000))
        print("communication_handler: stop listening")

    def _lcm_handler(self, _, state: vector_t) -> None:
        """Decode LCM messages sent by the other swarm agents.

        Args:
            state (vector_t): message
        """
        msg = vector_t.decode(state)
        # 6 dim: 1st x, 2nd y, 4th rotation
        if msg.seq_number >= self.seq_number:
            current_position = np.array(msg.value)[:2]
            self.swarm_info[msg.id_sender] = current_position

        return
