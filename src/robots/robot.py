from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from src.anomaly_detectors.anomaly_detector import AnomalyDetector
    from src.deployment_area.polygon import PolygonWrapper
    from src.robots.robot_modules.move_handler import MoveHandler, SimulationMoveHandler
    from src.robots.robot_modules.state_handler import RobotState, StateHandler
    from src.robots.robot_modules.swarm_communication_handler import (
        SwarmCommunicationHandler,
    )
    from src.robots.robot_swarm import RobotSwarm


class Robot(object, metaclass=abc.ABCMeta):

    def __init__(
        self,
        id: int,
        color: str,
        ts_control: float,
        max_vel: float,
        state_handler: StateHandler,
        move_handler: MoveHandler,
        swarm_communication_handler: SwarmCommunicationHandler,
    ) -> None:
        self.id = id
        self.color = color
        self.label = -1
        self.label_text = ""

        self.state_handler = state_handler
        self.move_handler = move_handler
        self.swarm_communication_handler = swarm_communication_handler

        self.set_ts_control(ts_control)
        self.set_max_vel(max_vel)

        self.target_history = []
        self.current_target = None

    @abc.abstractmethod
    async def update_target(self, robot_swarm: RobotSwarm) -> None:
        """
        Compute and set the target for the robot for the next timestep.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def update_position(self, remaining_time: float) -> None:
        """
        Compute and set the position of the robot for the next timestep.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def start(self) -> None:
        """Start the robot modules."""
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the robot modules."""
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear the information collected by different robot modules."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_successful(self, positions) -> bool | None:
        raise NotImplementedError()

    def trigger_movement(self, vel: np.ndarray) -> None:
        self.move_handler.trigger_movement(vel)
        return

    def record_state(self, intermediate_state: bool) -> None:
        self.state_handler.record_state(intermediate_state=intermediate_state)
        return

    def get_communicated_state_history(self) -> List[RobotState]:
        return self.get_state_history()

    def get_state_history(self) -> List[RobotState]:
        return self.state_handler.get_state_history()

    def get_communicated_position(self) -> np.ndarray:
        return self._get_current_position()

    def _get_current_position(self) -> np.ndarray:
        return self.state_handler.get_robot_state().position

    def get_vel_history(self) -> list:
        return self.move_handler.velocity_history

    def set_deployment_area(self, area: PolygonWrapper) -> None:
        self.deployment_area = area
        return

    def set_max_vel(self, max_vel: float) -> None:
        self.max_vel = max_vel
        return

    def set_ts_control(self, ts_control: float) -> None:
        self.ts_control = ts_control
        return

    def set_ts_communicate(self, ts_communicate: float) -> None:
        self.ts_communicate = ts_communicate
        return

    def set_anomaly_detector(self, anomaly_detector: AnomalyDetector) -> None:
        self.anomaly_detector = anomaly_detector
        return
