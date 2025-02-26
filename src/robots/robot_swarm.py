from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
from src.deployment_area.polygon import PolygonWrapper
from src.deployment_area.voronoi_helpers import compute_delaunay
from src.visualize.swarm_visualization import SwarmVisualization

if TYPE_CHECKING:
    from src.anomaly_detectors.anomaly_detector import AnomalyDetector
    from src.deployment_area.polygon import PolygonWrapper
    from src.robots.deployment_robot import DeploymentRobot, Robot
    from src.robots.robot_modules.state_handler import RobotState
    from src.robots.robot_modules.state_monitor import StateMonitor


class RobotSwarm:

    def __init__(
        self,
        swarm_robots: list[Robot],
        ts_communicate: float,
        anomaly_detector: AnomalyDetector | None = None,
        **kwargs,
    ):
        """Create a robot swarm from a list of robots.

        Args:
            swarm_robots (list[Robot]): robot entities belonging to the swarm
        """
        self.swarm_robots = swarm_robots
        self.n_robots = len(swarm_robots)
        self.ts_communicate = ts_communicate

        self.set_anomaly_detector(anomaly_detector)
        self.visualization_module = SwarmVisualization(self)  # type: ignore

    async def cover_area(
        self,
        conv_crit: float,
        max_steps: int,
        step: int = 0,
        **kwargs,
    ) -> None:
        """Given a coverage area, the robot swarm will iteratively cover the area by moving to the weighted center of the voronoi cells. The algorithm stops if the robots have converged such that their actions are smaller than a given threshold or if the maximum number of steps has been reached, with each step running for a duration of ts_communicate seconds.

        Args:
            conv_crit (float, optional): Convergence criterium for the robot actions. Defaults to 5e-2.
            max_steps (int, optional): Maximum number of steps that will be performed. Defaults to 15.
        """
        assert self.deployment_area is not None, "please specify a coverage area"

        is_converged = False
        self.step = step

        while not is_converged and self.step < max_steps:

            start_time_tasks = time.time()
            target_tasks = [
                asyncio.create_task(robot.update_target(self))
                for robot in self.swarm_robots
            ]
            await asyncio.wait(
                target_tasks,
                return_when=asyncio.ALL_COMPLETED,
            )

            remaining_time = self.ts_communicate  # - (time.time() - start_time_tasks)

            positions = self.get_positions()
            move_tasks = [
                asyncio.create_task(robot.update_position(remaining_time))
                for robot in self.swarm_robots
            ]

            _, pending = await asyncio.wait(
                move_tasks,
                timeout=self.ts_communicate + 1e-2,
                return_when=asyncio.ALL_COMPLETED,
            )
            if pending:
                print("move tasks pending!")
            self.step += 1

            actions = self.get_last_action()
            is_converged = (np.abs(actions) < conv_crit).all()

            if self.anomaly_detector is not None:
                self.run_anomaly_detection(
                    step=self.step,
                    last_positions=positions,
                    actions=actions,
                )

        return

    def run_anomaly_detection(
        self, step: int, last_positions: np.ndarray, actions: np.ndarray
    ) -> None:

        if self.anomaly_detector is not None:

            self.anomaly_detector.evaluate_step(
                pos=last_positions,
                actions=actions,
                deployment_area=self.deployment_area,
                n_samples=20,
                mask_anomal=True,
            )
            print(
                f"action {step}:, anomaly recognized: {np.where(self.anomaly_detector.get_anomaly_prediction_of_swarm())[0].tolist()}"
            )

    def is_anomal(self, robot: Robot, timestep: int = -1) -> bool:
        """Check if a robot agent has been classified as anomal.

        Args:
            robot (Robot): robot agent

        Returns:
            bool: anomaly prediction
        """
        if self.anomaly_detector is None:
            return False
        else:
            return self.anomaly_detector.is_robot_anomal(robot=robot, timestep=timestep)

    def get_positions(self) -> np.ndarray:
        """Get the positions of the swarm robots.

        Returns:
            np.ndarray: positions
        """
        return np.array(
            [robot.get_communicated_position() for robot in self.swarm_robots]
        )

    def get_last_action(self) -> np.ndarray:
        """Get the last actions of the swarm robots.

        Returns:
            np.ndarray: last actions
        """
        return np.array(
            [
                robot.get_communicated_position()
                - robot.get_communicated_state_history()[-1].position
                for robot in self.swarm_robots
            ]
        )

    def get_vel_history(
        self,
    ) -> list:
        """Get the swarm robots' motions.

        Returns
        -------
        list
            Motion history per swarm robot.
        """
        return [robot.get_vel_history() for robot in self.swarm_robots]

    def set_area_bounds(self, area: PolygonWrapper) -> None:
        """Define the area that the robots are allowed to move in. In case of the deployment problem, this area corresponds to the area that the robot swarm will cover.

        Parameters
        ----------
        deployment_area : PolygonWrapper

        """
        """Set the area that will be covered by the robot swarm.

        Args:
            robot_area (PolygonWrapper): 
                Area that the .
            boundary (PolygonWrapper): Convex coverage area.
        """
        self.deployment_area = area
        for robot in self.swarm_robots:
            robot.set_deployment_area(area)
        return

    def set_anomaly_detector(self, anomaly_detector: AnomalyDetector | None) -> None:
        """Set the anomaly detector and pass information about the number of robots in the swarm.

        Args:
            anomaly_detector (AnomalyDetector): Contains a model that has been trained to detect anomalies.
        """
        self.anomaly_detector = anomaly_detector
        if self.anomaly_detector is not None:
            self.anomaly_detector.initialize_run_prediction(self)
            for robot in self.swarm_robots:
                robot.set_anomaly_detector(self.anomaly_detector)
        return

    def start_run(
        self,
        deployment_area: PolygonWrapper,
        external_state_monitors: list[StateMonitor] | None = None,
    ) -> None:
        """Start the handler modules/threads of each robot and the communication/monitoring threads of external state monitors.

        Args:
            external_state_monitors (list[StateMonitor]): external state monitors that are not part of the robot swarm
        """
        self.set_area_bounds(deployment_area)
        tri = compute_delaunay(deployment_area)
        start_area = PolygonWrapper(tri[np.random.choice(len(tri))])

        if external_state_monitors is not None:
            for sm in external_state_monitors:
                sm.start(start_area)
        for robot in self.swarm_robots:
            robot.start()
        return

    def stop_run(
        self, external_state_monitors: list[StateMonitor] | None = None
    ) -> None:
        """Stop the handler modules/threads of each robot and the communication/monitoring threads of external state monitors.

        Args:
            external_state_monitors (list[StateMonitor]): external state monitors that are not part of the robot swarm
        """
        for robot in self.swarm_robots:
            robot.stop()
        if external_state_monitors is not None:
            for sm in external_state_monitors:
                sm.stop()
        return

    def run_info(self, print_run: bool = True) -> tuple[int, int, list]:

        success_per_robot = []

        positions = [
            r.state_handler.state_history[-1].position for r in self.swarm_robots
        ]
        if print_run:
            print(f"Number of robots: {self.n_robots}, number of steps: {self.step}")
        for robot in self.swarm_robots:
            success_per_robot.append(robot.is_successful(positions))
            if robot.label != 0 and print_run:
                print(
                    f"Success of {robot.__class__.__name__} with final position {robot.get_communicated_position()}: {robot.is_successful(positions)}"
                )
        successful_robots = success_per_robot.count(True)
        unsuccessful_robots = success_per_robot.count(False)
        return successful_robots, unsuccessful_robots, success_per_robot


# TODO: reset anomaly detector
