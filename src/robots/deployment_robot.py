from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING, List

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, weibull_min
from scipy.stats._multivariate import multivariate_normal_frozen
from src.deployment_area.voronoi_helpers import (
    compute_voronoi_cell,
    compute_weighted_center,
    l2_norm,
)
from src.robots.robot import Robot
from src.robots.robot_modules.move_handler import MoveHandler, SimulationMoveHandler
from src.robots.robot_modules.state_handler import (
    FeignedStateHandler,
    RobotState,
    StateHandler,
)
from src.robots.robot_modules.swarm_communication_handler import (
    SwarmCommunicationHandler,
)

if TYPE_CHECKING:
    from src.deployment_area.polygon import PolygonWrapper
    from src.robots.robot_modules.state_handler import StateHandler
    from src.robots.robot_modules.swarm_communication_handler import (
        SwarmCommunicationHandler,
    )
    from src.robots.robot_swarm import RobotSwarm


class DeploymentRobot(Robot, metaclass=abc.ABCMeta):

    def __init__(
        self,
        id: int,
        color: str,
        state_handler: StateHandler,
        move_handler: MoveHandler,
        swarm_communication_handler: SwarmCommunicationHandler,
        ts_control: float,
        max_vel: float,
        wait_for_ts_communicate: bool,
        **kwargs,
    ) -> None:
        """Robot solving a deployment task.

        Parameters
        ----------
        id : int
            robot id
        state_handler : StateHandler
            Collects state information, e.g. the current position.
        move_handler : MoveHandler
            Triggers the movement of the robot.
        swarm_communication_handler : SwarmCommunicationHandler
            Used to communicate with the robot swarm.
        ts_control : float
            Every ts_control seconds the target velocity is updated based on the current position.
        max_vel : float
            Maximum velocity per second that the robot can achieve.
        wait_for_ts_communicate : bool
            If True, the robot waits for the duration of ts_control before triggering the next movement.
        """

        super().__init__(
            id,
            color=color,
            state_handler=state_handler,
            move_handler=move_handler,
            swarm_communication_handler=swarm_communication_handler,
            ts_control=ts_control,
            max_vel=max_vel,
        )
        self.wait_for_ts_communicate = wait_for_ts_communicate

    async def update_target(self, robot_swarm: RobotSwarm) -> None:

        self.record_state(intermediate_state=False)

        swarm_info = await asyncio.gather(
            self.swarm_communication_handler.gather_swarm_info(
                info=self.get_communicated_position(), robot_swarm=robot_swarm
            )
        )
        swarm_positions = swarm_info[0]
        self.current_target = self._compute_target_position(
            robot_positions=swarm_positions
        )
        self.target_history.append(self.current_target)
        # print(f"{self.id} Time: {time.time() - start}")

        return

    async def update_position(self, remaining_time: float) -> None:
        """
        Executes the robot actions during one communication timestep ts_communicate. The robot exchanges position information with the other agents in the swarm. Based on the position information, a new target position is determined and the corresponding target velocity is triggered.

            Parameters
            ----------
            remaining_time:
                Remainder of time in ts_communicate until the next communication with the swarm.
        """
        target = self.current_target
        # move for the duration of the control timestep ts_control before adapting the velocity
        n_iter = int(remaining_time / self.ts_control)
        for i in range(n_iter):

            vel = target - self._get_current_position()
            vel *= self._stop_at_bounds()
            self.trigger_movement(vel)

            if self.wait_for_ts_communicate:
                await asyncio.sleep(self.ts_control)

            if i < n_iter - 1:
                self.record_state(intermediate_state=True)

        return

    @abc.abstractmethod
    def _compute_target_position(self, robot_positions: np.ndarray) -> np.ndarray:
        """Compute the new target position of a robot depending on its type. Implemented by the subclasses of DeploymentRobot.

        Args:
            robot_positions (np.array): current positions of other entities in the robot swarm

        Returns:
            np.array: Target position of the robot.
        """
        raise NotImplementedError()

    def _compute_optimal_target(self, robot_positions: np.ndarray) -> np.ndarray:
        """The optimal target of a robot is the center of its current voronoi cell.
        Args:
            robot_positions (np.array): current positions of other entities in the robot swarm

        Returns:
            np.array: Optimal target position of the robot based on the coverage area and the positions of the other swarm robots.
        """
        self_position = robot_positions[self.id]

        if hasattr(self, "anomaly_detector"):
            normal_prediction = np.invert(
                self.anomaly_detector.get_anomaly_prediction_of_swarm()
            )
            # robot considers itself normal
            normal_prediction[self.id] = True
            robot_positions = robot_positions[normal_prediction]

        robot_region = compute_voronoi_cell(
            self_position, robot_positions, self.deployment_area
        )
        optimal_target = robot_region.compute_center()

        return optimal_target

    def _set_antagonist_target(self):
        return

    def _stop_at_bounds(self) -> np.ndarray:
        """Allow the robot to move outside the coverage area, as long as it stays within the boundary area. Returns a velocity modifier that sets the velocity to zero immediately, as soon as the robot position is outside the boundary area.

        Returns:
            np.array: Velocity modifier. Multiply with the velocity.
        """
        inside_bounds = self.deployment_area.contains_point(
            self._get_current_position()
        )
        if inside_bounds:
            return np.array([1, 1])
        else:
            return np.array([0, 0])

    def is_successful(self, positions) -> bool | None:

        if hasattr(self, "AOI"):
            voronoi_cell = compute_voronoi_cell(
                positions[self.id], positions, self.deployment_area
            )
            AOI_under_control = voronoi_cell.contains_point(self.AOI)  # type: ignore
            return AOI_under_control
        else:
            return None

    def start(self) -> None:
        """Start the robot modules, set an anomalous target if relevant for the robot type."""
        self.move_handler.start()
        self.state_handler.start()
        self.swarm_communication_handler.start()
        self._set_antagonist_target()
        return

    def stop(self) -> None:
        """Record the current state of the robot and stop the robot modules."""
        self.record_state(intermediate_state=False)
        self.move_handler.stop()
        self.state_handler.stop()
        self.swarm_communication_handler.stop()
        return

    def reset(self) -> None:
        """Reset the history of robot motions, set a new density function. Does not restart the robot modules."""
        self.target_history = []

        self.move_handler.reset()
        self.state_handler.reset()
        self.swarm_communication_handler.reset()
        self._set_antagonist_target()
        return


class VoronoiRobot(DeploymentRobot):
    """
    Optimizes the voronoi decomposition within a polygonal coverage area by
    moving towards the center of its voronoi cell.
    """

    def __init__(self, id: int, color: str = "#f7f7f7", **kwargs):
        super().__init__(id, color, **kwargs)
        self.label = 0
        self.label_text = "normal"

    def _compute_target_position(self, robot_positions: np.ndarray) -> np.ndarray:

        new_target = super()._compute_optimal_target(robot_positions)

        return new_target


class BruteForceRobot(DeploymentRobot):

    def __init__(self, id: int, color: str = "#2C3586", **kwargs) -> None:
        super().__init__(id, color, **kwargs)
        self.label = 1
        self.label_text = "brute-force"

    def _compute_target_position(self, robot_positions: np.ndarray) -> np.ndarray:
        return self.AOI

    def _set_antagonist_target(self) -> None:
        target_in_area = False
        while not target_in_area:
            self.AOI = np.random.uniform(
                low=np.min(self.deployment_area.vertices),
                high=np.max(self.deployment_area.vertices),
                size=(2),
            )
            target_in_area = self.deployment_area.contains_point(self.AOI)
        return


class WeightedRobot(DeploymentRobot):
    """
    Optimizes the voronoi decomposition within a weighted polygonal coverage area by
    moving towards the center of its voronoi cell weighted by the density function.
    """

    def __init__(self, id: int, color: str = "#DB7093", **kwargs) -> None:
        super().__init__(id, color, **kwargs)
        self.max_dist_target = 10 * self.max_vel  # reachable within 10 seconds
        self.label = 2
        self.label_text = "Weibull"
        self.density = weibull_min(
            c=0.7,
            loc=-self.max_dist_target - 0.1 * self.max_dist_target,
            scale=(self.max_dist_target / 4),
        )

    def _set_antagonist_target(self) -> None:

        target_in_area = False
        target_in_reach = False

        while not (target_in_area and target_in_reach):
            self.AOI = np.random.uniform(
                low=np.min(self.deployment_area.vertices),
                high=np.max(self.deployment_area.vertices),
                size=(2),
            )
            target_in_area = self.deployment_area.contains_point(self.AOI)
            target_in_reach = (
                l2_norm(self._get_current_position() - self.AOI) < self.max_dist_target
            )
        # loc = l2_norm(self.AOI - self._get_current_position())
        # self.density = weibull_min(c=0.7, loc=-loc, scale=(loc / 6))
        # self.density = weibull_min(c=0.7, loc=-loc, scale=(loc / 2))
        # self.density = weibull_min(c=0.7, loc=-max_dist_target-0.1*max_dist_target, scale=(max_dist_target / 2))

        return

    def _compute_target_position(self, robot_positions: np.ndarray) -> np.ndarray:

        robot_region = compute_voronoi_cell(
            robot_positions[self.id], robot_positions, self.deployment_area
        )
        new_target = compute_weighted_center(robot_region, self.density_func)
        return new_target

    def density_func(self, x: np.ndarray) -> float:
        return self.density.cdf(-l2_norm(self.AOI - x)) + 0.1


class WeightedAggressiveRobot(WeightedRobot):
    def __init__(self, id: int, color: str = "#9F0048", **kwargs) -> None:
        super().__init__(id, color, aggressive=True, **kwargs)
        self.label = 3
        self.label_text = "Weibull aggressive"
        self.density = weibull_min(
            c=1.6,
            loc=-self.max_dist_target - 0.1 * self.max_dist_target,
            scale=(self.max_dist_target / 2),
        )


class SpoofingRobot(DeploymentRobot):

    def __init__(
        self,
        id: int,
        color: str = "#00C894",
        **kwargs,
    ):
        super().__init__(id, color, **kwargs)
        self.label = 5
        self.target_reached = False
        self.label_text = "spoofing"
        self.feigned_state_handler = FeignedStateHandler()

    def get_communicated_position(self) -> np.ndarray:

        if len(self.target_history) == 0 or self.spoofing_completed():
            return self._get_current_position()
        else:
            return self.normal_behavior_target

    def get_communicated_state_history(self) -> List[RobotState]:
        return self.feigned_state_handler.get_state_history()

    def _set_antagonist_target(self) -> None:
        """
        Set the density function weighting the coverage area.

            Parameters
            ----------
            density_func:
                Density function weighting the coverage area.
        """
        initial_position = self._get_current_position()
        self.normal_behavior_target = initial_position

        target_in_area = False
        corners = self.deployment_area.get_vertices()
        corner_is_reachable = (
            l2_norm(corners - initial_position) < 30 * self.max_vel
        )  # reachable in 30 seconds
        reachable_corners = corners[corner_is_reachable]
        if len(reachable_corners) == 0:
            AOI_corner = corners[0]
        else:
            AOI_corner = reachable_corners[np.random.choice(len(reachable_corners))]

        while not target_in_area:
            self.AOI = AOI_corner + np.random.uniform(
                low=-1,
                high=1,
                size=(2),
            )
            target_in_area = self.deployment_area.contains_point(self.AOI)
        return

    def _compute_target_position(self, robot_positions: np.ndarray) -> np.ndarray:

        self.normal_behavior_target = super()._compute_optimal_target(robot_positions)
        if self.spoofing_completed():
            return self.normal_behavior_target
        else:
            return self.AOI

    def spoofing_completed(self) -> bool:
        self.target_reached = self.target_reached or bool(
            l2_norm(self.AOI - self._get_current_position()) < 0.1
        )
        # wait until the robots have distributed in the area in order to decrease the likelihood of many robots being close to the AOI
        waited_for_swarm_to_spread = len(self.target_history) > 3

        return self.target_reached and waited_for_swarm_to_spread

    def record_state(self, intermediate_state: bool) -> None:
        self.state_handler.record_state(intermediate_state=intermediate_state)
        if intermediate_state:
            return
        elif self.spoofing_completed():
            self.feigned_state_handler.record_feigned_state(
                self._get_current_position()
            )
        else:
            self.feigned_state_handler.record_feigned_state(self.normal_behavior_target)
        return

    def start(self) -> None:
        """Start the robot modules, set an anomalous target if relevant for the robot type."""
        super().start()
        self.feigned_state_handler.start()
        return

    def reset(self) -> None:
        """Reset the history of robot motions, set a new density function. Does not restart the robot modules."""
        super().reset()
        self.feigned_state_handler.reset()
        return


class SneakyRobot(DeploymentRobot):
    """When close to convergence or the end of the deployment task, the robot starts to take small steps towards its target area."""

    def __init__(
        self,
        id: int,
        max_steps: int,
        sneaky_step_size: float,
        color: str = "#FFBC5C",
        **kwargs,
    ):
        super().__init__(id, color, **kwargs)
        self.label = 6
        self.label_text = "sneaky"

        self.max_steps = max_steps
        self.sneaky_step_size = sneaky_step_size

    def _set_antagonist_target(self) -> None:

        self.is_sneaky = False
        target_in_area = False
        target_in_reach = False
        max_dist_target = 30 * self.max_vel  # reachable within 30 seconds

        while not (target_in_area and target_in_reach):
            self.AOI = np.random.uniform(
                low=np.min(self.deployment_area.vertices),
                high=np.max(self.deployment_area.vertices),
                size=(2),
            )
            target_in_area = self.deployment_area.contains_point(self.AOI)
            target_in_reach = (
                l2_norm(self._get_current_position() - self.AOI) < max_dist_target
            )
        return

    def _compute_target_position(self, robot_positions: np.ndarray) -> np.ndarray:

        not_converged = len(self.target_history) < 3 or (
            l2_norm(self.target_history[-1] - self.target_history[-2])
            > self.sneaky_step_size
        )
        if not self.is_sneaky and not_converged:
            if len(self.target_history) == self.max_steps - 1:
                print(f"{self.id} is not sneaky!")
                self.color = "#f7f7f7"
                self.label = 0
            new_target = super()._compute_optimal_target(robot_positions)
        elif self.is_successful(robot_positions):
            new_target = self._get_current_position()
        else:
            if not self.is_sneaky:
                print(f"{self.id} is sneaky!")
                self.is_sneaky = True
            target_action = self.AOI - self._get_current_position()
            sneaky_action = (
                target_action / l2_norm(target_action)
            ) * self.sneaky_step_size
            new_target = self._get_current_position() + sneaky_action

        return new_target
