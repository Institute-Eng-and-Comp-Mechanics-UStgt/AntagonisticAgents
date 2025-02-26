from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Callable, List

import lcm
import numpy as np
from src.deployment_area.polygon import PolygonWrapper
from src.robots.deployment_robot import VoronoiRobot
from src.robots.robot_modules.move_handler import LCM_MoveHandler, SimulationMoveHandler
from src.robots.robot_modules.robot_simulation import Hera
from src.robots.robot_modules.state_handler import BasicStateHandler, LCM_StateHandler
from src.robots.robot_modules.state_monitor import SimulatedOptitrack, SimulatedSensor
from src.robots.robot_modules.swarm_communication_handler import (
    BasicCommunicationHandler,
    LCM_CommunicationHandler,
)
from src.robots.robot_swarm import RobotSwarm
from src.visualize.swarm_visualization import SwarmVisualization

if TYPE_CHECKING:
    from src.robots.robot import Robot


def save_robot_swarm(robot_swarm: RobotSwarm, filepath: str) -> None:

    def save_to_dict(item):
        if isinstance(item, list):
            return [save_to_dict(listitem) for listitem in item]
        elif isinstance(item, lcm.LCM):
            return "LCM"
        elif hasattr(item, "__dict__"):
            # add same method for robots and deployment area
            if item.__module__.startswith("src.robots") or item.__module__.startswith(
                "src.deployment_area"
            ):
                module_dict = {"class": item.__class__}
                for key, subitem in vars(item).items():
                    module_dict[key] = save_to_dict(subitem)
                return module_dict
            elif item.__module__.startswith("scipy.stats."):
                return item.__dict__
            else:
                return item.__class__
        else:
            if type(item) == lcm.LCM:
                return []
            return item

    swarm_state = save_to_dict(robot_swarm)

    with open(filepath, "wb") as fp:
        pickle.dump(swarm_state, fp)

    return


def load_robot_swarm(filepath: str) -> RobotSwarm:

    def load_from_dict(item):

        if isinstance(item, list):
            return [load_from_dict(listitem) for listitem in item]
        elif type(item) == dict and "class" in item:
            if item.__contains__("density"):
                item.pop("density")
            mod = item["class"](**item)
            item.pop("class")
            for key in item:
                item[key] = load_from_dict(item[key])
            mod.__dict__.update(item)
            return mod
        else:
            return item

    with open(filepath, "rb") as fp:
        swarm_state = pickle.load(fp)

    # load robot swarm
    robot_swarm = load_from_dict(swarm_state)
    assert type(robot_swarm) == RobotSwarm
    # initialize visualization module
    robot_swarm.visualization_module = SwarmVisualization(robot_swarm)

    return robot_swarm


def build_random_robot_swarm(
    n_robots: int,
    build_robot_fn: Callable,
    max_vel: float = 0.4,
    ts_communicate: float = 5,
    ts_control: float = 0.2,
    n_anomal_robots=0,
    anomal_types: List = [],
    p_anomal: List | None = None,
    communication_ids=None,
    **kwargs,
):

    swarm_robots = []
    sensors = []

    swarm_types = [VoronoiRobot] * (n_robots - n_anomal_robots)
    if n_anomal_robots > 0:
        if len(anomal_types) == 0:
            print("please speficy anomaly types")
        else:
            swarm_types += list(
                np.random.choice(
                    np.array(anomal_types),
                    replace=True,
                    p=p_anomal,
                    size=n_anomal_robots,
                )
            )

    for i, rtype in enumerate(swarm_types):
        new_robot, new_sensor = build_robot_fn(
            rtype,
            id=i,
            ts_communicate=ts_communicate,
            ts_control=ts_control,
            max_vel=max_vel,
            communication_ids=communication_ids,
            **kwargs,
        )
        swarm_robots.append(new_robot)
        sensors.append(new_sensor)

    robot_swarm = RobotSwarm(
        swarm_robots,
        ts_communicate=ts_communicate,
    )

    return robot_swarm, sensors


def build_simulation_robot(
    rtype: Callable[..., Robot],
    id: int,
    ts_control: float,
    max_vel: float,
    **kwargs,
):
    robot_simulation = Hera(id, use_lcm=False, ts_control=ts_control)
    sim_sensor = SimulatedSensor(robot_simulation, ts_control)

    state_handler = BasicStateHandler(sim_sensor, ts_control)
    swarm_communication_handler = BasicCommunicationHandler(sim_sensor)
    move_handler = SimulationMoveHandler(robot_simulation, ts_control)

    robot = rtype(
        id=id,
        state_handler=state_handler,
        move_handler=move_handler,
        swarm_communication_handler=swarm_communication_handler,
        ts_control=ts_control,
        max_vel=max_vel,
        wait_for_ts_communicate=False,
        **kwargs,
    )
    return robot, sim_sensor


def build_lcm_simulation_robot(
    rtype: Callable,
    id: int,
    start_area: PolygonWrapper,
    ts_communicate: float,
    ts_control: float,
    **kwargs,
):
    robot_simulation = Hera(id=id, use_lcm=True, ts_control=ts_control)
    robot_simulation.initialize_robot_state(start_area)

    sim_sensor = SimulatedOptitrack(id, robot_simulation)

    state_handler = LCM_StateHandler(id, communication_id=id, ts_control=ts_control)
    swarm_communication_handler = LCM_CommunicationHandler(id, ts_communicate)
    move_handler = LCM_MoveHandler(communication_id=id, ts_control=ts_control)

    robot = rtype(
        id=id,
        state_handler=state_handler,
        move_handler=move_handler,
        swarm_communication_handler=swarm_communication_handler,
        ts_communicate=ts_communicate,
        ts_control=ts_control,
        wait_for_ts_communicate=True,
        **kwargs,
    )
    return robot, sim_sensor


def build_lcm_robot(
    rtype: Callable,
    id: int,
    ts_communicate: float,
    ts_control: float,
    communication_ids,
    **kwargs,
):
    state_handler = LCM_StateHandler(
        id, communication_id=communication_ids[id], ts_control=ts_control, ttl=1
    )
    swarm_communication_handler = LCM_CommunicationHandler(id, ts_communicate, ttl=1)
    move_handler = LCM_MoveHandler(
        communication_id=communication_ids[id], ts_control=ts_control, ttl=1
    )

    robot = rtype(
        id=id,
        state_handler=state_handler,
        swarm_communication_handler=swarm_communication_handler,
        move_handler=move_handler,
        ts_communicate=ts_communicate,
        ts_control=ts_control,
        wait_for_ts_communicate=True,
        **kwargs,
    )
    return robot, None
