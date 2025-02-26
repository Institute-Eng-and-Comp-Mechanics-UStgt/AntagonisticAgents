import asyncio
import pickle
import time
from datetime import datetime

import numpy as np
import yaml
from hardware_simulation.build_deployment_area import build_deployment_area
from itm_pythonfig.pythonfig import PythonFig
from matplotlib import pyplot as plt
from src.anomaly_detectors.models.rnn_nsf import RNN_NSF
from src.anomaly_detectors.save_load_functions import load_model
from src.deployment_area.polygon import PolygonWrapper
from src.robots.robot_helpers import (
    build_lcm_robot,
    build_lcm_simulation_robot,
    build_random_robot_swarm,
    save_robot_swarm,
)


async def run():
    # initialize
    with open("hw_config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    ad = None

    boundary = PolygonWrapper(np.array([[0, 0], [4.5, 0], [4.5, 3.5], [0, 3.5]]))
    (
        robot_swarm,
        s,
    ) = build_random_robot_swarm(
        n_robots=cfg["n_robots"],
        build_robot_fn=build_lcm_robot,
        max_vel=cfg["max_vel"],
        ts_communicate=cfg["ts_communicate"],
        ts_control=cfg["ts_control"],
        n_anomal_robots=cfg["n_anomal_robots"],
        anomal_types=cfg["anomal_types"],
        communication_ids=cfg["communication_ids"],
        start_area=boundary,
    )
    robot_swarm.set_anomaly_detector(ad)
    robot_swarm.start_run(deployment_area=boundary, external_state_monitors=s)

    for i in range(10):
        run_id = cfg["run_id"] + i
        print("run: ", run_id)

        deployment_area = build_deployment_area(boundary, robot_swarm)
        robot_swarm.set_area_bounds(deployment_area)

        # cover area
        await robot_swarm.cover_area(cfg["conv_crit"], cfg["max_steps"])

        # plot
        pf = PythonFig()
        fig = pf.start_figure("PAMM", 10, 10)
        ax = fig.gca()
        robot_swarm.visualization_module.plot_coverage(fig=fig, ax=ax)
        pf.finish_figure(
            file_path=f"./hardware_data/test_lcm_simulate/swarm{run_id}.png"
        )

        # save data
        save_robot_swarm(
            robot_swarm, f"./hardware_data/test_lcm_simulate/swarm{run_id}"
        )
        print("run finished!")

        # reset
        for robot in robot_swarm.swarm_robots:
            # stop the robot
            robot.move_handler.trigger_movement(np.array([0, 0]))
            robot.reset()

        time.sleep(10)

    robot_swarm.stop_run(s)


if __name__ == "__main__":
    asyncio.run(run())
