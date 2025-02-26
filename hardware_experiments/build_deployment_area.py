from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from src.deployment_area.polygon import PolygonWrapper

if TYPE_CHECKING:
    from src.robots.robot_swarm import RobotSwarm


def build_deployment_area(boundary: PolygonWrapper, robot_swarm: RobotSwarm):

    deployment_area = None
    x_ext = boundary.vertices[1, 0]
    y_ext = boundary.vertices[2, 1]

    area_usable = False
    while not area_usable:
        n_nodes = np.random.choice([3, 4, 5, 6])
        nodes = np.hstack(
            [
                np.random.uniform(low=-0.2, high=x_ext, size=(n_nodes, 1)),
                np.random.uniform(low=-0.2, high=y_ext, size=(n_nodes, 1)),
            ]
        )
        deployment_area = PolygonWrapper(nodes).compute_convex_hull()
        area_usable = deployment_area.compute_polygon_area() > 3

        if robot_swarm:
            for robot in robot_swarm.swarm_robots:
                area_usable = area_usable and deployment_area.contains_point(
                    robot._get_current_position()
                )
    print("cov area found")
    assert deployment_area is not None
    return deployment_area
