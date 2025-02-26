from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import shapely
import torch
from sklearn.neighbors import kneighbors_graph

from src.deployment_area.voronoi_helpers import l2_norm

if TYPE_CHECKING:
    from src.deployment_area.polygon import PolygonWrapper
    from src.robots.robot_swarm import RobotSwarm


def compute_data(
    config: dict,
    run: int,
    step: int,
    robot_swarm: RobotSwarm,
    coverage_area: PolygonWrapper,
    prev_robot_pos: np.ndarray,
    robot_pos: np.ndarray,
    actions: np.ndarray,
    optimal_targets: np.ndarray,
    dim: int = 2,
) -> np.ndarray:
    """
    For each robot compute the distance to all other robots and one polygon edge as well as the loss of the corresponding
    voronoi cell. Stack the data into an array with columns [run, step number, robot index, anomaly label, loss, last robot
    robot_position, action, optimal action, distance to polygon edge, distance to robots (x number of robots)].

    Args:
        config (dict): Configuration.
        run (int): Run id.
        step (int): Current timestep in this run.
        robot_swarm (RobotSwarm)
        coverage_area (PolygonWrapper)
        prev_robot_pos (np.array):
            Robot robot_positions before performing an action.
        robot_pos (np.array):
            Robot robot_positions after performing an action.
        action (np.array):
            Last action performed.
        optimal_targets (np.array):
            Optimal target robot_positions of a robot.
            Relevant in case of obstacles, different objective function, failure etc.
        n_neighbors (int):
            Number of robot neighbors used for the creation of distance features.
        n_max_nodes: (int)
            Maximum number of polygon nodes.

    Returns:
        np.array: data
    """
    robot_dist = compute_distance_vectors_robots_to_entity(
        entity_pos=prev_robot_pos,
        robot_pos=prev_robot_pos,
        max_dim_distance_vectors=config["n_max_robots"] * dim,
    )

    vertex_dist = compute_distance_vectors_robots_to_entity(
        entity_pos=coverage_area.vertices[:-1],
        robot_pos=prev_robot_pos,
        max_dim_distance_vectors=config["n_max_vertices"] * dim,
    )

    edge_dist = compute_shortest_distance_robot_to_edges(
        coverage_area=coverage_area,
        robot_pos=prev_robot_pos,
        max_dim_distance_vectors=config["n_max_vertices"] * dim,
    )

    n_robots = robot_swarm.n_robots
    run_rep = np.repeat(run, n_robots)
    step_rep = np.repeat(step, n_robots)
    robot_idx = np.arange(0, n_robots)
    label = [robot.label for robot in robot_swarm.swarm_robots]

    data = np.vstack((run_rep, step_rep, robot_idx, np.array(label)))
    data = np.hstack(
        (
            data.T,
            prev_robot_pos,
            robot_pos,
            actions,
            optimal_targets,
            robot_dist,
            vertex_dist,
            edge_dist,
        )
    )

    return data


def compute_3d_action(actions, max_action_per_dim):

    assert (
        actions.shape[-1] == 2
    ), f"Action dimension must be 2, but is {actions.shape}."

    l2_distance = l2_norm(actions)[..., None] + 1e-10
    normalized_action_vec = actions / l2_distance

    max_action = np.sqrt(2 * max_action_per_dim**2)
    motion_perc = l2_distance / max_action

    actions = np.concatenate(
        (normalized_action_vec, motion_perc.reshape(-1, 1)), axis=-1
    )
    return actions


def compute_2d_action(actions, max_action):

    assert (
        actions.shape[-1] == 3
    ), f"Action dimension must be 3, but is {actions.shape}."

    motion_perc = actions[..., -1]
    max_action = np.sqrt(2 * max_action**2)

    abs_motion = motion_perc * max_action
    actions = actions[..., :2] * abs_motion.reshape(*actions.shape[:-1], 1)

    return actions


def test_action_conversion(max_action):

    test_array = np.random.rand(10, 2)
    assert (
        (
            compute_2d_action(
                compute_3d_action(test_array, max_action),
                max_action,
            )
            - test_array
        )
        < 1e-6
    ).all()


def compute_distance_features(distance_features, input_dim, scale_features):

    distance_vec = distance_features.reshape(distance_features.shape[0], -1, input_dim)
    l2_distance = l2_norm(distance_vec)[..., None] + 1e-10
    normalized_distance_vec = distance_vec / l2_distance

    if scale_features:
        scaled_distance = compute_scaled_distance(l2_distance)
        features = np.concatenate((normalized_distance_vec, scaled_distance), axis=-1)
    else:
        features = np.concatenate((normalized_distance_vec, l2_distance), axis=-1)
    return features


def compute_scaled_distance(l2_distance):

    # maximum distance based on area per robot
    max_area = 500
    max_ext = 2 * np.sqrt(max_area)
    max_dist = np.sqrt(max_ext**2 * 2)
    # minimum distance based on HERA radius
    min_dist = 0.29 / 2

    l2_distance_clipped = np.clip(l2_distance, a_min=min_dist, a_max=max_dist)
    scaled_l2_distance = (np.log(l2_distance_clipped) - np.log(min_dist)) / (
        np.log(max_dist) - np.log(min_dist)
    )
    return scaled_l2_distance


def compute_distance_vectors_robots_to_entity(
    entity_pos: np.ndarray,
    robot_pos: np.ndarray,
    max_dim_distance_vectors: int,
) -> np.ndarray:
    """
    For each robot position compute the distance vector to all entity positions (e.g. other robots, polygon vertices).

    Args:
        entity_pos (np.array): positions of entities to compute the distance to
        robot_pos (np.array): robot positions
        max_dim_distance_vectors (int): maximum number of entities times the dimension of entity position features

    Returns:
        np.array: (n_robots, n_robots)
            Euclidean distance vectors.
    """
    n_robots = robot_pos.shape[0]
    # compute the distances to the robot neighbors
    distance_vectors = entity_pos - robot_pos[:, np.newaxis]
    distance_vectors = distance_vectors.reshape(n_robots, -1)

    # pad the vectors with 0 to max_dim_distance_vectors
    distance_vectors_max_size = np.zeros((n_robots, max_dim_distance_vectors))
    distance_vectors_max_size[:, : distance_vectors.shape[1]] = distance_vectors

    return distance_vectors_max_size


def compute_shortest_distance_robot_to_edges(
    coverage_area: PolygonWrapper,
    robot_pos: np.ndarray,
    max_dim_distance_vectors: int,
) -> np.ndarray:
    """
    For each robot position compute the distance vector to all entity positions (e.g. other robots, polygon vertices).

    Args:
        coverage_area (PolygonWrapper): coverage area
        robot_pos (np.array): robot positions
        max_dim_distance_vectors (int): maximum number of edges times the dimension of their position features

    Returns:
        np.array: (n_robots, n_robots)
            Euclidean distance vectors.
    """
    vecs_shortest_dist = []

    for v1, v2 in zip(coverage_area.vertices[:-1], coverage_area.vertices[1:]):
        edge = v2 - v1
        # vector orthogonal to edge
        orth_edge = np.array([-edge[1], edge[0]])

        # find the point on the edge that has the shortest distance to the robot position by computing the intersection between the edge and the line orthogonal to the edge that passes through the robot position
        l1 = shapely.LineString([v1, v2])
        _, ext2d, *_ = coverage_area.compute_extension()
        l2 = [
            shapely.LineString([p - ext2d * orth_edge, p + ext2d * orth_edge])
            for p in robot_pos
        ]
        intersection = l1.intersection(l2)
        # if no intersection is found, the shortest distance is the distance between the robot position and one of the edge's vertices
        closest_vertex = np.vstack([v1, v2])[
            np.argmin(
                np.array([l2_norm(v1 - robot_pos), l2_norm(v2 - robot_pos)]), axis=0
            )
        ]
        targets = np.vstack(
            [
                [isec.x, isec.y] if isec else node
                for isec, node in zip(intersection, closest_vertex)
            ]
        )
        distance_vec = targets - robot_pos
        vecs_shortest_dist.append(distance_vec)

    # pad the vectors with 0 to max_dim_distance_vectors
    n_robots = robot_pos.shape[0]
    distance_vectors = np.hstack(vecs_shortest_dist)
    distance_vectors_max_size = np.zeros((n_robots, max_dim_distance_vectors))
    distance_vectors_max_size[:, : distance_vectors.shape[1]] = distance_vectors

    return distance_vectors_max_size
