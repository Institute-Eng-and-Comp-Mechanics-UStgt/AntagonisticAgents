from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING, Callable, Dict

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from src.anomaly_detectors.models.lstm_coupling_nf import EmbeddingNetTypedLSTM
from src.anomaly_detectors.preprocessing import (
    compute_2d_action,
    compute_3d_action,
    compute_distance_features,
    compute_distance_vectors_robots_to_entity,
    compute_shortest_distance_robot_to_edges,
)
from src.deployment_area.voronoi_helpers import l2_norm
from src.robots.robot_helpers import load_robot_swarm

if TYPE_CHECKING:
    from src.anomaly_detectors.detection_methods import DetectionMethod
    from src.anomaly_detectors.models.rnn_nsf import RNN_NSF
    from src.deployment_area.polygon import PolygonWrapper
    from src.robots.robot import Robot
    from src.robots.robot_swarm import RobotSwarm


class AnomalyDetector(metaclass=abc.ABCMeta):

    def __init__(
        self,
        detector: torch.nn.Module,
    ):
        self.detector = detector
        self.detection_method = None

    def set_and_tune_detection_method(
        self,
        detection_method: DetectionMethod,
        validation_data_folder: str,
        false_positive_rate: float,
    ) -> None:

        val_log_probs = []
        files = [
            name
            for name in os.listdir(validation_data_folder)
            if name.startswith("rs_")
        ]
        for f in files:
            torch.manual_seed(0)
            np.random.seed(0)
            rs = load_robot_swarm(f"{validation_data_folder}{f}")
            rs.set_anomaly_detector(self)
            assert rs.anomaly_detector is not None
            lp, _ = rs.anomaly_detector.evaluate_run()
            lp = np.vstack(lp)
            val_log_probs.append(lp)

        self.detection_method = detection_method
        self.detection_method.tune(val_log_probs, false_positive_rate)

    def initialize_run_prediction(self, robot_swarm: RobotSwarm) -> None:
        self.prediction_step = 0
        self.robot_swarm = robot_swarm
        self.n_robots = robot_swarm.n_robots

        self.action_log_probs = []
        self.sample_log_probs = []
        self.prediction_history = [np.zeros(shape=self.n_robots, dtype=bool)]
        return

    def evaluate_run(self, n_samples: int = 100, verbose=False) -> tuple[list, list]:

        positions = np.array(
            [
                [s.position for s in robot.get_communicated_state_history()]
                for robot in self.robot_swarm.swarm_robots
            ]
        ).transpose(1, 0, 2)
        actions = positions[1:] - positions[:-1]

        for pos, action in zip(positions[:-1], actions):
            self.evaluate_step(
                pos,
                action,
                self.robot_swarm.deployment_area,
                n_samples,
                mask_anomal=False,
            )

        if verbose:
            print(self.prediction_history[-1])
        return self.action_log_probs, self.sample_log_probs

    def get_anomaly_prediction_of_swarm(self) -> np.ndarray:
        if self.detection_method is None:
            print("Please set and tune a detection method!")
        return self.prediction_history[-1]

    def is_robot_anomal(self, robot: Robot, timestep: int) -> bool:
        if self.detection_method is None:
            print("Please set and tune a detection method!")
            return False
        elif len(self.prediction_history) > timestep:
            return self.prediction_history[timestep][robot.id]
        else:
            print("The prediction for this timestep is not available")
            return self.prediction_history[-1][robot.id]

    @abc.abstractmethod
    def evaluate_step(
        self,
        pos: np.ndarray,
        actions: np.ndarray,
        deployment_area: PolygonWrapper,
        n_samples: int,
        mask_anomal: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_robot_actions(
        self, positions: np.ndarray, deployment_area: PolygonWrapper, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


# TODO: function for anomaly detection for a single robot?
class Coverage_Anomaly_Detector(AnomalyDetector):

    def __init__(
        self,
        detector: torch.nn.Module,
        config: Dict,
    ):
        super().__init__(detector)
        self.config = config

    def evaluate_step(
        self,
        pos: np.ndarray,
        actions: np.ndarray,
        deployment_area: PolygonWrapper,
        n_samples: int,
        mask_anomal: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

        state = self.preprocess_state(pos, deployment_area, mask_anomal)
        if self.config["use_3dim_action"]:
            actions = compute_3d_action(actions, self.config["max_action"])
        action_log_probs = self.detector.log_prob(
            torch.tensor(actions, dtype=torch.float32).to(self.detector.device),
            state.to(self.detector.device),
        )
        self.action_log_probs.append(action_log_probs.detach().cpu().numpy())
        # np.clip(action_log_probs.detach().cpu().numpy(), a_min=np.log(1e-10),
        _, sample_log_probs = self.detector.sample_and_log_prob(n_samples, state)
        self.sample_log_probs.append(sample_log_probs.detach().cpu().numpy())

        if self.detection_method is not None:
            anomal_robots_found = self.detection_method(self.action_log_probs)
            self.prediction_history.append(anomal_robots_found)

        return (action_log_probs, sample_log_probs)

    def sample_robot_actions(
        self,
        positions: np.ndarray,
        deployment_area: PolygonWrapper,
        n_samples: int = 150,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self.preprocess_state(positions, deployment_area)

        action_samples, sample_log_prob = self.detector.sample_and_log_prob(
            num_samples=n_samples, context=state.to(self.detector.device)
        )
        action_samples = action_samples.detach().cpu().numpy()
        sample_log_prob = sample_log_prob.detach().cpu().numpy()

        # TODO: max_vel
        if self.config["use_3dim_action"]:
            action_samples = compute_2d_action(
                action_samples,
                self.robot_swarm.swarm_robots[0].max_vel
                * self.robot_swarm.ts_communicate,
            )
        return (
            action_samples,
            sample_log_prob,
        )

    def preprocess_state(
        self,
        robot_positions: np.ndarray,
        coverage_area: PolygonWrapper,
        mask_anomal: bool = False,
    ) -> torch.Tensor:
        env_dim = self.config["env_dim"]

        context_vertices = compute_distance_vectors_robots_to_entity(
            entity_pos=coverage_area.vertices[:-1],
            robot_pos=robot_positions,
            max_dim_distance_vectors=self.config["n_max_vertices"] * env_dim,
        )
        context_edges = compute_shortest_distance_robot_to_edges(
            coverage_area=coverage_area,
            robot_pos=robot_positions,
            max_dim_distance_vectors=self.config["n_max_vertices"] * env_dim,
        )
        if "use_edges" in self.config and self.config["use_edges"]:
            context_polygon = np.concatenate([context_vertices, context_edges], axis=-1)
        else:
            print("only using vertices")
            context_polygon = context_vertices

        context_neighbors = compute_distance_vectors_robots_to_entity(
            entity_pos=robot_positions,
            robot_pos=robot_positions,
            max_dim_distance_vectors=self.config["n_max_robots"] * env_dim,
        )

        context_polygon = compute_distance_features(
            context_polygon,
            input_dim=self.config["env_dim"],
            scale_features=self.config["scale_features"],
        )
        context_neighbors = compute_distance_features(
            context_neighbors,
            input_dim=self.config["env_dim"],
            scale_features=self.config["scale_features"],
        )
        if mask_anomal:
            context_neighbors[:, np.where(self.get_anomaly_prediction_of_swarm())] = 0

        context_polygon = torch.tensor(context_polygon, dtype=torch.float32)
        context_neighbors = torch.tensor(context_neighbors, dtype=torch.float32)

        assert type(self.detector._embedding_net) == EmbeddingNetTypedLSTM
        if self.config["one_hot_type"]:
            context_polygon = torch.nn.functional.pad(
                torch.nn.functional.pad(
                    context_polygon,
                    (0, 1),
                    "constant",
                    1,
                ),
                (0, 1),
                "constant",
                0,
            )
            context_neighbors = torch.nn.functional.pad(
                torch.nn.functional.pad(
                    context_neighbors,
                    (0, 1),
                    "constant",
                    0,
                ),
                (0, 1),
                "constant",
                1,
            )
        else:
            context_polygon = torch.nn.functional.pad(
                context_polygon,
                (0, 1),
                "constant",
                1,
            )
            context_neighbors = torch.nn.functional.pad(
                context_neighbors,
                (0, 1),
                "constant",
                -1,
            )
        state = torch.concatenate((context_neighbors, context_polygon), dim=1)

        return state

    def compute_anchor_data(
        self, robot_positions: np.ndarray, coverage_area: PolygonWrapper
    ) -> torch.Tensor:
        robot_dists = compute_shortest_distance_robot_to_edges(
            coverage_area=coverage_area,
            robot_pos=robot_positions,
            max_dim_distance_vectors=self.detector.n_context,
        )
        n_robots = robot_positions.shape[0]
        d = robot_positions.shape[1]

        # for each robot compute the closest point on each edge
        anchor_points = robot_positions[:, None] + robot_dists.reshape(n_robots, -1, d)
        # compute the distance of all robots to the anchors points of each robot
        distances = anchor_points[:, None] - robot_positions[:, None]
        state = torch.tensor(distances, dtype=torch.float32).permute(0, 3, 1, 2)

        return state


class RNN_NSF_Detector(AnomalyDetector):

    def __init__(
        self,
        detector: RNN_NSF,
    ):
        super().__init__(detector)

    def evaluate_step(
        self,
        pos: np.ndarray,
        actions: np.ndarray,
        deployment_area: PolygonWrapper,
        n_samples: int,
        mask_anomal: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

        state = self.preprocess_state(pos, deployment_area)
        action_log_probs = self.detector.log_prob(
            torch.tensor(actions, dtype=torch.float32), state
        )
        self.action_log_probs.append(action_log_probs.detach().cpu().numpy())
        _, sample_log_probs = self.detector.sample_and_log_prob(n_samples, state)
        self.sample_log_probs.append(sample_log_probs.detach().cpu().numpy())

        if self.detection_method is not None:
            anomal_robots_found = self.detection_method(self.action_log_probs)
            self.prediction_history.append(anomal_robots_found)

        return (action_log_probs, sample_log_probs)

    def sample_robot_actions(
        self,
        positions: np.ndarray,
        deployment_area: PolygonWrapper,
        n_samples=100,
    ) -> tuple[np.ndarray, np.ndarray]:

        state = self.preprocess_state(positions, deployment_area)

        all_action_samples = []
        all_lp = []
        for robot_id in range(self.n_robots):
            action_samples, sample_log_prob = self.detector.sample_and_log_prob(
                num_samples=n_samples,
                context={
                    "neighbour": state["neighbour"][robot_id][None, :],
                    "poly": state["poly"][robot_id][None, :],
                },
            )
            all_action_samples.append(action_samples.detach().cpu().numpy())
            all_lp.append(sample_log_prob.detach().cpu().numpy())

        return np.vstack(all_action_samples), np.vstack(all_lp)

    def preprocess_state(self, pos: np.ndarray, coverage_area: PolygonWrapper) -> Dict:
        neighbor_dists, poly_dists = compute_sorted_dists(
            coverage_area,
            pos,
            n_max_nodes=len(coverage_area.vertices),
            n_max_neighbors=pos.shape[0] - 1,
        )

        state = {
            "neighbour": torch.tensor(neighbor_dists, dtype=torch.float32).reshape(
                self.n_robots, -1, 2
            ),
            "poly": torch.tensor(poly_dists, dtype=torch.float32).reshape(
                self.n_robots, -1, 2
            ),
        }

        return state


## helpers


def compute_sorted_dists(
    coverage_area: PolygonWrapper,
    pos: np.ndarray,
    n_max_nodes: int,
    n_max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each robot compute the euclidean distance to all other robots and the polygon nodes
    and sort by smallest distance.
    Args:
        poly (PolygonWrapper): coverage area
        pos (np.array): robot positions
        n_neighbors (int): number of nearest neighbors considered
        n_max_nodes (int): maximum number of polygon nodes
    Returns:
        np.array: (n_robots, n_robots)
            Sorted euclidean distances.
    """

    n_robots = pos.shape[0]
    n_neighbors = n_robots - 1
    pos, poly = shift_and_scale_coords(pos, coverage_area)

    # compute the sorted distances to the robot neighbors
    A = kneighbors_graph(pos, n_neighbors=n_neighbors, mode="connectivity").toarray()  # type: ignore
    robot_dist = pos - pos[:, np.newaxis]
    robot_dist = robot_dist[A.astype(bool)].reshape(n_robots, n_neighbors, -1)
    sort_by_l2_idx = np.argsort(l2_norm(robot_dist), axis=-1)[:, ::-1]
    robot_dist_sorted = robot_dist[
        np.arange(robot_dist.shape[0])[:, None], sort_by_l2_idx
    ].reshape(n_robots, -1)

    neighbors_max_size = np.zeros((n_robots, n_max_neighbors * 2))
    neighbors_max_size[:, : n_neighbors * 2] = robot_dist_sorted

    # compute the sorted distances to the polygon nodes, ignore the last repeated node
    poly_dist = poly[:-1] - pos[:, np.newaxis]
    sort_by_l2_idx = np.argsort(l2_norm(poly_dist), axis=-1)[:, ::-1]
    poly_dist_sorted = poly_dist[
        np.arange(poly_dist.shape[0])[:, None], sort_by_l2_idx
    ].reshape(n_robots, -1)
    # pad the distances with zeros to return an array of size (n_robots, n_max_nodes) regardless of the actual number of polygon nodes
    n_nodes = poly_dist_sorted.shape[1]
    node_diff = n_max_nodes * 2 - n_nodes
    poly_dist_sorted_padded = np.pad(
        poly_dist_sorted, pad_width=[[0, 0], [0, node_diff]], constant_values=0
    )

    return neighbors_max_size, poly_dist_sorted_padded


def sort_2d_features_by_l2_norm(distance_vector) -> np.ndarray:
    # sort distance vectors from largest to smallest distance

    distance_vector_reshaped = distance_vector.reshape(distance_vector.shape[0], -1, 2)
    sort_by_l2_idx = np.argsort(l2_norm(distance_vector_reshaped), axis=-1)[:, ::-1]
    distance_vector = distance_vector_reshaped[
        np.arange(distance_vector_reshaped.shape[0])[:, None], sort_by_l2_idx
    ].reshape(distance_vector.shape[0], -1)

    return distance_vector


def shift_and_scale_coords(
    robot_robot_positions: np.ndarray, coverage_area: PolygonWrapper
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift the polygon nodes and robot robot_positions such that the polygon bounds lie within 0 and 1.

    Args:
        robot_robot_positions (np.array): robot robot_positions
        coverage_area (PolygonWrapper): coverage area

    Returns:
        np.array: shifted robot robot_positions
        np.array: shifted coverage area nodes
    """
    rp = robot_robot_positions.copy()
    ca = coverage_area.vertices.copy()
    # shift all coords such that the lower left corner of the rectangle enclosing the polygon is at 0/0
    box = coverage_area.polygon.bounds
    ext, *_ = coverage_area.compute_extension()
    ll = (box[0], box[1])
    ca = (ca - ll) / ext
    rp = (rp - ll) / ext

    return rp, ca
