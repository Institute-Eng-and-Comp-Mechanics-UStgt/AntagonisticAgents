from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from src.anomaly_detectors.preprocessing import (
    compute_3d_action,
    compute_distance_features,
)
from src.deployment_area.voronoi_helpers import l2_norm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def prepare_data(
    df: pd.DataFrame,
    batch_size: int,
    dataset_class: Callable,
    train_perc: float = 0.8,
    val_perc: float = 0.1,
    test_perc: float = 0.1,
    weighted=False,
    **kwargs,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Dataset,
    Dataset,
    Dataset,
]:
    """Prepare datasets and dataloaders for training

    Args:
        df (pd.DataFrame): data
        batch_size (int): batch size
        dataset_class (Dataset): Custom Dataset of DataFrame
        **kwargs: e.g. features, context_neighbors, context_polygon, used for CustomDataset

    Returns:
        [DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset]: train, validation and test dataloaders and datasets
    """
    assert train_perc + val_perc + test_perc == 1

    # split the data in percentage of the simulation runs
    train_split_perc = len(df[df["run"] < df["run"].max() * train_perc]) / len(df)
    val_split_perc = (
        len(df[df["run"] < df["run"].max() * (train_perc + val_perc)]) / len(df)
        - train_split_perc
    )

    n_data = len(df)
    train_split = int(n_data * train_split_perc)
    val_split = int(n_data * (train_split_perc + val_split_perc))

    train_dataset = dataset_class(df=df.iloc[:train_split], **kwargs)
    val_dataset = dataset_class(df=df.iloc[train_split:val_split], **kwargs)
    test_dataset = dataset_class(df=df.iloc[val_split:], **kwargs)

    if weighted:
        weights = get_weights(df)
        train_sampler = WeightedRandomSampler(
            weights[:train_split], num_samples=train_dataset.__len__()
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler
        )
        val_sampler = WeightedRandomSampler(
            weights[train_split:val_split], num_samples=val_dataset.__len__()
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


class CustomDatasetTypedDistances(Dataset):
    def __init__(self, df, config, **kwargs):

        self.df = df
        self.config = config
        self.length = len(df)
        self.run_min = df["run"].min()
        self.run_max = df["run"].max()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # actions
        self.feature_data = df.loc[:, self.config["feature_names"]].to_numpy()
        if config["use_3dim_action"]:
            self.feature_data = compute_3d_action(
                self.feature_data, self.config["max_action"]
            )
        self.feature_data = torch.tensor(self.feature_data, dtype=torch.float32).to(
            device
        )

        # polygon context
        context_vertices = df.loc[:, self.config["context_polygon_vertices"]].to_numpy()
        context_edges = df.loc[:, self.config["context_polygon_edges"]].to_numpy()
        if "use_edges" in self.config and self.config["use_edges"]:
            self.context_polygon = np.concatenate(
                [context_vertices, context_edges], axis=-1
            )
        else:
            self.context_polygon = context_vertices

        self.context_polygon = compute_distance_features(
            self.context_polygon,
            input_dim=self.config["env_dim"],
            scale_features=self.config["scale_features"],
        )

        # neighbor context
        self.context_neighbors = df.loc[:, self.config["context_neighbors"]].to_numpy()

        self.context_neighbors = compute_distance_features(
            self.context_neighbors,
            input_dim=self.config["env_dim"],
            scale_features=self.config["scale_features"],
        )

        # convert to tensors
        self.context_neighbors = torch.tensor(
            self.context_neighbors, dtype=torch.float32
        )
        self.context_polygon = torch.tensor(self.context_polygon, dtype=torch.float32)

        # optionally add type information
        if self.config["embedding_net_fn_typed"]:
            if self.config["one_hot_type"]:
                self.context_neighbors = torch.nn.functional.pad(
                    torch.nn.functional.pad(
                        self.context_neighbors,
                        (0, 1),
                        "constant",
                        0,
                    ),
                    (0, 1),
                    "constant",
                    1,
                )
                self.context_polygon = torch.nn.functional.pad(
                    torch.nn.functional.pad(
                        self.context_polygon,
                        (0, 1),
                        "constant",
                        1,
                    ),
                    (0, 1),
                    "constant",
                    0,
                )
            else:
                self.context_neighbors = torch.nn.functional.pad(
                    self.context_neighbors,
                    (0, 1),
                    "constant",
                    -1,
                )
                self.context_polygon = torch.nn.functional.pad(
                    self.context_polygon,
                    (0, 1),
                    "constant",
                    1,
                )
        self.context = torch.concatenate(
            (self.context_neighbors, self.context_polygon), dim=1
        )
        self.context = self.context.to(device)

        self.feature_dim = 3

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        x = self.feature_data[idx]
        context = self.context[idx]

        return x, context


class CustomDatasetRNN_NSF(Dataset):
    def __init__(self, df, config, coverage_areas, **kwargs):

        self.config = config
        self.length = len(df)
        self.run_min = df["run"].min()
        self.run_max = df["run"].max()
        self.feature_data = df.loc[:, self.config["feature_names"]].to_numpy()

        context_neighbor_features = []
        context_polygon_features = []

        # scale and shift features
        for key, value in df.groupby(["run"]):

            ca = coverage_areas[key[0]]
            box = ca.polygon.bounds
            ext, *_ = ca.compute_extension()

            context_neighbors = (
                value[self.config["context_neighbors"]]
                .apply(lambda x: x / ext, axis=1, raw=True, result_type="expand")
                .to_numpy()
            )

            context_polygon = (
                value[self.config["context_polygon"]]
                .apply(lambda x: x / ext, axis=1, raw=True, result_type="expand")
                .to_numpy()
            )

            context_neighbor_features.append(context_neighbors)
            context_polygon_features.append(context_polygon)

        self.context_neighbors = np.vstack(context_neighbor_features)
        self.context_polygon = np.vstack(context_polygon_features)

        self.feature_dim = 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        x = torch.tensor(self.feature_data[idx], dtype=torch.float32)

        pi_n = torch.tensor(self.context_neighbors[idx], dtype=torch.float32).reshape(
            -1, self.feature_dim
        )
        if self.config["shuffle_context"]:
            pi_n = pi_n[torch.randperm(pi_n.shape[0]), :]

        pi_p = torch.tensor(self.context_polygon[idx], dtype=torch.float32).reshape(
            -1, self.feature_dim
        )
        if self.config["shuffle_context"]:
            pi_p = pi_p[torch.randperm(pi_p.shape[0]), :]

        return x, {"neighbour": pi_n, "poly": pi_p}


def get_weights(df: pd.DataFrame) -> Sequence[float]:
    """Weight actions by their l2_norm due to higher probability of small actions.

    Args:
        df (DataFrame): data

    Returns:
        np.ndarray: weights
    """
    l2_norms = l2_norm(df.loc[:, ["action_x", "action_y"]].to_numpy())
    n, edges = np.histogram(l2_norms, bins=50, density=True)
    edges[-1] += 1e-4
    # bins with a small number of samples get higher weights
    bin_weights = np.max(n) / n
    bin_idx = np.digitize(l2_norms, edges) - 1
    weights = bin_weights[bin_idx]

    return weights
