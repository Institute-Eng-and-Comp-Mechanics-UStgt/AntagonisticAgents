import math
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from itm_pythonfig.pythonfig import PythonFig
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas import DataFrame
from shapely.geometry import Polygon
from torch.utils.data import DataLoader, Dataset

from src.anomaly_detectors.preprocessing import compute_2d_action, compute_3d_action
from src.deployment_area.polygon import PolygonWrapper
from src.deployment_area.voronoi_helpers import l2_norm
from src.robots.deployment_robot import VoronoiRobot
from src.robots.robot_swarm import RobotSwarm


def plot_action_density(
    idx: int,
    dataset: Dataset,
    trained_model: torch.nn.Module,
    max_action: float,
    use_3dim_action: bool,
) -> Figure:

    # set params
    n_samples = 10000
    action, context = dataset.__getitem__(idx)

    bounds = max_action

    # compute density over grid
    n_samples_per_dim = int(np.sqrt(n_samples))
    mg = np.meshgrid(
        np.linspace(-bounds, bounds, n_samples_per_dim)[:, None],
        np.linspace(-bounds, bounds, n_samples_per_dim),
    )
    mg = np.vstack((mg[0].flatten(), mg[1].flatten())).T
    if use_3dim_action:
        action_grid = compute_3d_action(mg, bounds)
    else:
        action_grid = mg
    action_grid = torch.tensor(action_grid, dtype=torch.float32).squeeze()

    all_actions_log_density = trained_model.log_prob(
        action_grid, torch.repeat_interleave(context[None, :], action_grid.shape[0], 0)
    )

    # action log probability estimation
    action_logprob = trained_model.log_prob(action[None, :], context[None, :]).item()
    n_samples = 100
    _, sample_log_prob = trained_model.sample_and_log_prob(n_samples, context[None, :])
    conf = torch.sum(sample_log_prob < action_logprob, dim=-1).item() / n_samples

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    im = ax[0].scatter(
        mg[:, 0],
        mg[:, 1],
        c=all_actions_log_density.detach().numpy(),
        cmap="winter",
        s=5,
        alpha=0.8,
    )
    plt.colorbar(im)
    if use_3dim_action:
        action = compute_2d_action(action.cpu(), max_action)
    ax[0].scatter(action[0], action[1], color="red", s=20)
    ax[0].set_title(f"lp: {np.round(action_logprob, 3)}, conf: {np.round(conf, 3)}")

    samples, lp = trained_model.sample_and_log_prob(1000, context[None, ...])
    samples = samples.cpu().detach().numpy().squeeze()
    lp = lp.detach().numpy().squeeze()
    mask = lp < trained_model.get_uniform_log_prob().numpy()
    if use_3dim_action:
        samples = compute_2d_action(samples, bounds)
    ax[1].scatter(samples[mask][:, 0], samples[mask][:, 1], s=20, color="orange")
    im = ax[1].scatter(samples[:, 0], samples[:, 1], s=10, c=lp)
    plt.colorbar(im)
    ax[1].scatter(action.cpu()[0], action.cpu()[1], s=20, color="red")

    return fig


def plot_robot_swarm_run(robot_swarm, detector):

    robot_swarm.set_anomaly_detector(detector)
    assert robot_swarm.anomaly_detector is not None
    log_prob = robot_swarm.anomaly_detector.detect_all_anomalies()

    pf = PythonFig()
    fig = pf.start_figure("PAMM", 10, 10)
    robot_swarm.visualization_module.plot_coverage(
        fig=fig, plot_intermediate_positions=False, plot_samples=True
    )
    fig.suptitle(str(np.round(log_prob, 3)))
    return fig


def plot_noise(
    model: torch.nn.Module, dataloader: DataLoader, plot: bool = True
) -> Figure:
    noise_batches = []
    with torch.no_grad():
        model.eval()
        for _, (x, pi) in enumerate(dataloader):
            noise = model.transform_to_noise(
                x,
                context=pi,
            )
            noise_batches.append(noise.cpu().detach().numpy())

    if not plot:
        plt.ioff()
    noise = np.vstack(noise_batches)

    if noise.shape[-1] == 2:
        cols = ["x", "y"]
    else:
        assert noise.shape[-1] == 3
        cols = ["x", "y", "theta"]
    noise_data = pd.DataFrame(data=noise, columns=cols)

    sns.set_theme(style="white")
    g = sns.PairGrid(noise_data, diag_sharey=True)
    g.map_upper(sns.scatterplot, s=5, alpha=0.2)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.kdeplot, lw=2, fill=True)
    plt.subplots_adjust(top=0.9)
    fig = plt.gcf()

    return fig
