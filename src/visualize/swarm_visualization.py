from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from itm_pythonfig.pythonfig import PythonFig
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import Voronoi
from src.anomaly_detectors.preprocessing import compute_2d_action
from src.deployment_area.polygon import PolygonWrapper, voronoi_plot_2d
from src.robots.deployment_robot import SpoofingRobot

if TYPE_CHECKING:
    from src.robots.deployment_robot import Robot
    from src.robots.robot_swarm import RobotSwarm


class SwarmVisualization:
    def __init__(self, robot_swarm: RobotSwarm) -> None:
        self.robot_swarm = robot_swarm

    def plot_coverage(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        plot_samples: bool = False,
        animate: bool = False,
        plot_square: bool = False,
    ):
        if ax == None:
            if fig == None:
                pf = PythonFig()
                fig = pf.start_figure("PAMM", 11, 11)
            ax = fig.gca()
            ax.clear()
        plt.ioff()

        # plot polygon once without voronoi cells
        self.robot_swarm.deployment_area.plot_polygon(
            fig=fig, ax=ax, plot_square=plot_square
        )
        # plot the weighted areas of the weighted voronoi robots
        self.plot_weighted_areas(ax)
        # plot the robot positions
        all_scatter_points = self.plot_positions(ax, animate)
        # legend and labels
        self.build_legend(ax)

        if animate:
            # create an animation
            images = self.collect_animation_images(ax, all_scatter_points)
            assert type(fig) == Figure
            line_anim = ArtistAnimation(
                fig,
                images,
                interval=150,
                blit=True,
            )
            return line_anim

        # plot actions and action samples
        self.plot_actions(ax)
        self.plot_action_samples(
            plot_samples,
            ax,
            self.robot_swarm.deployment_area,
        )
        return fig, ax

    def collect_animation_images(self, ax, all_scatter_points):

        images = []
        kwargs = {}
        kwargs["show_points"] = False
        kwargs["show_vertices"] = False
        start_positions = self.get_start_positions()

        # intermediate steps per action, exclude last position
        n_intermediate_steps = int(
            (all_scatter_points.shape[0] - 1) / (start_positions.shape[0] - 1)
        )
        scatter_points = all_scatter_points[:-1].reshape(
            -1, n_intermediate_steps, all_scatter_points.shape[-1]
        )
        for ts, (pos, points) in enumerate(zip(start_positions, scatter_points)):
            vor = Voronoi(pos)
            _, vlines = voronoi_plot_2d(vor, ax, **kwargs)
            vlines = ax.add_collection(vlines)

            sp = self.plot_spoofing_communication(ts, ax)
            for p in points:
                img = list(p) + [vlines] + sp
                images.append(img)

        last_point = all_scatter_points[-1]
        last_pos = start_positions[-1]
        vor = Voronoi(last_pos)
        _, vlines = voronoi_plot_2d(vor, ax, **kwargs)
        img = list(last_point) + [vlines]
        images.append(img)

        return images

    def save_animation_images(self, folder_path):

        frame = 0
        pf = PythonFig()
        start_positions = self.get_start_positions()

        for ts, start_pos in enumerate(start_positions):
            # Voronoi tesselation
            vor = Voronoi(start_pos)

            all_robot_positions = []
            for robot in self.robot_swarm.swarm_robots:
                state = robot.get_state_history()[ts]
                if len(state.intermediate_positions) > 0:
                    positions = np.vstack(
                        (state.position, np.vstack(state.intermediate_positions))
                    )
                else:
                    positions = state.position[None, :]
                all_robot_positions.append(positions)
            all_robot_positions = np.array(all_robot_positions).transpose(1, 0, 2)

            for swarm_pos in all_robot_positions:

                fig = pf.start_figure("PAMM", 13.97, 10)
                ax = fig.gca()
                ax.set_xticks([], labels=None)
                ax.set_yticks([], labels=None)
                ax.grid()
                ax.set_xlabel("")
                ax.set_ylabel("")

                self.robot_swarm.deployment_area.plot_polygon(fig=fig, ax=ax)
                voronoi_plot_2d(
                    vor, ax, **{"show_points": False, "show_vertices": False}
                )
                for p, robot in zip(swarm_pos, self.robot_swarm.swarm_robots):
                    # colored markers for robots
                    facecolor, legend_marker, marker_size, marker_line = (
                        self.get_markers(robot, ts)
                    )
                    ax.scatter(
                        p[0],
                        p[1],
                        edgecolor="black",
                        facecolor=facecolor,
                        linewidth=marker_line,
                        s=marker_size + 20,
                        marker=legend_marker,
                        zorder=5,
                    )
                self.plot_spoofing_communication(ts, ax)
                self.plot_weighted_areas(ax)
                # legend and labels
                self.build_legend(ax)
                pf.finish_figure(file_path=f"{folder_path}/animation_{frame:04d}.png")
                plt.close()
                frame += 1

    def plot_spoofing_communication(self, ts, ax):

        spoofing_idx = np.where(
            [type(robot) == SpoofingRobot for robot in self.robot_swarm.swarm_robots]
        )[0]

        # communicated behavior of spoofing robot, if existent
        if not spoofing_idx.any():
            return []

        spoofing_idx = spoofing_idx.item()
        spoofing_robot = self.robot_swarm.swarm_robots[spoofing_idx]
        pos = spoofing_robot.get_state_history()[ts].position
        communicated_pos = spoofing_robot.get_communicated_state_history()[ts].position
        
        if (pos == communicated_pos).all():
            return []

        facecolor, legend_marker, marker_size, marker_line = self.get_markers(
            spoofing_robot, ts
        )
        facecolor, _ = self.build_colormap(facecolor, granularity=1, max_alpha=0.6)
        sp = ax.scatter(
            communicated_pos[0],
            communicated_pos[1],
            edgecolor="black",
            facecolor=facecolor,
            linewidth=marker_line,
            s=marker_size,
            marker=legend_marker,
            zorder=5,
        )
        return [sp]

    def plot_positions(self, ax, animate: bool):

        all_scatter_points = []

        for robot in self.robot_swarm.swarm_robots:
            if animate:
                state_history = robot.get_state_history()
            else:
                state_history = robot.get_communicated_state_history()

            scatter_points_per_robot = []

            for ts, state in enumerate(state_history):
                # colored markers for robots
                facecolor, legend_marker, marker_size, marker_line = self.get_markers(
                    robot, ts
                )

                # include intermediate positions for animation
                if animate and len(state.intermediate_positions) > 0:
                    positions = np.vstack(
                        (state.position, np.vstack(state.intermediate_positions))
                    )
                else:
                    positions = state.position[None, :]

                for pos in positions:
                    rp = ax.scatter(
                        pos[0],
                        pos[1],
                        edgecolor="black",
                        facecolor=facecolor,
                        linewidth=marker_line,
                        s=marker_size,
                        marker=legend_marker,
                        zorder=5,
                    )
                    scatter_points_per_robot.append(rp)

            all_scatter_points.append(scatter_points_per_robot)

        return np.vstack(all_scatter_points).T

    def plot_actions(self, ax: Axes) -> None:

        for robot in self.robot_swarm.swarm_robots:
            state_history = robot.get_communicated_state_history()

            for start_state, end_state in zip(state_history[:-1], state_history[1:]):
                # plot robot actions
                arrow = FancyArrowPatch(
                    (start_state.position[0], start_state.position[1]),
                    (end_state.position[0], end_state.position[1]),
                    color="black",
                    mutation_scale=6,
                    linewidth=0.3,
                    arrowstyle="-|>",
                    zorder=4,
                )
                _ = ax.add_patch(arrow)

            if type(robot) == SpoofingRobot:
                start_state = robot.get_state_history()[0]
                arrow = FancyArrowPatch(
                    (start_state.position[0], start_state.position[1]),
                    (robot.AOI[0], robot.AOI[1]),
                    color="black",
                    mutation_scale=6,
                    linewidth=0.3,
                    linestyle=(0, (10, 6)),
                    arrowstyle="-|>",
                    zorder=4,
                )
                _ = ax.add_patch(arrow)
        return

    def plot_action_samples(
        self, plot_samples, ax: Axes, deployment_area: PolygonWrapper
    ) -> List:
        if not plot_samples:
            return []
        if self.robot_swarm.anomaly_detector is None:
            print("please set an anomaly detector")
            return []
        else:
            all_sample_arrows = []
            positions = np.array(
                [
                    [s.position for s in robot.get_communicated_state_history()]
                    for robot in self.robot_swarm.swarm_robots
                ]
            ).transpose(1, 0, 2)

            for start_pos in positions[:-1]:
                sample_arrows = []
                n_action_samples = 500
                action_samples, _ = (
                    self.robot_swarm.anomaly_detector.sample_robot_actions(
                        positions=start_pos,
                        deployment_area=deployment_area,
                        n_samples=n_action_samples,
                    )
                )
                for r in range(len(self.robot_swarm.swarm_robots)):
                    for n in range(n_action_samples):
                        # if low_lp_mask[t, n]:
                        arrow = FancyArrowPatch(
                            (start_pos[r, 0], start_pos[r, 1]),
                            (
                                start_pos[r, 0] + action_samples[r, n, 0],
                                start_pos[r, 1] + action_samples[r, n, 1],
                            ),
                            color="#515151",
                            mutation_scale=6,
                            linewidth=0.2,
                            arrowstyle="->",
                            zorder=3,
                            alpha=0.05,
                        )
                        sa = ax.add_patch(arrow)
                        sample_arrows.append(sa)
                    all_sample_arrows.append(sample_arrows)
            return all_sample_arrows

    def plot_weighted_areas(self, ax):

        # plot contours
        x, y = np.mgrid[
            ax.get_xlim()[0] : ax.get_xlim()[1] : 0.1,
            ax.get_ylim()[0] : ax.get_ylim()[1] : 0.1,
        ]
        cont = []
        for robot in self.robot_swarm.swarm_robots:
            if hasattr(robot, "density_func"):
                zorder = 0
                levels = 15
                z = robot.density_func(np.dstack((x, y)))  # type: ignore
                rgb, rgb_cmap = self.build_colormap(
                    robot.color, granularity=levels, max_alpha=0.9
                )
                c = ax.contourf(
                    x, y, z, zorder=zorder, cmap=rgb_cmap, levels=levels, extend="min"
                )
                cont += c.collections
            elif hasattr(robot, "AOI"):
                rgb, rgb_cmap = self.build_colormap(
                    robot.color, granularity=1, max_alpha=0.9
                )
                ax.scatter(
                    robot.AOI[0],  # type: ignore
                    robot.AOI[1],  # type: ignore
                    color=rgb,
                    s=300,
                    zorder=1,
                )

        return cont

    def build_colormap(self, color, granularity, max_alpha=0.8):

        if granularity == 1:
            alphas = [max_alpha]
        else:
            alphas = np.zeros(shape=(granularity,))
            alphas = np.linspace(0, max_alpha, granularity)

        rgba = np.zeros(shape=(granularity, 4))
        rgba[:, :3] = to_rgb(color)
        rgba[:, -1] = alphas
        rgba_cmap = ListedColormap(rgba)

        rgb = np.ones(shape=(granularity, 3), dtype="float32")
        r, g, b, a = (
            rgba[:, 0],
            rgba[:, 1],
            rgba[:, 2],
            rgba[:, 3],
        )
        # a = np.asarray(a, dtype="float32")
        rgb[:, 0] = r * a + (1.0 - a)
        rgb[:, 1] = g * a + (1.0 - a)
        rgb[:, 2] = b * a + (1.0 - a)

        rgb_cmap = ListedColormap(rgb)
        rgb_cmap.set_under("white", alpha=0)

        return rgb, rgb_cmap

    def get_markers(self, robot: Robot, timestep: int):

        color = robot.color

        if self.robot_swarm.is_anomal(robot, timestep):
            marker = "s"
            size = 20
            lines = 1
        else:
            marker = "o"
            size = 15
            lines = 0.5

        return color, marker, size, lines

    def get_start_positions(self):
        all_start_positions = []
        for robot in self.robot_swarm.swarm_robots:
            # positions used for the Voronoi tesselation
            start_positions_per_robot = [
                state.position for state in robot.get_communicated_state_history()
            ]
            all_start_positions.append(start_positions_per_robot)
        return np.array(all_start_positions).transpose(1, 0, 2)

    def build_legend(self, ax):

        legend_markers = []
        legend_labels = []

        # coverage area
        legend_markers.append(
            Line2D(
                [0, 0],
                [0, 0],
                linewidth=0,
                marker="_",
                color="#7878ff",
                linestyle="solid",
            )
        )
        legend_labels.append("deployment area")

        legend_markers.append(
            Line2D(
                [0, 0],
                [0, 0],
                marker="s",
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="",
            )
        )
        legend_labels.append("label anomal")

        for robot in self.robot_swarm.swarm_robots:

            if (
                robot.label_text
                not in legend_labels
                # and robot.label != 0
            ):
                legend_labels.append(robot.label_text)
                legend_markers.append(
                    Line2D(
                        [0, 0],
                        [0, 0],
                        markeredgecolor="black",
                        markerfacecolor=robot.color,
                        marker="o",
                        linestyle="",
                        markeredgewidth=0.5,
                    )
                )
        legend = ax.legend(
            legend_markers,
            legend_labels,
            numpoints=1,
            # loc="upper left",
            # loc="lower right",
            loc="best",
            # bbox_to_anchor=(0, 0.88),
            # ncol=3,  # ,
        )
        legend.get_frame().set_linewidth(0.5)

        return legend

    # def show_time(self, ax, ts):

    #     ti = np.round(
    #         0.3 * ts,  # TODO
    #         decimals=2,
    #     )
    #     tx = ax.text(
    #         x=0.03,
    #         y=0.96,
    #         s=f"t = {ti}",
    #         ha="left",
    #         va="top",
    #         fontsize=12,
    #         zorder=6,
    #         transform=ax.transAxes,
    #         bbox=dict(facecolor="white", alpha=0.8, linewidth=0.5, pad=2),
    #     )
    #     return tx
