from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from itm_pythonfig.pythonfig import PythonFig
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from shapely.geometry import Point, Polygon


class PolygonWrapper:
    def __init__(self, vertices: np.ndarray, **kwargs) -> None:
        """
        Create a complex polygon.

        Parameters
        ----------
        vertices: np.ndarray
            (x, y) coordinates of the polygon vertices.
        """

        if np.array_equal(vertices[0], vertices[-1]):
            self.vertices = vertices
        else:
            self.vertices = np.vstack((vertices, vertices[0]))
        self.polygon = Polygon(vertices)
        self.n_vertices = self.vertices.shape[0] - 1
        # self.plot_polygon()

    def compute_polygon_area(self) -> float:
        return self.polygon.area

    def get_vertices(self) -> np.ndarray:
        return self.vertices[:-1]

    def compute_coords(self, polygon):
        x, y = polygon.exterior.coords.xy
        vertices = np.vstack((x, y)).T

        return vertices

    def contains_point(self, point):
        return self.polygon.contains(Point(point[0], point[1]))

    def compute_convex_hull(self):
        return PolygonWrapper(self.compute_coords(self.polygon.convex_hull))

    def intersect(self, polygon2):
        poly_intersection = self.polygon.intersection(polygon2.polygon)
        return PolygonWrapper(self.compute_coords(poly_intersection))

    def compute_center(self) -> np.ndarray:
        return np.array(self.polygon.centroid.coords.xy).flatten()

    def compute_extension(self):
        x_ext = max(self.vertices[:, 0]) - min(self.vertices[:, 0])
        y_ext = max(self.vertices[:, 1]) - min(self.vertices[:, 1])
        ext_1d = np.max([x_ext, y_ext])
        ext_2d = np.sqrt((x_ext) ** 2 + (y_ext) ** 2)

        border = ext_1d * 0.02
        ratio = (x_ext + 2 * border) / (y_ext + 2 * border)

        return ext_1d, ext_2d, border, ratio, x_ext, y_ext

    def compute_polygon_points(self, step_size=0.1) -> Tuple[np.ndarray, np.ndarray]:
        # make a polygon
        polygon_path = Path(self.vertices)

        # make a canvas with coordinates
        gx, gy = np.meshgrid(
            np.arange(
                np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0]), step_size
            ),
            np.arange(
                np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1]), step_size
            ),
        )
        grid_points = np.vstack((gx.flatten(), gy.flatten())).T

        grid = polygon_path.contains_points(grid_points)
        mask = grid.reshape(gx.shape[0], gx.shape[1])

        polygon_points = np.hstack((gx[mask].reshape(-1, 1), gy[mask].reshape(-1, 1)))
        outside_points = np.hstack(
            (gx[1 - mask].reshape(-1, 1), gy[1 - mask].reshape(-1, 1))
        )
        return polygon_points, outside_points

    def plot_polygon(
        self,
        fig=None,
        ax=None,
        voronoi=None,
        coords=None,
        label="deployment area",
        plot_square=False,
        figsize=11,
        **kwargs,
    ) -> list:
        img_components = []

        # set the figsize as well as the range of the x- and y-axis
        lx = np.min(self.vertices[:, 0])
        ly = np.min(self.vertices[:, 1])
        ext, _, border, ratio, x_ext, y_ext = self.compute_extension()

        if fig is None:
            pf = PythonFig()
            fig = pf.start_figure("PAMM", figsize, figsize)

        if ax is None:
            ax = fig.gca()

        ax.set_aspect(1)  # x and y axis have the same scale
        ax.set_xlim(lx - border, lx + ext + border)
        ax.set_ylim(ly - border, ly + ext + border)

        if plot_square:
            ax.set_xlim(lx - border, lx + ext + border)
            ax.set_ylim(ly - border, ly + ext + border)
        else:
            # print("adapt figsize")
            # size = (fig.get_figwidth(), fig.get_figheight() / ratio)
            # fig.set_figwidth(size[0])
            # fig.set_figheight(size[1])
            ax.set_xlim(lx - border, lx + x_ext + border)
            ax.set_ylim(ly - border, ly + y_ext + border)
        # plot the voronoi diagram
        if voronoi:
            kwargs["show_points"] = False
            kwargs["show_vertices"] = False
            _, vlines = voronoi_plot_2d(voronoi, ax, **kwargs)
            img_components += [vlines]

        # plot the polygon edges
        poly = ax.plot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            label=label,
            color="#7878ff",
            zorder=2,
        )
        img_components += poly

        # plot the robot positions
        if coords is not None:
            c = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                edgecolor="black",
                facecolor="white",
                linewidth=0.5,
                s=15,
                marker="o",
                zorder=5,
            )
            img_components += [c]

        return img_components


# https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/spatial/_plotutils.py#L151-L264
def voronoi_plot_2d(vor, ax, **kw):
    """
    Plot the given Voronoi diagram in 2-D
    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points : bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points
    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot
    See Also
    --------
    Voronoi
    Notes
    -----
    Requires Matplotlib.
    Examples
    --------
    Set of point:
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> points = rng.random((10,2))
    Voronoi diagram of the points:
    >>> from scipy.spatial import Voronoi, voronoi_plot_2d
    >>> vor = Voronoi(points)
    using `voronoi_plot_2d` for visualisation:
    >>> fig = voronoi_plot_2d(vor)
    using `voronoi_plot_2d` for visualisation with enhancements:
    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
    ...                 line_width=2, line_alpha=0.6, point_size=2)
    >>> plt.show()
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get("show_points", True):
        point_size = kw.get("point_size", None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], ".", markersize=point_size)
    if kw.get("show_vertices", True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], "o")

    line_colors = kw.get("line_colors", "k")
    line_width = kw.get("linewidth", 1.0)
    line_alpha = kw.get("line_alpha", 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if vor.furthest_site:
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max() * 5

            infinite_segments.append([vor.vertices[i], far_point])

    lines = LineCollection(
        finite_segments + infinite_segments,
        colors=line_colors,
        lw=line_width,
        alpha=line_alpha,
        # linestyle="dashed",
        linewidth=0.5,
        # label="Voronoi cells",
        zorder=1,
    )

    ax.add_collection(lines)

    return ax.figure, lines
