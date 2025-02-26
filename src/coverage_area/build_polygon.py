import numpy as np
from src.deployment_area.polygon import PolygonWrapper


def build_random_polygon(
    n_robots: int,
    min_area_per_robot: float = 3,
    max_area_per_robot: float = 25,
    n_min_vertices: int = 3,
    n_max_vertices: int = 8,
    max_border_ratio: float = 5,
    boundary: PolygonWrapper | None = None,
    **kwargs
) -> PolygonWrapper:
    """
    Build a random convex polygon with the specified area and number of nodes. The maximum extension of the area in x- and y-direction is set to 2 * sqrt(max_area).

        Parameters
        ----------
        min_area:
            Minimum area size of the polygon.
        max_area:
            Maximum area size of the polygon.
        n_min_nodes:
            Minimum number of polygon nodes.
            Default 3
        n_max_nodes:
            Maximum number of polygon nodes.
            Default 5

        Returns
        -------
        poly:
            Convex polygon.
    """
    n_nodes = np.random.randint(low=n_min_vertices, high=n_max_vertices + 1)
    correct_size = False
    correct_n_nodes = False
    correct_border_ratio = False
    correct_boundary = True if boundary is None else False

    min_area = min_area_per_robot * n_robots
    max_area = max_area_per_robot * n_robots
    max_ext = 2 * np.sqrt(max_area)
    poly = boundary

    while not (
        correct_size and correct_n_nodes and correct_border_ratio and correct_boundary
    ):
        nodes = np.random.uniform(low=0, high=max_ext, size=(n_nodes, 2))
        poly = PolygonWrapper(nodes)

        # compute convex polygon from nodes
        # this is necessary for the delaunay triangulation to work
        poly = PolygonWrapper(poly.compute_convex_hull().vertices)

        # the polygon should not contain any intersections
        # not relevant for convex polygons
        # while not poly.polygon.is_simple:
        #     poly = PolygonWrapper(np.random.permutation(nodes))

        # constrain the polygon to have a minimal size
        area = poly.compute_polygon_area()
        correct_size = area > min_area and area < max_area
        correct_n_nodes = n_nodes == poly.n_vertices
        if boundary is not None:
            correct_boundary = np.all(
                [boundary.contains_point(v) for v in poly.vertices]
            )

        *_, ratio, _, _ = poly.compute_extension()
        correct_border_ratio = (
            ratio > (1 / max_border_ratio) and ratio < max_border_ratio
        )
    assert poly is not None
    return poly
