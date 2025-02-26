from typing import Callable, List, Tuple

import numpy as np
import scipy.integrate as si
from scipy.spatial import Delaunay, Voronoi
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from shapely.geometry import Polygon
from src.deployment_area.polygon import PolygonWrapper


def compute_voronoi_cell(
    position: np.ndarray, robot_positions: np.ndarray, coverage_area: PolygonWrapper
) -> PolygonWrapper:
    """
    Compute the voronoi cell of a robot r within the coverage area.

        Parameters
        ----------
        position:
            Position of robot r.
        robot_positions:
            Positions of all swarm robots.
        coverage_area:
            Area that the robots are supposed to cover, described by a polygonal shape.

        Returns
        -------
        robot_region:
            Voronoi cell of the robot.
    """
    # intersect the voronoi cell with the polygonal area in order to define the region that the robot will cover
    voronoi = Voronoi(robot_positions)
    length_scale = 1e10
    poly_voronoi = PolygonWrapper(
        compute_voronoi_cell_vertices(voronoi, position, length_scale)
    )
    poly_voronoi = poly_voronoi.compute_convex_hull()
    robot_region = coverage_area.intersect(poly_voronoi)

    return robot_region


# see https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/spatial/_plotutils.py#L151-L264
def compute_voronoi_cell_vertices(
    voronoi: Voronoi, vpoint: np.ndarray, len_scale: float
) -> np.ndarray:
    """
    Compute the coordinates of all voronoi cell vertices of the cell containing vpoint.

        Parameters
        ----------
        voronoi:
            Voronoi decomposition.
        vpoint:
            Marks the voronoi cell whose vertices are being computed.
        len_scale:
            Scale the voronoi lines towards infinity.

        Returns
        -------
        region_vertices:
            Coordinates of the vertices of the voronoi cell.
    """
    # find ridge points and ridge vertices relevant for the region of vpoint
    # ridge_vertices[idx] corresponds to ridge_points[idx]
    vpoint_idx = np.where((voronoi.points == vpoint).all(axis=1))[0]
    v_idx = np.where(voronoi.ridge_points == vpoint_idx)[0]
    # ridge point pairs that contain vpoint (ridge points are the robots)
    rp = voronoi.ridge_points[v_idx]
    # ridge vertices enclosing the vpoint region (ridge vertices are the cell vertices)
    rv = np.array(voronoi.ridge_vertices)[v_idx]

    region_vertices = []

    # add finite vertices, compute vertices for infinite lines
    for pointidx, simplex in zip(rp, rv):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            region_vertices.append(voronoi.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = voronoi.points[pointidx[1]] - voronoi.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - voronoi.points.mean(axis=0), n)) * n
            if voronoi.furthest_site:
                direction = -direction
            far_point = voronoi.vertices[i] + direction * len_scale

            region_vertices.append([voronoi.vertices[i], far_point])

    # each vertex is contained twice: create array of unique vertices
    region_vertices = np.unique(np.array(region_vertices).reshape(-1, 2), axis=0)

    return region_vertices


def compute_weighted_center(
    polygon: PolygonWrapper, density_func: Callable
) -> np.ndarray:
    """
    Compute the center of a polygon shape weighted by a density function.

        Parameters
        ----------
        polygon
        density_func:
            Density function used to weight the polygon area.

        Returns
        -------
        weighted_center
    """
    tri_list = compute_delaunay(polygon)

    area_sum = 0
    x_sum = 0
    y_sum = 0

    gfunc = lambda b1: 0
    hfunc = lambda b1: 1 - b1

    for tri in tri_list:
        absdetjacob = compute_abs_det_jacob(tri)

        int_fun = lambda weight_cx, weight_cy: si.dblquad(
            func=compute_density,
            a=0,
            b=1,
            gfun=gfunc,
            hfun=hfunc,
            args=(tri, absdetjacob, density_func, weight_cx, weight_cy),
            epsabs=1e-1,
            epsrel=1e-1,
        )[0]

        weighted_area = int_fun(False, False)
        weighted_area_x = int_fun(True, False)
        weighted_area_y = int_fun(False, True)

        area_sum += weighted_area
        x_sum += weighted_area_x
        y_sum += weighted_area_y

    if area_sum == 0:
        weighted_center = polygon.compute_center()
    else:
        weighted_center = np.array([x_sum / area_sum, y_sum / area_sum])

    return weighted_center


def compute_delaunay(polygon: PolygonWrapper) -> List[np.ndarray]:
    """
    Compute the delaunay triangulation of a convex polygon.

        Parameters
        ----------
        polygon

        Returns
        -------
        tri_list:
            List of delaunay triangles.
    """
    delaunay = Delaunay(polygon.vertices)
    tri_list = []

    # check if the triangle lies within the polygon since delaunay
    # might use a convex hull. Does only work for convex polygons.
    for simplex in delaunay.simplices:
        tri = delaunay.points[simplex]
        assert polygon.polygon.contains(Polygon(tri))
        tri_list.append(tri)

    return tri_list


def compute_density(
    b2: float,
    b1: float,
    tri: np.ndarray,
    absdetjacob: float,
    density_func: Callable,
    weight_cx: bool,
    weight_cy: bool,
) -> float:
    """
    Compute the density weight of a point specified by the barycentric coordinates b1 and b2.

        Parameters
        ----------
        b1, b2:
            Barycentric coordinates.
        tri:
            Delaunay triangle.
        density_func:
            Density function used to weight the point.
        weight_cx, weight_cy:
            Specifies the dimension to be weighted.

        Returns
        -------
        density
    """
    cx, cy = transform_to_baryzentric(tri, b1, b2)
    dv = density_func(np.array([cx, cy]))

    if weight_cx:
        dcx = dv * cx
        return dcx * absdetjacob
    elif weight_cy:
        dcy = dv * cy
        return dcy * absdetjacob
    else:
        return dv * absdetjacob


def l2_norm(input_array: np.ndarray) -> np.ndarray:
    """
    Compute the l2 norm over the last dimension of an array.

        Parameters
        ----------
        input_array: (*, 2)

        Returns
        -------
        l2_norm
    """
    l2_norm = np.sqrt(np.sum(input_array**2, axis=-1))
    return l2_norm


def transform_to_baryzentric(
    tri: np.ndarray, b1: float, b2: float
) -> Tuple[float, float]:
    """
    Compute a coordinate using the barycentric coordinates.

        Parameters
        ----------
        tri:
            Delaunay triangle.
        b1, b2:
            Barycentric coordinates.

        Returns
        -------
        cx, cy:
            x and y coordinate.
    """
    x1 = tri[0, 0]
    y1 = tri[0, 1]
    x2 = tri[1, 0]
    y2 = tri[1, 1]
    x3 = tri[2, 0]
    y3 = tri[2, 1]

    cx = b1 * x1 + b2 * x2 + (1 - b2 - b1) * x3
    cy = b1 * y1 + b2 * y2 + (1 - b2 - b1) * y3

    return cx, cy


def compute_abs_det_jacob(tri: np.ndarray) -> float:
    """
    Compute the absolute value of the jacobian determinant of a delaunay triangle.

        Parameters
        ----------
        tri:
            Delaunay triangle.

        Returns
        -------
        absdetjacob
    """
    x1 = tri[0, 0]
    y1 = tri[0, 1]
    x2 = tri[1, 0]
    y2 = tri[1, 1]
    x3 = tri[2, 0]
    y3 = tri[2, 1]

    absdetjacob = np.abs((x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3))
    return absdetjacob


def compute_voronoi_cell_loss(
    voronoi_cell: PolygonWrapper, robot_pos: np.ndarray
) -> float:
    """
    Compute the summed distance of the points within a voronoi cell to a point p within the cell.

        Parameters
        ----------
        voronoi_cell:
            Voronoi cell.
        robot_pos:
            Specifies the point p the distance is computed to.

        Returns
        -------
        cell_loss:
            Integral over distances.
    """
    tri_list = compute_delaunay(voronoi_cell)
    cell_loss = 0

    gfunc = lambda b1: 0
    hfunc = lambda b1: 1 - b1

    for tri in tri_list:
        # compute the loss for each delaunay triangle
        delaunay_loss = si.dblquad(
            func=loss_fn, a=0, b=1, gfun=gfunc, hfun=hfunc, args=(tri, robot_pos)
        )[0]
        cell_loss += delaunay_loss

    return cell_loss


def loss_fn(b2: float, b1: float, tri: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Compute the squared distance of a position pos to a point specified by the barycentric coordinates b1 and b2
    within a delaunay triangle.

        Parameters
        ----------
        b1, b2:
            Barycentric coordinates.
        tri:
            Delaunay triangle.
        pos:
            Specifies the position pos the distance is computed to.

        Returns
        -------
        dist
    """
    absdetjacob = compute_abs_det_jacob(tri)
    cx, cy = transform_to_baryzentric(tri, b1, b2)
    l2 = l2_norm(np.array([cx, cy]) - pos)
    dist = np.square(l2) * absdetjacob
    return dist
