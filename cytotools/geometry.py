#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
For the purpose of cytometry analysis we often think of a population
of cells as having a particular phenotype that can be identified by
sub-setting cells in one or two dimensional space. This results in
geometric objects that define a population. This module houses the
functionality around those geometric objects.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import alphashape
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from numba import jit
from scipy import linalg
from scipy import stats
from shapely.geometry import Point
from shapely.geometry import Polygon


class GeometryError(Exception):
    pass


def point_in_poly(coords: Tuple[float], poly: Polygon) -> bool:
    return poly.contains(Point(coords))


def inside_polygon(data: pd.DataFrame, x: str, y: str, poly: Polygon) -> pd.DataFrame:
    """
    Return rows in dataframe who's values for x and y are contained in some polygon coordinate shape

    Parameters
    ----------
    data: polars.DataFrame or Pandas.DataFrame
        Data to query
    x: str
        name of x-axis plane
    y: str
        name of y-axis plane
    poly: shapely.geometry.Polygon
        Polygon object to search

    Returns
    --------
    Pandas.DataFrame
        Masked DataFrame containing only those rows that fall within the Polygon
    """
    with Pool(cpu_count()) as pool:
        mask = pool.map(partial(point_in_poly, poly=poly), data[[x, y]].values)
    return data.iloc[mask, :]


def polygon_overlap(poly1: Polygon, poly2: Polygon, threshold: float = 0.0) -> float:
    """
    Compare the area of two polygons and give the fraction overlap.
    If fraction overlap does not exceed given threshold or the polygon's do not overlap,
    return 0.0

    Parameters
    ----------
    poly1: Polygon
    poly2: Polygon
    threshold: float (default = 0.0)

    Returns
    -------
    float
    """
    if poly1.intersects(poly2):
        overlap = float(poly1.intersection(poly2).area / poly1.area)
        if overlap >= threshold:
            return overlap
    return 0.0


@jit(nopython=True)
def inside_ellipse(
    data: np.array,
    center: Tuple[float, float],
    width: Union[int, float],
    height: Union[int, float],
    angle: Union[int, float],
) -> List[bool]:
    """
    Return mask of two-dimensional matrix specifying if a data point (row) falls
    within an ellipse

    Parameters
    -----------
    data: numpy.ndarray
        two-dimensional matrix (x,y)
    center: tuple
        x,y coordinate corresponding to center of ellipse
    width: int or float
        semi-major axis of eplipse
    height: int or float
        semi-minor axis of elipse
    angle: int or float
        angle of ellipse

    Returns
    --------
    List[bool]
    """
    cos_angle = np.cos(np.radians(180.0 - angle))
    sin_angle = np.sin(np.radians(180.0 - angle))

    x = data[:, 0]
    y = data[:, 1]

    xc = x - center[0]
    yc = y - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2.0) ** 2) + (yct ** 2 / (height / 2.0) ** 2)

    in_ellipse = []

    for r in rad_cc:
        if r <= 1.0:
            # point in ellipse
            in_ellipse.append(True)
        else:
            # point not in ellipse
            in_ellipse.append(False)
    return in_ellipse


def probabilistic_ellipse(covariances: np.ndarray, conf: float) -> Tuple[float, float, float]:
    """
    Given the covariance matrix of a mixture component, calculate a elliptical shape that
    represents a probabilistic confidence interval.

    Parameters
    ----------
    covariances: Numpy.Array
        Covariance matrix
    conf: float
        The confidence interval (e.g. 0.95 would give the region of 95% confidence)

    Returns
    -------
    Tuple[float, float, float]
        Width, Height and Angle of ellipse
    """
    eigen_val, eigen_vec = linalg.eigh(covariances)
    chi2 = stats.chi2.ppf(conf, 2)
    eigen_val = 2.0 * np.sqrt(eigen_val) * np.sqrt(chi2)
    u = eigen_vec[0] / linalg.norm(eigen_vec[0])
    angle = 180.0 * np.arctan(u[1] / u[0]) / np.pi
    return eigen_val[0], eigen_val[1], (180.0 + angle)


def create_envelope(xy: np.ndarray, alpha: Optional[float] = 0.0) -> Polygon:
    """
    Given the x and y coordinates of a cloud of data points generate an envelope (alpha shape)
    that encapsulates these data points.

    Parameters
    ----------
    xy: Numpy.Array
        X and Y coordinates of a cloud of data points
    alpha: float, optional (default = 0.0)
        By default alpha is 0, generating a convex hull (can be thought of as if wrapping an elastic band
        around the data points). Increase alpha to create a concave envelope. Warning, as alpha increases,
        more data points will fall outside the range of the envelope.


    Returns
    -------
    shapely.geometry.Polygon

    Raises
    ------
    GeometryError
        Failed to generate alpha shape; likely due to insufficient data or alpha being too large.
    """
    try:
        poly = alphashape.alphashape(points=xy, alpha=alpha)
        assert isinstance(poly, Polygon)
        return poly
    except AssertionError:
        raise GeometryError(
            "Failed to generate alpha shape. Check for insufficient data or whether alpha is too large."
        )


def ellipse_to_polygon(
    centroid: Tuple[float, float],
    width: float,
    height: float,
    angle: float,
    ellipse: Optional[Ellipse] = None,
) -> Polygon:
    """
    Convert an ellipse to a shapely Polygon object.

    Parameters
    ----------
    centroid: Tuple[float, float]
    width: float
    height: float
    angle: float
    ellipse: Ellipse, optional

    Returns
    -------
    shapely.geometry.Polygon
    """
    ellipse = ellipse or Ellipse(centroid, width, height, angle)
    vertices = ellipse.get_verts()
    return Polygon(vertices)
