import numpy as np
import pytest
from shapely.geometry import Polygon
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from ..geometry import create_envelope
from ..geometry import inside_ellipse
from ..geometry import polygon_overlap
from ..geometry import probabilistic_ellipse


@pytest.mark.parametrize(
    "poly1,poly2,expected",
    [
        (
            np.array([[0, 4.0], [10, 4.0], [10, 8.2], [10, 8.2], [0, 8.2], [0, 4.0]]),
            np.array([[0, 4.0], [5, 4.0], [5, 8.2], [5, 8.2], [0, 8.2], [0, 4.0]]),
            0.5,
        ),
        (
            np.array([[0, 4.0], [10, 4.0], [10, 8.2], [10, 8.2], [0, 4.0]]),
            np.array([[12, 4.0], [15, 4.0], [15, 8.2], [15, 8.2], [12, 4.0]]),
            0.0,
        ),
    ],
)
def test_polygon_overlap(poly1, poly2, expected):
    poly1, poly2 = Polygon(poly1), Polygon(poly2)
    assert polygon_overlap(poly1, poly2) == expected
    assert polygon_overlap(poly1, poly2, threshold=0.6) == 0.0


def test_create_envelope():
    test_data = make_blobs(n_samples=1000, n_features=2, centers=1, center_box=(0, 5), random_state=42)[0]
    x, y = test_data[:, 0], test_data[:, 1]
    envelope = create_envelope(np.array(list(zip(x, y))))
    assert isinstance(envelope, Polygon)
    for idx, t in enumerate([x, y]):
        lower = np.quantile(t, 0.05)
        upper = np.quantile(t, 0.95)
        t_ = [i for i in t if lower < i < upper]
        for i in range(100):
            s = np.random.choice(t_, 1)[0]
            assert s >= np.min(envelope.exterior.xy[idx])
            assert s <= np.max(envelope.exterior.xy[idx])


@pytest.mark.parametrize("conf", [0.95, 0.8, 0.5])
def test_probablistic_ellipse(conf):
    test_data = make_blobs(n_samples=1000, n_features=2, centers=1, center_box=(1, 5), random_state=42)[0]
    model = GaussianMixture(random_state=42, n_components=1)
    model.fit(test_data)
    center = model.means_[0]
    width, height, angle = probabilistic_ellipse(model.covariances_[0], conf)
    mask = inside_ellipse(test_data, center=center, width=width, height=height, angle=angle)
    assert test_data[mask].shape[0] / test_data.shape[0] == pytest.approx(conf, 0.1)


@pytest.mark.parametrize(
    "test_data,expected_mask",
    [
        (np.array([[3, 4.5], [7.5, 9]]), [True, False]),
        (np.array([[3, 4.5], [0, 0]]), [True, False]),
        (np.array([[11, 5], [6.2, 4.3]]), [False, True]),
    ],
)
def test_inside_ellipse(test_data, expected_mask):
    center, width, height, angle = (5, 5), 10, 5, 15
    mask = inside_ellipse(data=test_data, center=center, width=width, height=height, angle=angle)
    assert isinstance(mask, list)
    assert np.array_equal(mask, expected_mask)
