from numpy import testing as np_testing
import numpy as np

from .. import fda_norm
from sklearn.datasets import make_blobs
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
import pytest


@pytest.fixture()
def create_multimodal_data():
    x1, _ = make_blobs(n_features=1, random_state=42, centers=2)
    x2, _ = make_blobs(n_features=1, random_state=43, centers=2)
    return x1, x2


@pytest.mark.parametrize(
    "threshold,n",
    [
        (0.1, 5),
        (0.15, 4),
        (0.2, 3)
    ]
)
def test_merge_peaks(threshold, n):
    peaks = [0.5, 0.51, 0.95, 1.1, 2.4, 2.6]
    fda_norm.merge_peaks(peaks, threshold)


def test_landmark_reg(create_multimodal_data):
    x1, x2 = create_multimodal_data
    landmark_reg = fda_norm.LandmarkRegistration(bw="silverman")
    landmark_reg.fit([x1, x2])
    np_testing.assert_array_almost_equal(
        landmark_reg.landmarks, np.array([[-2.797, 9.02], [-7.565, 2.179]]), decimal=3
    )
    assert isinstance(landmark_reg.original_functions, fda_norm.FDataGrid)
    assert isinstance(landmark_reg.warping_functions, fda_norm.FDataGrid)
    landmark_reg.plot_warping()
    plt.show()
    aligned_data = landmark_reg.transform(x1)
    assert isinstance(aligned_data, np.ndarray)
    x, y = FFTKDE(bw="silverman").fit(aligned_data).evaluate()
    peaks = fda_norm.peaks(y, x)
    assert len(peaks) == 2
    assert -7.5 < peaks[0] < -3
    assert 2.2 < peaks[1] < 9.
