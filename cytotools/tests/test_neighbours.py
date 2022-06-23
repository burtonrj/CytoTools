import pandas as pd
from sklearn.datasets import make_blobs
from ..neighbours import calculate_optimal_neighbours, knn
from sklearn.neighbors import KNeighborsClassifier
import pytest


@pytest.fixture()
def dummy_data():
    x, y = make_blobs(n_samples=5000, n_features=10, random_state=42, centers=3)
    x = pd.DataFrame(x, columns=list(range(10)))
    return x, y


def test_calculate_optimal_neighbours(dummy_data):
    x, y = dummy_data
    k, score = calculate_optimal_neighbours(x=x, y=y, scoring="accuracy")
    assert k == 5
    assert score > 0.7


def test_knn(dummy_data):
    x, y = dummy_data
    train_acc, test_acc, model = knn(data=x, labels=y, features=list(range(10)), n_neighbours=5, return_model=True)
    assert isinstance(model, KNeighborsClassifier)
    assert train_acc > 0.7
    assert test_acc > 0.7
