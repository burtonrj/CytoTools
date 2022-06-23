import pytest
from sklearn.datasets import load_iris
from .. import sampling


def test_uniform_sampling():
    data = load_iris(as_frame=True).data
    sample = sampling.uniform_downsampling(data=data, sample_size=0.5)
    assert sample.shape[0] == int(data.shape[0]/2)
    sample = sampling.uniform_downsampling(data=data, sample_size=100)
    assert sample.shape[0] == 100
    sample = sampling.uniform_downsampling(data=data, sample_size=int(1e10))
    assert sample.shape[0] == data.shape[0]


def test_faithful_downsampling():
    data = load_iris(as_frame=True).data
    sample = sampling.faithful_downsampling(data=data.values)
    assert sample.shape[0] < data.shape[0]


def test_density_dependent_downsampling():
    data = load_iris(as_frame=True).data
    sample = sampling.density_dependent_downsampling(
        data=data, sample_size=100
    )
    assert sample.shape[0] == 100


def test_upsample_density():
    data = load_iris(as_frame=True).data
    bigger_data = sampling.upsample_density(
        data=data, upsample_factor=2, tree_sample=0.7, outlier_dens=3
    )
    assert bigger_data.shape[0] >= data.shape[0] * 2


@pytest.mark.parametrize("method", ["uniform", "faithful", "density"])
def test_sample_dataframe(method):
    data = load_iris(as_frame=True).data
    sample = sampling.sample_dataframe(data=data, method=method, sample_size=100)
    assert sample.shape[0] < data.shape[0]
