from ..read import read_from_disk, polars_to_pandas
from .. import dimension_reduction
from . import assets
import pytest

from umap import UMAP


@pytest.mark.parametrize(
    "method", ["UMAP", "PCA", "TSNE", "PHATE", "KernelPCA", "MDS", "Isomap", UMAP(n_components=2)]
)
def test_dimension_reduction(method):
    data = polars_to_pandas(read_from_disk(f"{assets.__path__._path[0]}/levine32.csv")).sample(n=1000)
    reducer = dimension_reduction.DimensionReduction(method=method, n_components=2)
    if not isinstance(method, str):
        method = type(method).__name__
    reducer.fit(data=data, features=data.columns.tolist())
    data_with_embeddings = reducer.transform(data=data, features=data.columns.tolist())
    assert all([x in data_with_embeddings.columns for x in [f"{method}1", f"{method}2"]])
    data_with_embeddings = reducer.fit_transform(data=data, features=data.columns.tolist())
    assert all([x in data_with_embeddings.columns for x in [f"{method}1", f"{method}2"]])


def test_dimension_reduction_with_sampling():
    data = polars_to_pandas(read_from_disk(f"{assets.__path__._path[0]}/levine32.csv")).sample(n=10000)
    data_with_embeddings, reducer = dimension_reduction.dimension_reduction_with_sampling(
        data=data,
        features=data.columns.tolist(),
        method="UMAP",
        sampling_size=1000
    )
    assert data_with_embeddings.shape[0] == 10000
    assert set(data.index.values) == set(data_with_embeddings.index.values)
    assert all([x in data_with_embeddings.columns for x in ["UMAP1", "UMAP2"]])
