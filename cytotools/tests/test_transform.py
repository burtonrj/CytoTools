import pandas.testing as pd_testing
from ..read import read_from_disk, polars_to_pandas
from .. import transform
from . import assets
import pytest


@pytest.fixture()
def dummy_data():
    return polars_to_pandas(read_from_disk(f"{assets.__path__._path[0]}/test.fcs"))


@pytest.mark.parametrize(
    "transformer",
    [transform.LogicleTransformer, transform.AsinhTransformer, transform.HyperlogTransformer]
)
def test_transformers(dummy_data, transformer):
    transformer = transformer()
    transformed = transformer.scale(data=dummy_data, features=dummy_data.columns.tolist())
    assert ((transformed.mean() < 10) & (transformed.mean() > -1)).all()
    inverse = transformer.inverse_scale(data=transformed, features=transformed.columns.tolist())
    pd_testing.assert_frame_equal(inverse, dummy_data)


@pytest.mark.parametrize(
    "method,return_transformer,features,kwargs",
    [
        ("logicle", True, None, {}),
        ("logicle", True, None, {"w": 1.0}),
        ("logicle", False, ["FSC-A", "SSC-A"], {"m": 5.0}),
        ("asinh", True, None, {}),
        ("asinh", True, None, {"cofactor": 150}),
        ("asinh", False, ["FSC-A", "SSC-A"], {"cofactor": 100}),
    ]
)
def test_apply_transform(dummy_data, method, return_transformer, features, kwargs):
    features = dummy_data.columns.tolist() if features is None else features
    if return_transformer:
        data, transformer = transform.apply_transform(
            data=dummy_data,
            features=features,
            method=method,
            return_transformer=return_transformer,
            **kwargs
        )
        assert isinstance(transformer, transform.TRANSFORMERS.get(method))
    else:
        data = transform.apply_transform(
            data=dummy_data,
            features=features,
            method=method,
            **kwargs
        )
    assert ((data[features].mean() < 10) & (data[features].mean() > -1)).all()


def test_apply_transform_map(dummy_data):
    data = transform.apply_transform_map(
        data=dummy_data,
        feature_method={
            "7-AAD-A": "asinh",
            "PE-Cy7-A": "logicle"
        },
        **{
            "asinh": {"cofactor": 150},
        }
    )
    pd_testing.assert_series_equal(dummy_data["FSC-A"], data["FSC-A"])
    assert -1 < data["PE-Cy7-A"].mean() < 1.
    assert -1 < data["7-AAD-A"].mean() < 6.
