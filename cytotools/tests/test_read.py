import os
import shutil
from pathlib import Path

import flowio
import polars as pl
import pytest

from .. import read
from . import assets


@pytest.fixture()
def make_examples():
    os.mkdir(f"{os.getcwd()}/test_filter")
    os.mkdir(f"{os.getcwd()}/test_filter/ignore")
    Path(f"{os.getcwd()}/test_filter/primary.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD40.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD45.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD80.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD85.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/spillover.csv").touch()
    for i in range(5):
        Path(f"{os.getcwd()}/test_filter/{i + 1}_Compensation.fcs").touch()
    for i in range(5):
        Path(f"{os.getcwd()}/test_filter/ignore/{i + 1}.fcs").touch()
    yield
    shutil.rmtree(f"{os.getcwd()}/test_filter")


def test_filter_fcs_files(make_examples):
    assert (
        len(read.filter_fcs_files(
            f"{os.getcwd()}/test_filter", exclude_files="Compensation", exclude_dir="ignore"
        )) == 5
    )
    assert (
        len(read.filter_fcs_files(f"{os.getcwd()}/test_filter", exclude_dir="ignore")) == 10
    )


def test_parse_directory_for_cytometry_files(make_examples):
    tree = read.parse_directory_for_cytometry_files(
        f"{os.getcwd()}/test_filter",
        control_names=["CD45", "CD80", "CD40", "CD85"],
        control_id="FMO",
        exclude_files="Compensation",
        exclude_dir="ignore",
        compensation_file="spillover.csv"
    )
    assert tree.get("primary") == f"{os.getcwd()}/test_filter/primary.fcs"
    assert all([x in tree.keys() for x in ["CD45", "CD80", "CD40", "CD85"]])
    assert tree.get("compensation_file") == f"{os.getcwd()}/test_filter/spillover.csv"


def test_fcs_mappings():
    mappings = read.fcs_mappings(f"{assets.__path__._path[0]}/test.fcs")
    assert mappings["10"]["PnN"] == "Alexa Fluor 405-A"
    assert mappings["10"]["PnS"] == "CCR7"


def test_get_channel_mappings():
    mappings = read.get_channel_mappings(read.fcs_mappings(f"{assets.__path__._path[0]}/test.fcs"))
    assert len(mappings) == 19
    assert mappings[0]["channel"] == "FSC-A"
    assert mappings[0]["marker"] == ""

    assert mappings[10]["channel"] == "AmCyan-A"
    assert mappings[10]["marker"] == "L/D"


def test_match_file_ext():
    assert read.match_file_ext(f"{assets.__path__._path[0]}/test.fcs", ".fcs")
    assert read.match_file_ext(f"{assets.__path__._path[0]}/test.FCS", ".fcs")
    assert not read.match_file_ext(f"{assets.__path__._path[0]}/test.fcs", ".csv")


def test_load_compensation_matrix():
    comp_matrix = read.load_compensation_matrix(flowio.FlowData(f"{assets.__path__._path[0]}/test.fcs"))
    assert isinstance(comp_matrix, pl.DataFrame)
    assert comp_matrix.shape == (12, 12)


def test_read_from_disk():
    data = read.read_from_disk(f"{assets.__path__._path[0]}/test.fcs")
    assert isinstance(data, pl.DataFrame)
    data = read.read_from_disk(f"{assets.__path__._path[0]}/levine32.csv")
    assert isinstance(data, pl.DataFrame)
