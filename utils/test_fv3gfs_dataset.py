import copy
import os
import numpy as np
from .data_loader_fv3gfs import FV3GFSDataset
import pytest

TEST_PATH = "/traindata"
TEST_STATS_PATH = "/statsdata"
TEST_PARAMS = {
    "n_history": 0,
    "crop_size_x": None,
    "crop_size_y": None,
    "roll": False,
    "in_channels": [0, 1, 4, 19],
    "out_channels": [0, 1, 5, 19],
    "two_step_training": False,
    "orography": False,
    "add_noise": False,
    "dt": 1,
    "n_history": 0,
    "normalization": "zscore",
    "add_grid": False,
    "global_means_path": os.path.join(TEST_STATS_PATH, "fv3gfs-mean.nc"),
    "global_stds_path": os.path.join(TEST_STATS_PATH, "fv3gfs-stddev.nc"),
    "time_means_path": os.path.join(TEST_STATS_PATH, "fv3gfs-time-mean.nc"),
}
TEST_PARAMS_SPECIFY_BY_NAME = copy.copy(TEST_PARAMS)
TEST_PARAMS_SPECIFY_BY_NAME.pop("in_channels")
TEST_PARAMS_SPECIFY_BY_NAME.pop("out_channels")
TEST_PARAMS_SPECIFY_BY_NAME["in_names"] = ["UGRD10m", "VGRD10m", "PRMSL", "TCWV"]
TEST_PARAMS_SPECIFY_BY_NAME["out_names"] = ["UGRD10m", "VGRD10m", "TMP850", "TCWV"]


class DotDict:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, key):
        return self.items[key]

    def __getattr__(self, attr):
        return self.items[attr]

    def __contains__(self, key):
        return key in self.items


@pytest.mark.parametrize("params", [TEST_PARAMS, TEST_PARAMS_SPECIFY_BY_NAME])
def test_FV3GFSDataset_init(params):
    dataset = FV3GFSDataset(DotDict(params), TEST_PATH, True)
    assert dataset.in_names == ["UGRD10m", "VGRD10m", "PRMSL", "TCWV"]
    assert dataset.out_names == ["UGRD10m", "VGRD10m", "TMP850", "TCWV"]


def test_FV3GFSDataset_len():
    dataset = FV3GFSDataset(DotDict(TEST_PARAMS), TEST_PATH, True)
    assert len(dataset) == 235


def test_FV3GFSDataset_getitem():
    dataset = FV3GFSDataset(DotDict(TEST_PARAMS), TEST_PATH, True)
    output = dataset[150]
    assert len(output) == 2
    assert output[0].shape == (len(TEST_PARAMS["in_channels"]), 180, 360)
    assert output[1].shape == (len(TEST_PARAMS["out_channels"]), 180, 360)
