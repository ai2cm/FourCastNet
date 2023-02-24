import logging
import os
from typing import MutableMapping
import numpy as np
from torch.utils.data import Dataset
import netCDF4
from utils.constants import CHANNEL_NAMES

# conversion from 'standard' names defined in utils/constants.py to those
# in FV3GFS output netCDFs
FV3GFS_NAMES = {
    'u10': 'UGRD10m',
    'v10': 'VGRD10m',
    't2m': 'TMP2m',
    'sp': 'PRESsfc',
    'msl': 'PRMSL',
    't850': 'TMP850',
    'u1000': 'UGRD1000',
    'v1000': 'VGRD1000',
    'z1000': 'h1000',
    'u850': 'UGRD850',
    'v850': 'VGRD850',
    'z850': 'h850',
    'u500': 'UGRD500',
    'v500': 'VGRD500',
    'z500': 'h500',
    't500': 'TMP500',
    'z50': 'h50',
    'rh500': 'RH500',
    'rh850': 'RH850',
    'tcwv': 'TCWV',
}

class FV3GFSDataset(Dataset):
    def __init__(self, params: MutableMapping, path: str, train: bool):
        self.params = params
        self._screen_for_not_implemented_features()
        self.path = path
        self.full_path = os.path.join(path, '*.nc')
        self.train = train
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.roll = params.roll
        self.two_step_training = params.two_step_training
        self.orography = params.orography
        self.precip = True if "precip" in params else False
        self.add_noise = params.add_noise if train else False
        self._get_files_stats()

    def _screen_for_not_implemented_features(self):
        if self.params.n_history != 0:
            raise NotImplementedError(
                'non-zero n_history is not implemented for FV3GFSDataset'
                )
        if self.params.crop_size_x is not None or self.params.crop_size_y is not None:
            raise NotImplementedError(
                'non-null crop_size_x or crop_size_y is not implemented for FV3GFSDataset'
                )
        if self.params.roll:
            raise NotImplementedError('roll=True not implemented for FV3GFSDataset')
        if self.params.two_step_training:
            raise NotImplementedError('two_step_training not implemented for FV3GFSDataset')
        if self.params.orography:
            raise NotImplementedError('training w/ orography not implemented for FV3GFSDataset')
        if "precip" in self.params:
            raise NotImplementedError('precip training not implemented for FV3GFSDataset')

    def _get_files_stats(self):
        logging.info(f"Opening data at {self.full_path}")
        f = netCDF4.MFDataset(self.full_path)
        self.n_samples_total = len(f.variables['time'][:])
        self.img_shape_x = len(f.variables['grid_xt'][:])
        self.img_shape_y = len(f.variables['grid_yt'][:])
        logging.info(f"Found {self.n_samples_total} samples.")
        logging.info(f"Image shape is {self.img_shape_x} x {self.img_shape_y}.")
        logging.info(f"Following variables are available: {list(f.variables)}.")

    def __len__(self):
        return self.n_samples_total
    
    def __getitem__(self, idx):
        pass