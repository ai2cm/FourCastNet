#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#Gideon Dresdner <gideond@allenai.org>

# TODO(gideond) imports

import os
import sys
import time
from train import Trainer
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import (
    weighted_rmse_torch_channels,
    weighted_acc_torch_channels,
    unweighted_acc_torch_channels,
    weighted_acc_masked_torch_channels,
    weighted_global_mean_channels,
    weighted_global_mean_gradient_magnitude_channels,
)
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from utils.constants import CHANNEL_NAMES
from networks.afnonet import AFNONet
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime

import tempfile

def _get_test_yaml_file(train_data_path, 
                        valid_data_path, 
                        inf_data_path, 
                        results_dir,
                        time_means_path,
                        global_means_path,
                        global_stds_path,
                        config_name="unit_test"):

    string = f"""
     {config_name}: &{config_name}
       loss: 'l2'
       lr: 5E-4
       scheduler: 'CosineAnnealingLR'
       num_data_workers: 4
       dt: 1 # how many timesteps ahead the model will predict
       n_history: 0 #how many previous timesteps to consider
       prediction_type: 'iterative'
       prediction_length: 41 #applicable only if prediction_type == 'iterative'
       n_initial_conditions: 1 #applicable only if prediction_type == 'iterative'
       ics_type: "default"
       save_raw_forecasts: !!bool True
       save_channel: !!bool False
       masked_acc: !!bool False
       maskpath: None
       perturb: !!bool False
       add_grid: !!bool False
       N_grid_channels: 0
       gridtype: 'sinusoidal' #options 'sinusoidal' or 'linear'
       roll: !!bool False
       max_epochs: 50
       batch_size: 2

       #afno hyperparams
       num_blocks: 8
       nettype: 'afno'
       patch_size: 8
       width: 56
       modes: 32
       #options default, residual
       target: 'default'
       in_channels: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
       out_channels: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] #must be same as in_channels if prediction_type == 'iterative'
       normalization: 'zscore' #options zscore (minmax not supported)
       train_data_path: '{train_data_path}'
       valid_data_path: '{valid_data_path}'
       inf_data_path: '{inf_data_path}'
       exp_dir: '{results_dir}'
       time_means_path:   '{time_means_path}'
       global_means_path: '{global_means_path}'
       global_stds_path:  '{global_stds_path}'

       orography: !!bool False
       orography_path: None

       log_to_screen: !!bool True
       log_to_wandb: !!bool True
       save_checkpoint: !!bool True

       enable_nhwc: !!bool False
       optimizer_type: 'FusedAdam'
       crop_size_x: None
       crop_size_y: None

       two_step_training: !!bool False
       plot_animations: !!bool False

       add_noise: !!bool False
       noise_std: 0

    """

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(string)
        return f.name

def _save_to_tmpfile(data, dir, filetype='h5'):
    suffices = {'h5': '.h5', 'npy': '.npy'}
    if filetype not in suffices:
        raise ValueError(f'Unknown save format {filetype}')

    with tempfile.NamedTemporaryFile(dir=dir, mode='w', delete=False, suffix=suffices[filetype]) as f:
        if filetype == 'h5':
            with h5py.File(f.name, 'w') as hf:
                hf.create_dataset('fields', data=data)
        elif filetype == 'npy':
            np.save(f.name, data)  # TODO(gideond) allow_pickle?
        else:
            raise ValueError(f'Unknown save format {filetype}')
        return f.name

def test_train_runs_era5():
    """Make sure that the training runs without errors."""

    # TODO(gideond) parameterize
    seed = 0
    np.random.seed(seed)
    num_time_steps, num_channels, height, width = 8, 20, 720 + 1, 1440

    with tempfile.TemporaryDirectory() as train_dir, \
         tempfile.TemporaryDirectory() as valid_dir, \
         tempfile.TemporaryDirectory() as stats_dir, \
         tempfile.TemporaryDirectory() as results_dir:
        _ = _save_to_tmpfile(np.random.randn(
            num_time_steps, num_channels, height, width), dir=train_dir, filetype='h5')
        _ = _save_to_tmpfile(np.random.randn(
            num_time_steps, num_channels, height, width), dir=valid_dir, filetype='h5')
        time_means = _save_to_tmpfile(np.random.randn(
            1, num_channels + 1, height, width), dir=valid_dir, filetype='npy')
        global_means = _save_to_tmpfile(np.random.randn(
            1, num_channels + 1, height - 1, width), dir=stats_dir, filetype='npy')
        global_stds = _save_to_tmpfile(abs(np.random.randn(
            1, num_channels + 1, height - 1, width)), dir=stats_dir, filetype='npy')

        yaml_config = _get_test_yaml_file(
            train_dir, valid_dir, valid_dir, 
            results_dir, time_means, global_means, global_stds)
        params = YParams(yaml_config, "unit_test")
        params. N_in_channels = num_channels
        params. N_out_channels = num_channels
        params.enable_amp = True  # seems to be the common default
        params.log_to_wandb = False
        params.resuming = False  # do not use checkpoint
        params.roll = False
        world_rank = 0

        trainer = Trainer(params, world_rank)
        trainer.train()

test_train_runs_era5()