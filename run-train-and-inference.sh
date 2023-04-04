#!/bin/bash

YAML_CONFIG=$1
CONFIG_SETTING=$2
NPROC_PER_NODE=$3

export WANDB_RUN_GROUP="tr-and-inf"

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node $NPROC_PER_NODE /opt/ERA5_wind/train.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING

echo ===============================================================================
echo ==================== FINISHED TRAINING / STARTING INFERENCE ===================
echo ===============================================================================

# run inference
export WANDB_JOB_TYPE=inference
python /opt/ERA5_wind/inference/inference.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING --vis
