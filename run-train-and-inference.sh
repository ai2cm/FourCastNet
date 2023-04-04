#!/bin/bash

YAML_CONFIG=$1
CONFIG_SETTING=$2
NPROC_PER_NODE=$3
WANDB_UUID=$(python -c "import wandb; print(wandb.util.generate_id())")
export WANDB_RUN_GROUP=fourcastnet-train-and-inference-$WANDB_UUID

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node $NPROC_PER_NODE /opt/ERA5_wind/train.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING

echo ===============================================================================
echo ======================FINISHED TRAINING STARTING INFERENCE=====================
echo ===============================================================================

# run inference
export WANDB_JOB_TYPE=inference
#TODO(gideond) factor out /output
python /opt/ERA5_wind/inference/inference.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING --vis
