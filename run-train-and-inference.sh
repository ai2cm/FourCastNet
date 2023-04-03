#!/bin/bash

YAML_CONFIG=/configmount/config.yaml
CONFIG_SETTING=full_field

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node 1 /opt/ERA5_wind/train.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING

echo ===============================================================================
echo ======================FINISHED TRAINING STARTING INFERENCE=====================
echo ===============================================================================

# run inference
export WANDB_JOB_TYPE=inference
TODO(gideond) factor out /output
python /opt/ERA5_wind/inference/inference.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING --override_dir /output --weights /output/$CONFIG_SETTING/00/training_checkpoints/best_ckpt.tar --vis
