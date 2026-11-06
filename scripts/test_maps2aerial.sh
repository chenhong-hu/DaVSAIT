#!/usr/bin/env bash
# usage: bash scripts/test_maps2aerial.sh <path/to/checkpoint.ckpt>
CKPT=${1:?"usage: bash scripts/test_maps2aerial.sh <path/to/checkpoint.ckpt>"}
python test.py --name davsait_maps2aerial_test \
  --checkpoint "$CKPT" \
  --data_config ./configs/maps2aerial.yaml \
  --model_config ./configs/maps_model.yaml \
  --aug_config ./configs/square256_aug.yaml \
  --output_dir ./test_results
