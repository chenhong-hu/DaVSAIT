#!/usr/bin/env bash
# usage: bash scripts/test_aerial2maps.sh <path/to/checkpoint.ckpt>
CKPT=${1:?"usage: bash scripts/test_aerial2maps.sh <path/to/checkpoint.ckpt>"}
python test.py --name davsait_aerial2maps_test \
  --checkpoint "$CKPT" \
  --data_config ./configs/aerial2maps.yaml \
  --model_config ./configs/maps_model.yaml \
  --aug_config ./configs/square256_aug.yaml \
  --output_dir ./test_results
