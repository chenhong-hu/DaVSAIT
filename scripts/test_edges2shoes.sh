#!/usr/bin/env bash
# Download davsait_edges2shoes.ckpt from the GitHub Release page into ./pretrained/ first
python test.py \
  --name davsait_edges2shoes_test \
  --checkpoint ./pretrained/davsait_edges2shoes.ckpt \
  --data_config ./configs/edges2shoes.yaml \
  --model_config ./configs/new_tasks_model.yaml \
  --aug_config ./configs/square256_aug.yaml \
  --output_dir ./test_results
