#!/usr/bin/env bash
# Download davsait_gta2cityscapes.ckpt from the GitHub Release page into ./pretrained/ first
python test.py \
  --name davsait_gta2cityscapes_test \
  --checkpoint ./pretrained/davsait_gta2cityscapes.ckpt \
  --data_config ./configs/gta2cityscapes.yaml \
  --model_config ./configs/vsait.yaml \
  --aug_config ./configs/gta2cityscapes_aug.yaml \
  --output_dir ./test_results
