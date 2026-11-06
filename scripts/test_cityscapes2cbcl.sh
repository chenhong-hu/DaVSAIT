#!/usr/bin/env bash
# Download davsait_cityscapes2cbcl.ckpt from the GitHub Release page into ./pretrained/ first
python test.py --name davsait_cityscapes2cbcl_test \
  --checkpoint ./pretrained/davsait_cityscapes2cbcl.ckpt \
  --data_config ./configs/cityscapes2cbcl.yaml \
  --model_config ./configs/vsait.yaml \
  --aug_config ./configs/gta2cityscapes_aug.yaml \
  --output_dir ./test_results
