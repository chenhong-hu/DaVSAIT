#!/usr/bin/env bash
# Cityscapes -> CBCL (resize 512x256, random crop 256, lambda = 10, 20 epochs)
python train.py --name davsait_cityscapes2cbcl \
  --data_config ./configs/cityscapes2cbcl.yaml \
  --model_config ./configs/vsait.yaml \
  --aug_config ./configs/gta2cityscapes_aug.yaml \
  --output_dir ./checkpoints \
  --check_val_every_n_epoch 1 --max_epochs 20 --max_steps -1
