#!/usr/bin/env bash
# Google Maps -> Aerial Photo (256x256, lambda = 5, 400 epochs)
python train.py --name davsait_maps2aerial \
  --data_config ./configs/maps2aerial.yaml \
  --model_config ./configs/maps_model.yaml \
  --aug_config ./configs/square256_aug.yaml \
  --output_dir ./checkpoints \
  --check_val_every_n_epoch 10 --max_epochs 400 --max_steps -1
