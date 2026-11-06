#!/usr/bin/env bash
# Aerial Photo -> Google Maps (256x256, lambda = 5, 400 epochs)
python train.py --name davsait_aerial2maps \
  --data_config ./configs/aerial2maps.yaml \
  --model_config ./configs/maps_model.yaml \
  --aug_config ./configs/square256_aug.yaml \
  --output_dir ./checkpoints \
  --check_val_every_n_epoch 10 --max_epochs 400 --max_steps -1
