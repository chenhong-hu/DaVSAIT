# Windows PowerShell — run from the repository root
python train.py `
  --name davsait_edges2shoes `
  --data_config ./configs/edges2shoes.yaml `
  --model_config ./configs/new_tasks_model.yaml `
  --aug_config ./configs/square256_aug.yaml `
  --output_dir ./checkpoints `
  --check_val_every_n_epoch 10 `
  --batch_size 1 `
  --seed 42 `
  --max_epochs 400 `
  --max_steps -1
