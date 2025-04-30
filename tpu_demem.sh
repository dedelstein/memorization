#!/bin/sh 

python3 -m torch_xla.distributed.xmp_dist_launcher --nproc_per_host=8 conditional_ddpm_train_v2.py \
  --data_dir ~/demem_mount/CheXpert-v1.0-small \
  --output_dir ~/demem_mount/demem-output \
  --resolution 128 \
  --train_batch_size 4 \
  --num_epochs 100 \
  --dataloader_num_workers 4 \
  --learning_rate 1e-4 \
  --use_ema \
  --mixed_precision bf16 \
  --tau 3.0
