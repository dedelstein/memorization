#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J OverfitDDPM
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 1:00 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o OVERFITDDPM_%J.out
#BSUB -e OVERFITDDPM_%J.err

module load pandas/2.1.3-python-3.10.13

source venv/bin/activate

export HF_HOME="/work3/s243891"

python3 conditional_ddpm_train_v2.py --overfit --data_dir /dtu/blackhole/1d/214141/CheXpert-v1.0-small   --output_dir /work3/s243891/chest_xray_diffusion_overfit_A  --resolution 128   --train_batch_size 4  --num_epochs 30000   --dataloader_num_workers 4   --learning_rate 1e-4   --use_ema   --mixed_precision fp16 --resume_from_checkpoint latest --checkpointing_steps 10000
# python3 conditional_ddpm_train_v2.py --overfit --data_dir /dtu/blackhole/1d/214141/CheXpert-v1.0-small   --output_dir /work3/s243891/chest_xray_diffusion  --resolution 128   --train_batch_size 4  --num_epochs 300   --dataloader_num_workers 4   --learning_rate 1e-4   --use_ema   --mixed_precision fp16 --resume_from_checkpoint latest
