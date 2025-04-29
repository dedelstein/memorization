#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -J DDPM_Training_128_v_ambient # Job name
#BSUB -q gpuv100             # GPU queue for V100s
#BSUB -n 4                   # Number of cores
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -R "span[hosts=1]"     # All cores on same host
#BSUB -R "rusage[mem=8GB]"   # 4GB RAM per core
#BSUB -R "select[gpu32gb]"   # 32 GB GPU
#BSUB -M 9GB                 # Memory limit per core
#BSUB -W 24:00                # Walltime limit (hours:minutes)

### -- Notification options --
#BSUB -B                     # Notify at job start
#BSUB -N                     # Notify at job completion
##BSUB -u your_email_address # Uncomment and change to your email for notifications

### -- Output options --
#BSUB -o logs/DDPM_ambient_%J.out    # Standard output log
#BSUB -e logs/DDPM_ambient_%J.err    # Error log
#BSUB -e Output_ambient_%J.err 

source /zhome/91/9/214141/default_venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set any other environment variables
export CUDA_VISIBLE_DEVICES=0

python3 train_ambient_diffusion.py --batch_size 4 --ambient_t_nature 0.5 --epochs 40 --img_size=128
