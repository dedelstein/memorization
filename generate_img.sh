#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J DDPM
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
#BSUB -W 23:00 
### -- Specify the output and error file. %J is the job-id -- 

module load pandas/2.1.3-python-3.10.13

source venv/bin/activate

export HF_HOME="/work3/s243891"

python3 conditional_ddpm_inference.py --model_path /work3/s243891/chest_xray_diffusion/checkpoint-138000 --output_dir chest_xray_diffusion_imgs --num_images 4 --resolution 128 --num_inference_steps 20 --guidance_scale 1.5 --batch_size 4