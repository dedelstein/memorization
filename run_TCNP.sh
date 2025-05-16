#!/bin/sh
#BSUB -q gpuv100
#BSUB -J TCNP
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=300MB]"
#BSUB -W 1:30
#BSUB -o TCNP_%J.out
#BSUB -e TCNP_%J.err

module load python3/3.10.13
 
source venv/bin/activate

export HF_HOME="/work3/s243891"

python TCNP.py /work3/s243891/chest_xray_diffusion_overfit_800e/checkpoint-145000 
