#!/bin/sh
#BSUB -q gpuv100
#BSUB -J TCNP
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=300MB]"
#BSUB -W 2:00
#BSUB -o batch_output/%J.out
#BSUB -e batch_output/%J.err

module load python3/3.10.13

source venv_1/bin/activate

python TCNP.py checkpoint-.....
