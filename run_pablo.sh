#!/bin/sh
###############################################################################
#                       DDPM Model Training Job Script                        #
###############################################################################

### -- Job information and resources --
#BSUB -J DDPM_Training        # Job name
#BSUB -q gpuv100             # GPU queue for V100s
#BSUB -n 8                   # Number of cores
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode
#BSUB -R "span[hosts=1]"     # All cores on same host
#BSUB -R "rusage[mem=8GB]"   # 4GB RAM per core
#BSUB -M 9GB                 # Memory limit per core
#BSUB -W 2:00                # Walltime limit (hours:minutes)

### -- Notification options --
#BSUB -B                     # Notify at job start
#BSUB -N                     # Notify at job completion
##BSUB -u your_email_address # Uncomment and change to your email for notifications

### -- Output options --
#BSUB -o logs/DDPM_%J.out    # Standard output log
#BSUB -e logs/DDPM_%J.err    # Error log

### -- Create log directory if it doesn't exist --
mkdir -p logs

### -- Print job information --
echo "======== Job Information ========"
echo "Job ID: $LSB_JOBID"
echo "Job name: $LSB_JOBNAME"
echo "Hostname: $(hostname)"
echo "Started at: $(date)"
echo "================================="

### -- Setup environment --
echo "Setting up environment..."

# Load required modules
module load pandas/2.1.3-python-3.10.13
module list  # List loaded modules for debugging

# Initialize Conda 
source /zhome/5c/6/219415/miniconda3/etc/profile.d/conda.sh

# Activate conda environment (replace 'venv' with your actual environment name)
echo "Activating Conda environment..."
conda activate memorization
echo "Python version: $(python --version)"
echo "Conda version: $(conda --version)"

# Check GPU before running
if command -v nvidia-smi &> /dev/null; then
    echo "GPU allocated: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# Set PyTorch memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set any other environment variables
export CUDA_VISIBLE_DEVICES=0

### -- Run the training script --
echo "Starting training at $(date)"
echo "Command: python3 train_cfg_diffusion.py --batch_size 4"

# Run training
python3 train_cfg_diffusion.py --batch_size 8 --img_size 64 --debug_mode

### -- Job cleanup and information --
echo "Training finished at $(date)"

# Output GPU statistics if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi

# Save important information to job report
{
    echo "======== Job Report ========"
    echo "Job completed at: $(date)"
    echo "Total runtime: $(($(date +%s) - $LSB_JOBSTART)) seconds"
    echo "Exit status: $?"
    echo "============================"
} > "logs/job_report_${LSB_JOBID}.txt"

# Deactivate the conda environment
conda deactivate

echo "Job completed"