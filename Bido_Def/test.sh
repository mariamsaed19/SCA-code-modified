#!/bin/bash
#SBATCH --job-name=Bd_M
#SBATCH --time=23:00:00
#SBATCH --output=python_%j.txt
#SBATCH --error=python_%j.err
#SBATCH --partition=v100_12
#SBATCH --requeue
#SBATCH --gres=gpu:2
# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="discovery"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

# The code you want to run in your job
# The code you want to run in your job
conda activate iguard

export OMP_NUM_THREADS=1

nohup python mnist_etn.py &

sleep 1000000