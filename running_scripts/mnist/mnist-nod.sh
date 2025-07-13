#!/bin/bash
#SBATCH --job-name=mnist-nod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/mnist-nod-%j.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### nodef"
python3 ./MNIST/mnist_etn_nod.py 2>&1 | tee "logs/mnist-nodscasc.log"