#!/bin/bash
#SBATCH --job-name=mn-train-ln
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/ln.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### ln"
python3 MNIST/mnist_etn_ln.py 2> >(tee logs/ln-error.log >&2) | tee logs/ln-output.log