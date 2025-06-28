#!/bin/bash
#SBATCH --job-name=mn-train-lca
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/lca1.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### lca1"
python3 MNIST/mnist_etn_lca1.py 2> >(tee logs/lca1-error.log >&2) | tee logs/lca1-output.log