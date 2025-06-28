#!/bin/bash
#SBATCH --job-name=mn-train-lca2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/lca2-large-batch-%j.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### lca2"
python3 MNIST/mnist_etn_lca2-large-batch.py 2>&1 | tee "logs/lca2-mnist-large-batch-$(date '+%Y-%m-%d_%H-%M-%S').log"