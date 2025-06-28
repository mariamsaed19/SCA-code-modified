#!/bin/bash
#SBATCH --job-name=mn-test-lca2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=./cluster-logs/lca2-test-%j.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### lca2"
torchrun --nproc_per_node=4 MNIST/test-lca2.py 2>&1 | tee "logs/lca2-test-mnist-$(date '+%Y-%m-%d_%H-%M-%S').log"