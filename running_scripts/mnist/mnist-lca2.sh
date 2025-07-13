#!/bin/bash
#SBATCH --job-name=mnist-lca2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/mnist-lca2-test-%j.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### lca2"
# python3 ./MNIST/mnist_etn_lca2.py 2>&1 | tee "logs/mnist-lca2.log"
python3 ./MNIST/mnist_etn_lca2-test.py 2>&1 | tee "logs/mnist-lca2-test.log"