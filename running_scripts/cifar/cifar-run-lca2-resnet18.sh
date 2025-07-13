#!/bin/bash
#SBATCH --job-name=cifar-lca2-resnet18
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/cifar-lca2-resnet18-test.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### lca2"
python3 CIFAR10/cifar10_etn_lca2_resnet18_test.py 2>&1 | tee logs/cifar-lca2-resnet18_test.log