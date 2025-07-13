#!/bin/bash
#SBATCH --job-name=cifar-lca2-resnet_test
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/transfer/cifar-lca2-resnet_test-transfer.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### lca2"
python3 CIFAR10/transfer/cifar10_etn_lca2_resnet_test.py 2>&1 | tee logs/cifar-lca2-resnet_test_transfer.log
