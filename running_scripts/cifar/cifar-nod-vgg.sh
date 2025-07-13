#!/bin/bash
#SBATCH --job-name=cifar-nod-vgg
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/cifar-nod-vgg.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### nodef"
python3 CIFAR10/cifar10_etn_nod_vgg.py 2>&1 | tee logs/cifar-nod-vgg.log