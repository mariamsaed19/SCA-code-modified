#!/bin/bash
#SBATCH --job-name=celeb-lca2-resnet18
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/resnet18-celeba-lca2-%j.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### lca2-resnet"
python3 CelebA/celeba_etn_lca2_resnet18.py 2>&1 | tee logs/resnet18-celeba-lca2.log
# python3 CelebA/celeba_etn_lca2_resnet18_test.py 2>&1 | tee logs/resnet18-celeba-lca2-test-output-again.log