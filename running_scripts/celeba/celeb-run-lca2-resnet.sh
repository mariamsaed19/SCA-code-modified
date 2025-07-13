#!/bin/bash
#SBATCH --job-name=celeb-lca2-resnet
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/resnet-celeba-lca2-%j.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### lca2-resnet"
python3 CelebA/celeba_etn_lca2_resnet.py 2>&1 | tee logs/resnet-celeba-lca2.log
# python3 CelebA/celeba_etn_lca2_resnet_test.py 2>&1 | tee logs/resnet-celeba-lca2-test-output-again.log