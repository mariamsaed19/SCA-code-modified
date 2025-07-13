#!/bin/bash
#SBATCH --job-name=celeb-lca2-cnn
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/celeba-lca2.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### lca2"
python3 CelebA/celeba_etn_lca2.py 2>&1 | tee logs/celeba-lca2.log