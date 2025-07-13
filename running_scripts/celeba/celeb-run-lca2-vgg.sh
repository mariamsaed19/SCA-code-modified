#!/bin/bash
#SBATCH --job-name=celeb-lca2-vgg
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/vgg-celeba-lca2-%j.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### lca2-vgg"
python3 CelebA/celeba_etn_lca2_vgg.py 2>&1 | tee logs/vgg-celeba-lca2.log