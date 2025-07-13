#!/bin/bash
#SBATCH --job-name=celeb-nod-vgg
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/celeba-nod-vgg-%j.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### nodef"
python3 CelebA/celeba_etn_nod_vgg.py 2>&1 | tee logs/celeba-nod_vgg.log