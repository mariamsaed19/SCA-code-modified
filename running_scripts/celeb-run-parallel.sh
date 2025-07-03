#!/bin/bash
#SBATCH --job-name=mn-train-celeb
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --output=./new-cluster-logs/celeba-nodef-para.out

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### nodef"
python3 CelebA/celeba_etn_nod_parallel.py 2>&1 | tee logs/celeba-nodef-para.log