#!/bin/bash
#SBATCH --job-name=mn-train-bgan
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/bgan.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### bgan"
python3 MNIST/mnist_etn_bgan.py 2> >(tee logs/bgan-error.log >&2) | tee logs/bgan-output.log

