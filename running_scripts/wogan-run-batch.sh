#!/bin/bash
#SBATCH --job-name=mn-train-wogan
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/wogan.out


source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca


echo "########### wogan"
python3 MNIST/mnist_etn_wogan.py 2> >(tee logs/wogan-error.log >&2) | tee logs/wogan-output.log

