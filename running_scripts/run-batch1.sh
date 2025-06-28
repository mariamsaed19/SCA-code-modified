#!/bin/bash
#SBATCH --job-name=mn-train-gan
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/batch1.out
#SBATCH --error=./cluster-logs/batch1.err

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

echo "########### bgan"
python3 MNIST/mnist_etn_bgan.py 2> >(tee logs/bgan-error.log >&2) | tee logs/bgan-output.log

# echo "########### gan"
# python3 MNIST/mnist_etn_gan.py 2> >(tee logs/gan-error.log >&2) | tee logs/gan-output.log

echo "########### wogan"
python3 MNIST/mnist_etn_wogan.py 2> >(tee logs/wogan-error.log >&2) | tee logs/wogan-output.log

echo "########### lca2"
python3 MNIST/mnist_etn_lca2.py 2> >(tee logs/lca2-error.log >&2) | tee logs/lca2-output.log