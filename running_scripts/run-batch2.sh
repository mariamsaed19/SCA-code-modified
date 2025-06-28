#!/bin/bash
# #SBATCH --job-name=mn-train-noise
# #SBATCH --time=24:00:00
# #SBATCH --nodes=1
# #SBATCH --partition=cluster2
# #SBATCH --ntasks=1
# #SBATCH --gres=gpu:1
# #SBATCH --output=./cluster-logs/batch2.out
# #SBATCH --error=./cluster-logs/batch2.err

# source ~/.bashrc
# conda activate /scratch/mt/new-structure/conda/envs/sca

echo "########### gn"
python3 MNIST/mnist_etn_gn.py 2> >(tee logs/gn-error.log >&2) | tee logs/gn-output.log

echo "########### ln"
python3 MNIST/mnist_etn_ln.py 2> >(tee logs/ln-error.log >&2) | tee logs/ln-output.log

echo "########### lca1"
python3 MNIST/mnist_etn_lca1.py 2> >(tee logs/lca1-error.log >&2) | tee logs/lca1-output.log