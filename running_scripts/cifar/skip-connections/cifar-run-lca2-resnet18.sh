#!/bin/bash
#SBATCH --job-name=$EXP_NAME
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=./cluster-logs/skip-connections/lca2-%x-%j.out  # %x = job-name, %j = job-id

source ~/.bashrc
conda activate /home/m.saeed/miniconda3/envs/sca

# Read SKIP_LIST env var into x1, x2, x3, x4
IFS=',' read x1 x2 x3 x4 <<< "$SKIP_LIST"

echo "########### Running experiment: $EXP_NAME"
echo "########### With skip_list: $SKIP_LIST"
echo "########### Parsed values: [$x1, $x2, $x3, $x4]"

# Run Python (no need to pass vars if not required)
python3 CIFAR10/skip-connections/cifar10_etn_lca2_resnet18-study.py \
  2>&1 | tee logs/${EXP_NAME}.log
