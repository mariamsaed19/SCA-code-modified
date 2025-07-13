#!/bin/bash

# Define all combinations
combinations=(
  "0 0 0 0"
  "1 1 1 0"
  "0 1 1 1"
  "0 1 1 0"
  "1 0 0 1"
  # "0 0 0 1"
  # "0 0 1 0"
  # "0 1 0 0"
  # "1 0 0 0"
  # "0 0 1 1"
  # "0 1 0 1"
  # "1 0 1 0"
  # "1 0 1 1"
  # "1 1 0 0"
  # "1 1 0 1"
)

for combo in "${combinations[@]}"; do
  read x1 x2 x3 x4 <<< "$combo"
  skip_list="${x1},${x2},${x3},${x4}"
  exp_name="resnet_nod_${skip_list}"

  export SKIP_LIST="${skip_list}"
  export EXP_NAME="${exp_name}"

  echo "Submitting job: $EXP_NAME"
  sbatch running_scripts/cifar/skip-connections/cifar-nod-resnet18.sh
done
