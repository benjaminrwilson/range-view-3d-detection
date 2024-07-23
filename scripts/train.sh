#!/bin/sh

export HYDRA_FULL_ERROR=1  # Report full hydra error.
export NCCL_NSOCKS_PERTHREAD=4  # Pytorch tuning parameter for NCCL.
export NCCL_SOCKET_NTHREADS=2  # Pytorch tuning parameter for NCCL.

# export OMP_NUM_THREADS=4
# export SRUN_CPUS_PER_TASK=4
export WANDB_MODE=disabled

export NUM_DEVICES=${2:-1}
export NUM_EPOCHS=${3:-1}
export LIMIT_TRAIN_BATCHES=${4:-5}
export PYTORCH_JIT=1

python train.py \
    experiment=$1 \
    ++trainer.max_epochs=$NUM_EPOCHS \
    ++dataset._train_dataset.subsampling_rate=$LIMIT_TRAIN_BATCHES \
    ++dataset._val_dataset.subsampling_rate=$LIMIT_TRAIN_BATCHES \
    ++trainer.devices=$NUM_DEVICES
