#!/bin/sh

export HYDRA_FULL_ERROR=1  # Report full hydra error.
export NCCL_NSOCKS_PERTHREAD=4  # Pytorch tuning parameter for NCCL.
export NCCL_SOCKET_NTHREADS=2  # Pytorch tuning parameter for NCCL.

export NUM_DEVICES=${2:-4}
export NUM_EPOCHS=${3:-5}
export LIMIT_TRAIN_BATCHES=${4:-5}

conda run -n tb3 python train.py \
    experiment=$1 \
    ++hydra.mode=MULTIRUN \
    ++hydra.launcher.partition=overcap \
    ++trainer.max_epochs=$NUM_EPOCHS \
    ++dataset._train_dataset.subsampling_rate=$LIMIT_TRAIN_BATCHES \
    ++dataset._val_dataset.subsampling_rate=$LIMIT_TRAIN_BATCHES \
    ++trainer.devices=$NUM_DEVICES &
