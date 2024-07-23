#!/bin/sh

export HYDRA_FULL_ERROR=1  # Report full hydra error.
export NCCL_NSOCKS_PERTHREAD=4  # Pytorch tuning parameter for NCCL.
export NCCL_SOCKET_NTHREADS=2  # Pytorch tuning parameter for NCCL.
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_JIT=0

export WANDB_MODE=disabled

python train.py \
    experiment=$1 \
    ++model.batch_size=2 \
    ++dataset._train_dataset.subsampling_rate=100000 \
    ++dataset._val_dataset.subsampling_rate=100000 \
    ++trainer.check_val_every_n_epoch=50 \
    ++trainer.logger.group="debug" \
    ++trainer.max_epochs=1000 \
    ++dataset._train_dataset.split_name="train" \
    ++dataset._val_dataset.split_name="train" \
    ++trainer.devices=1 \
    ++model.debug=true
