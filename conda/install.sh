#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENVIRONMENT_FILE=environment.yml

if [[ "$OSTYPE" == "darwin"* ]]; then
    ENVIRONMENT_FILE=environment-mps.yml
fi

echo "Using ${ENVIRONMENT_FILE} ..."

# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#     export CONDA_OVERRIDE_CUDA="11.2"
#     export FORCE_CUDA=1
#     export TORCH_CUDA_ARCH_LIST="7.0+PTX 8.0"
# fi

# Ensure mamba is installed.
conda install -y mamba

# Create library environment.
mamba env create -f ${SCRIPT_DIR}/${ENVIRONMENT_FILE} \
&& eval "$(conda shell.bash hook)" \
&& conda activate tb3 \
&& OPENSSL_DIR=$CONDA_PREFIX ETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install --no-build-isolation --no-deps -e .. \
&& pip install waymo-open-dataset-tf-2-11-0 --no-deps
