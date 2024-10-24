# What Matters in Range View 3D Object Detection [CoRL 2024]

[Paper](https://openreview.net/forum?id=EifoVoIyd5)

## Datasets

- Argoverse 2.
- Waymo Open.

You will need to download both of these datasets in their entirety. After downloading, run the converters in `./converters`. This will output each dataset in the format that our codebase expects.

## Environment

You will need to install conda and install our conda environment. Please run:

```
bash conda/install.sh
```

This will install the environment to run the codebase. Note: This environment has only been tested on Ubuntu 20.04 using A40 gpus.

Additionally, to use weighted NMS you will need to install: https://github.com/Abyssaledge/TorchEx.

## Training Script

The entrypoint is found in `scripts`. For example, to train our SOTA comparison model for Argoverse 2, you would run:

`bash train.sh rv-av2 4 20 1`.
