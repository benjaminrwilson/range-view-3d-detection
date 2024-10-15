# Exporting the datasets.

This library supports the Argoverse 2 and Waymo Open datasets. You will need to preprocess both of them before they can be used. The entrypoint to export the data is found in:

1. Argoverse 2: `av2/export.py`
2. Waymo Open: `waymo/export.py`

You will need to set the `src_root_dir` and `dst_root_dir` to the source directory of the raw data and the destination directory of the processed data. Exporting may take a significant amount of time depending on your hardware.
