#
# Created on Sun Mar 10 2024
#
# Copyright (c) 2024 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#

DATA_DIR="/mnt/SSD2/kvasir-instrument/"
OUT_DIR="$PWD/out"
CONFIG_DIR=$PWD/codes/
NOTEBOOK_DIR=$PWD/notebooks/

GPUS=3

docker run -it --rm \
    --gpus all \
    --mount type=bind,source=$CONFIG_DIR,target=/configs \
    --mount type=bind,source=$DATA_DIR,target=/data \
    --mount type=bind,source=$OUT_DIR,target=/out \
    --mount type=bind,source=$NOTEBOOK_DIR,target=/notebooks \
    --shm-size 8g \
    -p 8888:8888 \
    mmdetection:latest \
    jupyter-lab  --ip 0.0.0.0 --port 8888 --allow-root /notebooks