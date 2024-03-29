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

GPUS=3
# You can use the variables DATA_DIR, OUT_DIR, CONFIG_DIR, GPUS in your script from this point onward.

docker run -it --rm \
    --gpus all \
    --mount type=bind,source=$CONFIG_DIR,target=/configs \
    --mount type=bind,source=$DATA_DIR,target=/data \
    --mount type=bind,source=$OUT_DIR,target=/out \
    --shm-size 8g \
    mmdetection:latest \
    torchrun --nnodes 1 --nproc_per_node=$GPUS  /configs/main_train_mmengine.py /configs/rtmdet_s_8xb32-300e_coco.py