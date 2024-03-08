DATA_DRIVE="/mnt/SSD2/Brackish_Underwater_v2_960x540_coco/"
TRAIN_IMG_FOLDER=train/
VAL_IMG_FOLDER=valid/
TEST_IMG_FOLDER=test/
TRAIN_ANNOT=train/_annotations.coco.json
VAL_ANNOT=valid/_annotations.coco.json
TEST_ANNOT=test/_annotations.coco.json

OUT_DIR="$PWD/out"
WORK_DIR="/out"
WARMUP_ITERS=1000
MAX_EPOCHS=20
STAGE2_NUM_EPOCHS=10
BASE_LR=0.01
INTERVAL=10


GPUS=3

docker run -it --rm \
    --gpus all \
    -e TRAIN_IMG_FOLDER=$TRAIN_IMG_FOLDER \
    -e VAL_IMG_FOLDER=$VAL_IMG_FOLDER \
    -e TEST_IMG_FOLDER=$TEST_IMG_FOLDER \
    -e TRAIN_ANNOT=$TRAIN_ANNOT \
    -e VAL_ANNOT=$VAL_ANNOT \
    -e TEST_ANNOT=$TEST_ANNOT \
    -e WARMUP_ITERS=$WARMUP_ITERS \
    -e STAGE2_NUM_EPOCHS=$STAGE2_NUM_EPOCHS \
    -e MAX_EPOCHS=$MAX_EPOCHS \
    -e BASE_LR=$BASE_LR \
    -e INTERVAL=$INTERVAL \
    -e WORK_DIR=$WORK_DIR \
    --mount type=bind,source=$PWD/codes/,target=/configs \
    --mount type=bind,source=$DATA_DRIVE,target=/data \
    --mount type=bind,source=$OUT_DIR,target=/out \
    --shm-size 8g \
    mmdetection:latest \
    torchrun --nnodes 1 --nproc_per_node=$GPUS  /configs/main_train_mmengine.py /configs/rtmdet_s_8xb32-300e_coco.py