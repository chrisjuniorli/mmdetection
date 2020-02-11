#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train_fix.py $CONFIG --launcher pytorch ${@:3}
