#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_addr 192.168.1.1 --master_port 30000 \
    tools/train_fix.py $CONFIG --launcher pytorch ${@:3}
