#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON tools/train_fix.py $CONFIG --gpus $GPUS --validate
#${@:3}
