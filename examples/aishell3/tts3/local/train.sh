#!/bin/bash

config_path=$1
train_output_path=$2

python ${BIN_DIR}/train.py \
    --train-metadata=${DUMP_DIR}/train/norm/metadata.jsonl \
    --dev-metadata=${DUMP_DIR}/dev/norm/metadata.jsonl \
    --config=${config_path} \
    --output-dir=${train_output_path} \
    --ngpu=2 \
    --phones-dict=${DUMP_DIR}/phone_id_map.txt \
    --speaker-dict=${DUMP_DIR}/speaker_id_map.txt
