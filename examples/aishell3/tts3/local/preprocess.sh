#!/bin/bash

stage=0
stop_stage=100

config_path=$1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # get durations from MFA's result
    echo "Generate durations.txt from MFA results ..."
    python ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=${ALIGNED_DATA} \
        --output durations.txt \
        --config=${config_path}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # extract features
    echo "Extract features ..."
    python ${TOOL}/preprocess.py \
        --dataset=aishell3 \
        --rootdir=${RAW_DATA} \
        --dumpdir=${DUMP_DIR} \
        --dur-file=durations.txt \
        --config=${config_path} \
        --num-cpu=20 \
        --cut-sil=True
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # get features' stats(mean and std)
    echo "Get features' stats ..."
    python ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=${DUMP_DIR}/train/raw/metadata.jsonl \
        --field-name="speech"

    python ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=${DUMP_DIR}/train/raw/metadata.jsonl \
        --field-name="pitch"

    python ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=${DUMP_DIR}/train/raw/metadata.jsonl \
        --field-name="energy"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # normalize and covert phone/speaker to id, dev and test should use train's stats
    echo "Normalize ..."
    python ${TOOL}/normalize.py \
        --metadata=${DUMP_DIR}/train/raw/metadata.jsonl \
        --dumpdir=${DUMP_DIR}/train/norm \
        --speech-stats=${DUMP_DIR}/train/speech_stats.npy \
        --pitch-stats=${DUMP_DIR}/train/pitch_stats.npy \
        --energy-stats=${DUMP_DIR}/train/energy_stats.npy \
        --phones-dict=${DUMP_DIR}/phone_id_map.txt \
        --speaker-dict=${DUMP_DIR}/speaker_id_map.txt

    python ${TOOL}/normalize.py \
        --metadata=${DUMP_DIR}/dev/raw/metadata.jsonl \
        --dumpdir=${DUMP_DIR}/dev/norm \
        --speech-stats=${DUMP_DIR}/train/speech_stats.npy \
        --pitch-stats=${DUMP_DIR}/train/pitch_stats.npy \
        --energy-stats=${DUMP_DIR}/train/energy_stats.npy \
        --phones-dict=${DUMP_DIR}/phone_id_map.txt \
        --speaker-dict=${DUMP_DIR}/speaker_id_map.txt

    python ${TOOL}/normalize.py \
        --metadata=${DUMP_DIR}/test/raw/metadata.jsonl \
        --dumpdir=${DUMP_DIR}/test/norm \
        --speech-stats=${DUMP_DIR}/train/speech_stats.npy \
        --pitch-stats=${DUMP_DIR}/train/pitch_stats.npy \
        --energy-stats=${DUMP_DIR}/train/energy_stats.npy \
        --phones-dict=${DUMP_DIR}/phone_id_map.txt \
        --speaker-dict=${DUMP_DIR}/speaker_id_map.txt
fi
