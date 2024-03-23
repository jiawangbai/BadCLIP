#!/bin/bash

cd ../..

# custom config
DATA=/path/to/datasets
TRAINER=BadClip
# TRAINER=CoOp

DATASET=caltech101
SEED=1
CFG=vit_b16_c4_ep10_batch1_ctxv1_init
SHOTS=16


DIR=output/train_seen/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python -m backdoor_attack \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi