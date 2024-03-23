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
LOADEP=10
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/train_seen/${COMMON_DIR}
DIR=output/test_unseen/${COMMON_DIR}
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
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi