#!/bin/bash
# Script for small-scale graph condensation (dblp, reddit)

# Dataset and basic settings
DATASET="dblp"
SEED=1
CUDA=0

# Model settings
TEACHER_MODEL="TGCN"
VAL_MODEL="TGCN"
NLAYERS=2
HIDDEN=128
DROPOUT=0.5

# Condensation settings
REDUCTION_RATE=0.05
LOSS_FACTOR=50
TEMPORAL_ALPHA=0.5
LR_FEAT=0.05
LR_ADJ=0.05

# Training settings
CONDENSING_LOOP=1000
CONDENSING_VAL_STAGE=100

# Step 1: Preprocess data (only needed once)
if [ ! -f "data/processed/${DATASET}.pt" ]; then
    echo "Step 1: Preprocessing ${DATASET} dataset..."
    python -u data/scripts/get_data.py --data ${DATASET}
fi

# Step 2: Run condensation
echo "Step 2: Running condensation on ${DATASET}..."
python -u src/condense.py \
    --seed ${SEED} \
    --cuda ${CUDA} \
    --dataset ${DATASET} \
    --teacher_model ${TEACHER_MODEL} \
    --val_model ${VAL_MODEL} \
    --nlayers ${NLAYERS} \
    --hidden ${HIDDEN} \
    --dropout ${DROPOUT} \
    --reduction_rate ${REDUCTION_RATE} \
    --loss_factor ${LOSS_FACTOR} \
    --temporal_alpha ${TEMPORAL_ALPHA} \
    --lr_feat ${LR_FEAT} \
    --lr_adj ${LR_ADJ} \
    --condensing_loop ${CONDENSING_LOOP} \
    --condensing_val_stage ${CONDENSING_VAL_STAGE}

# Step 3: Test condensed graph
echo "Step 3: Testing condensed graph..."
python -u src/test.py \
    --seed ${SEED} \
    --cuda ${CUDA} \
    --dataset ${DATASET} \
    --test_model ${VAL_MODEL} \
    --nlayers ${NLAYERS} \
    --hidden ${HIDDEN} \
    --dropout ${DROPOUT} \
    --reduction_rate ${REDUCTION_RATE} \
    --test_loop 1000 \
    --val_stage 50
