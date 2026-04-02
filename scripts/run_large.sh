#!/bin/bash
# Example script for large-scale graph condensation

# Dataset and basic settings
DATASET="arxiv"
SEED=1
CUDA=0

# Model settings
TEACHER_MODEL="TGCN_L"
VAL_MODEL="TGCN_L"
NLAYERS=3
HIDDEN=256
DROPOUT=0.5

# Condensation settings
REDUCTION_RATE=0.01
LOSS_FACTOR=10
TEMPORAL_ALPHA=0.05
LR_FEAT=0.05
LR_ADJ=0.05

# Training settings
CONDENSING_LOOP=1500
CONDENSING_VAL_STAGE=100

# Step 1: Download and preprocess arxiv data (only needed once)
if [ ! -f "data/processed/arxiv.pt" ]; then
    echo "Step 1: Downloading arxiv dataset..."
    python -u src/datasets/get_arxiv.py
fi

# Step 2: Extract subgraphs (only needed once)
echo "Step 2: Extracting subgraphs..."
python -u src/subgraph_extracter.py \
    --dataset ${DATASET} \
    --cuda ${CUDA} \
    --seed 2024 \
    --sample_depth 3 \
    --graph_size 100000

# Step 3: Run condensation
for seed in 1 2 3 4 5
do
    echo "Running condensation with seed=${seed}..."
    python -u src/condense_large.py \
        --seed ${seed} \
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
done

# Step 4: Test condensed graph
echo "Testing condensed graph..."
python -u src/test_large.py \
    --seed 1 \
    --cuda ${CUDA} \
    --dataset ${DATASET} \
    --test_model ${VAL_MODEL} \
    --nlayers ${NLAYERS} \
    --hidden ${HIDDEN} \
    --dropout ${DROPOUT} \
    --reduction_rate ${REDUCTION_RATE} \
    --test_loop 2000 \
    --val_stage 100
