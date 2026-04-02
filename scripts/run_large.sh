#!/bin/bash
# Script for large-scale graph condensation (arxiv)

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

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
K=3

# Condensation settings
REDUCTION_RATE=0.0025
LOSS_FACTOR=10
TEMPORAL_ALPHA=0.5
LR_FEAT=0.05
LR_ADJ=0.05

# Subgraph extraction settings
GRAPH_SIZE=100000
SAMPLE_DEPTH=3

# Training settings
BATCH_SIZE=100000
CONDENSING_LOOP=1000
CONDENSING_VAL_STAGE=100

# Step 1: Download and preprocess arxiv data (only needed once)
if [ ! -f "data/processed/arxiv.pt" ]; then
    echo "Step 1: Downloading arxiv dataset..."
    python -u data/scripts/get_arxiv.py
fi

# Step 2: Extract subgraphs (only needed once)
if [ ! -f "data/splited/${DATASET}_train_batch.pt" ]; then
    echo "Step 2: Extracting subgraphs..."
    python -u src/subgraph_extracter.py \
        --dataset ${DATASET} \
        --cuda ${CUDA} \
        --seed ${SEED} \
        --graph_size ${GRAPH_SIZE} \
        --sample_depth ${SAMPLE_DEPTH}
fi

# Step 3: Run condensation
echo "Step 3: Running condensation on ${DATASET}..."
python -u src/condense_large.py \
    --seed ${SEED} \
    --cuda ${CUDA} \
    --dataset ${DATASET} \
    --teacher_model ${TEACHER_MODEL} \
    --val_model ${VAL_MODEL} \
    --nlayers ${NLAYERS} \
    --hidden ${HIDDEN} \
    --dropout ${DROPOUT} \
    --K ${K} \
    --reduction_rate ${REDUCTION_RATE} \
    --loss_factor ${LOSS_FACTOR} \
    --temporal_alpha ${TEMPORAL_ALPHA} \
    --lr_feat ${LR_FEAT} \
    --lr_adj ${LR_ADJ} \
    --batch_size ${BATCH_SIZE} \
    --condensing_loop ${CONDENSING_LOOP} \
    --condensing_val_stage ${CONDENSING_VAL_STAGE}

# Step 4: Test condensed graph
echo "Step 4: Testing condensed graph..."
python -u src/test_large.py \
    --seed ${SEED} \
    --testseed ${SEED} \
    --cuda ${CUDA} \
    --dataset ${DATASET} \
    --test_model ${VAL_MODEL} \
    --nlayers ${NLAYERS} \
    --hidden ${HIDDEN} \
    --dropout ${DROPOUT} \
    --reduction_rate ${REDUCTION_RATE} \
    --batch_size ${BATCH_SIZE} \
    --test_loop 2000 \
    --val_stage 100
