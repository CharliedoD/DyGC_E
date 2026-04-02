<h1 align="center">Evolution-Consistent Dynamic Graph Condensation</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2506.13099"><img src="https://img.shields.io/badge/arXiv-2506.13099-b31b1b" alt="arXiv"></a>
</p>

<p align="center">
  <a href="README_zh.md">中文版</a>
</p>

DyGC (Dynamic Graph Condensation) is a framework for compressing large-scale dynamic graphs while preserving their temporal and structural properties. It learns a small synthetic graph that can train GNN models to achieve comparable performance to training on the full graph.

## Framework

![DyGC Framework](figs/framework.png)

## Project Structure

```
DyGC/
├── scripts/                         # Run scripts
│   ├── run_small.sh                 # Script for small graphs (dblp, reddit)
│   └── run_large.sh                 # Script for large graphs (arxiv, tmall)
├── src/                             # Source code
│   ├── condense.py                  # Small-scale graph condensation
│   ├── condense_large.py            # Large-scale graph condensation
│   ├── subgraph_extracter.py        # Subgraph extraction for large graphs
│   ├── test.py                      # Test script for small graphs
│   ├── test_large.py                # Test script for large graphs
│   ├── models/                      # Model implementations
│   │   ├── DGNN.py                  # Dynamic GNN models (TGCN, DySAT, etc.)
│   │   ├── basicgnn.py              # Basic GNN components
│   │   ├── structure_generation.py  # Structure learning module
│   │   └── convs/                   # Graph convolution layers
│   └── utils/                       # Utility functions
│       ├── graph_utils.py           # Graph data utilities
│       ├── kernels.py               # Kernel functions
│       └── losses.py                # Loss functions (MMD)
├── data/                            # Data directory
│   ├── raw/                         # Raw data files (.npz)
│   ├── processed/                   # Processed data files (.pt)
│   ├── splited/                     # Sampled subgraphs for large graphs
│   └── scripts/                     # Data preprocessing scripts
│       ├── get_data.py              # Process small datasets
│       └── get_arxiv.py             # Download and process arxiv
├── syn/                             # Condensed graphs output
├── teacher/                         # Trained teacher models
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Place raw data files in `data/raw/`:

```
data/raw/
├── dblp.npz       # DBLP co-authorship graph
└── reddit.npz     # Reddit discussion graph
```

For arxiv dataset, it will be downloaded automatically when running `run_large.sh`.

## Usage

### Quick Start

```sh
# Small-scale graphs (dblp, reddit)
sh scripts/run_small.sh

# Large-scale graphs (arxiv, tmall)
sh scripts/run_large.sh
```

### Step-by-Step

#### Small Graphs

```bash
# Step 1: Preprocess data
python data/scripts/get_data.py --data dblp

# Step 2: Run condensation
python src/condense.py \
    --dataset dblp \
    --reduction_rate 0.05 \
    --teacher_model TGCN \
    --val_model TGCN

# Step 3: Test condensed graph
python src/test.py \
    --dataset dblp \
    --reduction_rate 0.05 \
    --test_model TGCN
```

#### Large Graphs

```bash
# Step 1: Download and preprocess arxiv
python data/scripts/get_arxiv.py

# Step 2: Extract subgraphs
python src/subgraph_extracter.py \
    --dataset arxiv \
    --graph_size 100000 \
    --sample_depth 3

# Step 3: Run condensation
python src/condense_large.py \
    --dataset arxiv \
    --reduction_rate 0.0025 \
    --teacher_model TGCN_L \
    --val_model TGCN_L

# Step 4: Test condensed graph
python src/test_large.py \
    --dataset arxiv \
    --reduction_rate 0.0025 \
    --test_model TGCN_L
```

### Command-line Arguments

#### Condensation (condense.py / condense_large.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `dblp` | Dataset name |
| `--cuda` | int | `0` | GPU device id |
| `--seed` | int | `2025` | Random seed |
| `--reduction_rate` | float | `0.1` | Graph reduction rate |
| `--teacher_model` | str | `TGCN` | Teacher model architecture |
| `--val_model` | str | `TGCN` | Validation model architecture |
| `--nlayers` | int | `2` | Number of GNN layers |
| `--hidden` | int | `128` | Hidden dimension |
| `--dropout` | float | `0.5` | Dropout rate |
| `--K` | int | `2` | Propagation depth for feature aggregation |
| `--lr_feat` | float | `0.01` | Learning rate for features |
| `--lr_adj` | float | `0.01` | Learning rate for adjacency |
| `--loss_factor` | float | `10` | MMD loss weight |
| `--temporal_alpha` | float | `0.1` | Temporal smoothing factor |
| `--condensing_loop` | int | `200` | Condensation iterations |

#### Testing (test.py / test_large.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `dblp` | Dataset name |
| `--reduction_rate` | float | `0.1` | Reduction rate used in condensation |
| `--test_model` | str | `TGCN` | Model to train on condensed graph |
| `--test_loop` | int | `1000` | Training epochs |
| `--val_stage` | int | `50` | Validation frequency |

### Supported Models

- **TGCN**: Temporal Graph Convolutional Network
- **TGCN_L**: TGCN for large-scale graphs
- **DySAT**: Dynamic Self-Attention Network
- **STGCN**: Spatio-Temporal Graph Convolutional Network
- **GCRN**: Graph Convolutional Recurrent Network
- **ROLAND**: Robust Dynamic Graph Neural Network

## License

This project is licensed under the MIT License.
