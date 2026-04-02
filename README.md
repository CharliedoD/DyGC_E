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

## Quick Start

```sh
# Small-scale graphs (dblp, reddit)
sh scripts/run_small.sh

# Large-scale graphs (arxiv, tmall)
sh scripts/run_large.sh
```

## License

This project is licensed under the MIT License.
