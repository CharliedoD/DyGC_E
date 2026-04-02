# Command-line Arguments

## Condensation (condense.py / condense_large.py)

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

## Testing (test.py / test_large.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `dblp` | Dataset name |
| `--reduction_rate` | float | `0.1` | Reduction rate used in condensation |
| `--test_model` | str | `TGCN` | Model to train on condensed graph |
| `--test_loop` | int | `1000` | Training epochs |
| `--val_stage` | int | `50` | Validation frequency |

## Supported Models

- **TGCN**: Temporal Graph Convolutional Network
- **TGCN_L**: TGCN for large-scale graphs
- **DySAT**: Dynamic Self-Attention Network
- **STGCN**: Spatio-Temporal Graph Convolutional Network
- **GCRN**: Graph Convolutional Recurrent Network
- **ROLAND**: Robust Dynamic Graph Neural Network
