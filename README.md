# DyGC: Dynamic Graph Condensation

Dynamic Graph Condensation for compressing large-scale dynamic graphs.

## Structure

- src/condense.py - Small graph condensation  
- src/condense_large.py - Large graph condensation
- src/subgraph_extracter.py - Subgraph extraction for large graphs
- src/test.py / test_large.py - Test scripts

## Usage

Small graphs:
  python src/condense.py --dataset dblp --reduction_rate 0.1
  python src/test.py --dataset dblp --reduction_rate 0.1

Large graphs:
  python src/subgraph_extracter.py --dataset arxiv
  python src/condense_large.py --dataset arxiv --reduction_rate 0.01  
  python src/test_large.py --dataset arxiv --reduction_rate 0.01

## Models

TGCN, TGCN_L, DySAT, STGCN, GCRN, ROLAND
