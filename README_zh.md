<h1 align="center">Evolution-Consistent Dynamic Graph Condensation</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2506.13099"><img src="https://img.shields.io/badge/arXiv-2506.13099-b31b1b" alt="arXiv"></a>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

DyGC（动态图蒸馏）是一个用于压缩大规模动态图的框架，同时保持图的时序和结构特性。它学习一个小型合成图，使得在该图上训练的 GNN 模型能够达到与在完整图上训练相当的性能。

## 框架

![DyGC Framework](figs/framework.png)

## 项目结构

```
DyGC/
├── scripts/                         # 运行脚本
│   ├── run_small.sh                 # 小规模图脚本 (dblp, reddit)
│   └── run_large.sh                 # 大规模图脚本 (arxiv, tmall)
├── src/                             # 源代码
│   ├── condense.py                  # 小规模图蒸馏
│   ├── condense_large.py            # 大规模图蒸馏
│   ├── subgraph_extracter.py        # 大图子图提取
│   ├── test.py                      # 小图测试脚本
│   ├── test_large.py                # 大图测试脚本
│   ├── models/                      # 模型实现
│   │   ├── DGNN.py                  # 动态 GNN 模型
│   │   ├── basicgnn.py              # 基础 GNN 组件
│   │   ├── structure_generation.py  # 结构学习模块
│   │   └── convs/                   # 图卷积层
│   └── utils/                       # 工具函数
│       ├── graph_utils.py           # 图数据工具
│       ├── kernels.py               # 核函数
│       └── losses.py                # 损失函数 (MMD)
├── data/                            # 数据目录
│   ├── raw/                         # 原始数据文件 (.npz)
│   ├── processed/                   # 处理后数据文件 (.pt)
│   ├── splited/                     # 大图采样子图
│   └── scripts/                     # 数据预处理脚本
│       ├── get_data.py              # 处理小规模数据集
│       └── get_arxiv.py             # 下载处理 arxiv
├── syn/                             # 蒸馏图输出
├── teacher/                         # 训练好的教师模型
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

将原始数据文件放入 `data/raw/`：

```
data/raw/
├── dblp.npz       # DBLP 合作网络
└── reddit.npz     # Reddit 讨论网络
```

对于 arxiv 数据集，运行 `run_large.sh` 时会自动下载。

## 快速开始

```sh
# 小规模图 (dblp, reddit)
sh scripts/run_small.sh

# 大规模图 (arxiv, tmall)
sh scripts/run_large.sh
```

## 命令行参数

### 蒸馏 (condense.py / condense_large.py)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `dblp` | 数据集名称 |
| `--cuda` | int | `0` | GPU 设备 ID |
| `--seed` | int | `2025` | 随机种子 |
| `--reduction_rate` | float | `0.1` | 图压缩率 |
| `--teacher_model` | str | `TGCN` | 教师模型架构 |
| `--val_model` | str | `TGCN` | 验证模型架构 |
| `--nlayers` | int | `2` | GNN 层数 |
| `--hidden` | int | `128` | 隐藏层维度 |
| `--dropout` | float | `0.5` | Dropout 率 |
| `--K` | int | `2` | 特征聚合传播深度 |
| `--lr_feat` | float | `0.01` | 特征学习率 |
| `--lr_adj` | float | `0.01` | 邻接矩阵学习率 |
| `--loss_factor` | float | `10` | MMD 损失权重 |
| `--temporal_alpha` | float | `0.1` | 时序平滑因子 |
| `--condensing_loop` | int | `200` | 蒸馏迭代次数 |

### 测试 (test.py / test_large.py)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `dblp` | 数据集名称 |
| `--reduction_rate` | float | `0.1` | 蒸馏时使用的压缩率 |
| `--test_model` | str | `TGCN` | 在蒸馏图上训练的模型 |
| `--test_loop` | int | `1000` | 训练轮数 |
| `--val_stage` | int | `50` | 验证频率 |

## 许可证

本项目采用 MIT 许可证。
