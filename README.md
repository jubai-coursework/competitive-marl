# 竞争型多智能体强化学习：IPPO / MAPPO

在竞争型多智能体场景下实现并对比 IPPO 与 MAPPO，包含训练、模型保存以及曲线与轨迹可视化。

实验原理与结果分析见 [实验报告.md](实验报告.md)。

## 结构

| 路径 | 说明 |
|---|---|
| `algo.py` | 算法实现（Actor-Critic 网络与更新逻辑） |
| `train.py` | 训练入口，支持 IPPO / MAPPO 切换 |
| `main.py` | 一键脚本：依次训练 IPPO 与 MAPPO 并绘图 |
| `eval_plot.py` | 绘制评估曲线与轨迹图 |
| `results/` | 模型权重、评估记录与可视化结果 |

## 运行

```bash
# 训练单个算法
python train.py --algo IPPO  --episodes 50000
python train.py --algo MAPPO --episodes 50000

# 一键跑完两个算法并绘图（耗时较长）
python main.py
```

`train.py` 支持的参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--algo` | `IPPO` | 算法选择：`IPPO` 或 `MAPPO` |
| `--episodes` | `50000` | 训练回合数 |
| `--eval_interval` | `200` | 每隔多少回合评估一次 |
| `--eval_episodes` | `20` | 每次评估的回合数 |
| `--max_cycles` | `100` | 每回合最大步数 |
| `--resume` | 关闭 | 从已保存的模型继续训练 |

依赖：`gymnasium`、`pettingzoo[mpe]`、`torch`、`matplotlib`、`numpy`。

## 结果

`results/` 中已附带两个算法的最终权重（`*_actor.pth`、`*_best.pth`）、评估记录（`*.npy`）以及曲线图与轨迹图，可直接查看结果。
