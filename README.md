# Generals.io RL AI — 强化学习智能体

基于 C++ 引擎 + PyTorch 的 generals.io AI 训练与对战系统。

## 项目架构

```
generals.io/
├── src/
│   ├── cpp/                    # C++ 高性能引擎
│   │   ├── game_state.h        # 完美信息游戏状态
│   │   ├── belief_state.h      # 迷雾 + 贝叶斯滤波
│   │   ├── flow_field.h        # 矢量流场寻路
│   │   ├── is_mcts.h           # IS-MCTS 搜索树 (含 NN 先验)
│   │   ├── script_agent.h      # 脚本 AI (扩张/城市/进攻)
│   │   ├── nn_predictor.h      # LibTorch C++ 推理
│   │   ├── generals_engine.cpp # C API + main
│   │   ├── libgenerals.so      # 游戏引擎 (81KB)
│   │   └── libgenerals_nn.so   # NN 推理引擎 (194KB)
│   └── python/                 # Python RL 层
│       ├── play_gui.py         # Pygame 图形化对战界面
│       ├── verify_rule_consistency.py  # 规则一致性验证
│       ├── v4_attention.py     # v4 Attention 网络训练
│       ├── train_v3.py         # v3 FCN 训练脚本
│       ├── train_resnet.py     # ResNet 训练脚本
│       ├── evaluate_nn_vs_pure.py      # NN+MCTS vs 纯MCTS 评估
│       ├── generate_cycle2.py  # Cycle 2 数据生成
│       ├── generate_cycle3.py  # Cycle 3 数据生成
│       ├── generals_gym_cpp.py # C++ 引擎 Python 包装器
│       ├── script_agent.py     # 脚本 AI Python 接口
│       ├── rl_models/          # 训练好的模型
│       └── rl_data/            # 训练数据 (npz)
├── libtorch/                   # LibTorch 2.4.1 CPU 版
└── archive/rl/                 # 历史训练代码
```

## 核心架构

### 游戏引擎 (C++)

- **DOD 设计**: 每个模块独立头文件，零依赖耦合
- **IS-MCTS**: 信息集 MCTS，支持 NN 先验 + 流场 rollout
- **BeliefState**: 贝叶斯滤波推断敌方将军位置
- **Flow Field**: 矢量流场寻路，替代随机 rollout (2x 速度)

### RL 训练循环

```
NN+MCTS 自对弈 → 收集数据 (8x 对称增强) → 训练 CNN → 先验指导 MCTS → 循环
```

### 网络架构

| 模型 | 参数 | 数据 | 结果 |
|:---|:---:|:---|:---|
| v3 FCN | 2.7M | 598K 增强 | 62% vs 纯 MCTS |
| v4 Attention (tiny) | 712K | 1.8M | 80% 纯先验 vs v3 |
| ResNet v4 | 819K | 1.8M | 46% (不及预期) |

**关键发现**: 小模型 (712K) + 多数据 > 大模型 (2.7M) + 少数据

## 编译

### C++ 引擎

```bash
cd src/cpp
g++ -shared -O2 -fPIC -o libgenerals.so generals_engine.cpp
```

### C++ 引擎 (含 LibTorch NN 推理)

```bash
cd src/cpp
g++ -std=c++17 -DUSE_TORCH -shared -O2 -fPIC \
    -o libgenerals_nn.so generals_engine.cpp \
    -I ../../libtorch/include \
    -I ../../libtorch/include/torch/csrc/api/include \
    -L ../../libtorch/lib \
    -Wl,--no-as-needed -ltorch -lc10 \
    -Wl,-rpath,../../libtorch/lib
```

## 图形化对战界面

```bash
cd src/python
python3 play_gui.py
```

操作说明:

| 按键 | 功能 |
|:---|:---|
| W/A/S/D 或 ↑↓←→ | 移动光标 |
| 鼠标左键 | 选中己方格子 / 确认移动 |
| 鼠标右键 | 移动半兵 |
| Enter | 选中/取消光标位置 |
| Space | 跳过回合 (初始 1 兵时使用) |
| ← / → | 复盘模式: 穿梭历史 |
| F | 切换迷雾/全图视野 |
| R | 重新开始 |
| Q | 退出 |

界面特性:
- 迷雾战争: 己方 + 相邻格子可见
- 城池占领变色: 红方红色 / 蓝方蓝色 / 中立黄色
- 地形标记: 城池=方块 / 将军=菱形
- AI 思考时间实时显示
- 自动检测最佳模型

## 规则一致性验证

```bash
cd src/python
python3 verify_rule_consistency.py
```

验证 6 项:
1. 同 seed 初始一致性
2. step_dual 只 tick 一次
3. 兵力增长时序 (每 25 步 bonus)
4. 合法动作掩码 (army<=1 无移动)
5. 将军攻占后变城池
6. 确定性先后手

## 技术细节

### 双方同时行动

游戏使用 `generals_step_dual()` 实现双方同时行动 + 一次 tick:

```
人类动作 + AI 动作 → step_dual() → tick() → 兵力增长
```

MCTS 推演使用交替回合制 (apply_action + tick per ply)，与真实游戏存在时序差异，但由于:
- MCTS 树浅 (主要依赖 NN 先验)
- 训练和部署使用相同的 step_dual 规则
- 兵力增长全局对称

NN 在训练中会吸收此差异，实际影响 <1 Elo。

### 数据增强

8 倍对称变换 (4 旋转 x 2 镜像)，不增加自对弈开销。

### 增量保存

每 50 局一批 npz 文件，避免 OOM (总数据可达 5GB+)。

## License

本项目为研究用途。
