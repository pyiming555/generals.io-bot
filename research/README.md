# 竞品研究 (Generals.io AI Bots)

本目录收录和分析业界顶尖的 generals.io AI 项目，为我们的 RL 训练提供参考。

## 目录结构

```
research/
├── README.md              # 本文件
├── EklipZgit/             # EklipZgit/generals-bot (Human.exe)
│   └── notes.md           # 核心算法分析
├── strakam/               # strakam/generals-bots (JAX 训练框架)
│   └── notes.md           # RL 训练方法论
└── ideas.md               # 可借鉴的算法点子
```

---

## 1. EklipZgit/generals-bot (Human.exe)

**仓库**: https://github.com/EklipZgit/generals-bot
**Star**: ~34 | **语言**: Python | **类型**: 纯规则 AI (无 ML)
**作者**: Travis Drake (EklipZ), 基于 harrischristiansen/generals-bot 重写
**排名**: 人类 1v1 天梯 Top 10

### 核心算法

#### 1.1 最小生成树 (MST) 兵力收集
- 使用 MST 算法连接所有己方领地
- 修剪低价值分支
- 测试向高价值区域添加非 MST 路径
- **效果**: 最优兵力收集路径，避免浪费 tick

#### 1.2 DFS/BFS 扩张搜索
- 从大兵力格子出发，发起启发式 DFS/BFS 搜索
- 每回合重新计算（无状态）
- 攻击/扩张阶段分别处理

#### 1.3 迷雾预测 (Fog of War)
- 维护信念状态 (Belief State)
- 预测军队来源（城市/格子线）
- 更新被撤离城市的估计值为 1
- **效果**: 即使对方隐藏将军，也能精确定位

#### 1.4 咽喉点防御 (Chokepoint Defense)
- 使用图论识别咽喉点
- 将咽喉点格子特殊处理
- 允许延迟防御（咽喉点可以比开放路径晚一步防守）

### 可借鉴点
1. **MST 兵力收集** — 我们的 ScriptBot 完全用 BFS 距离场替代，但 MST 在收集效率上更优
2. **迷雾预测** — 我们的 BeliefState 已有贝叶斯滤波，但可以借鉴其"城市兵力估计"逻辑
3. **咽喉点识别** — 可以加入防守模式：识别关键位置，优先防守

---

## 2. strakam/generals-bots

**仓库**: https://github.com/strakam/generals-bots
**Star**: ~89 | **语言**: Python (JAX) | **类型**: RL 训练框架
**作者**: Matej Straka, Martin Schmid
**论文**: "Artificial Generals Intelligence: Mastering Generals.io with RL" (arXiv:2507.06825)
**成绩**: 单 H100 训练 36 小时，1v1 天梯 Top 0.003%

### 核心特性

#### 2.1 JAX 极速模拟器
- 10M+ steps/second
- 完全 JIT 编译
- vmap 向量化并行
- 支持数千局游戏同时运行

#### 2.2 纯函数式设计
- 不可变状态
- 可复现轨迹
- 兼容 Gymnasium / PettingZoo

#### 2.3 内置 Agent
- `RandomAgent`: 随机动作
- `ExpanderAgent`: 扩张型脚本 AI
- `Flobot`: 经典规则 AI (已被 RL 超越)

#### 2.4 奖励函数
- 默认: 赢 +1, 输 -1, 其他 0
- `FrequentAssetRewardFn`: 频繁奖励 (领地/兵力/城市变化)
- 自定义: 支持 potential-based reward shaping

#### 2.5 记忆特征
- 加入历史观测作为输入
- 帮助 agent 理解游戏节奏和对手策略

### 可借鉴点
1. **Potential-based Reward Shaping** — 我们的 RL reward 可以加入"领地变化"等频繁奖励
2. **并行训练** — 如果我们要扩大训练规模，JAX 是最佳选择
3. **ExpanderAgent** — 作为脚本 AI 的 baseline，可以参考其扩张逻辑

---

## 3. 可借鉴的算法点子

### 3.1 兵力收集优化 (来自 EklipZ)
```
当前问题: 我们的 ScriptBot 用 BFS 距离场, 但 BFS 只考虑距离, 不考虑收集效率
改进方案: 
  - 对每个己方领地格子, 计算"被收集优先级" = 周围空地数 / 距离
  - 优先从"高价值"格子调兵, 而不是最近的格子
```

### 3.2 咽喉点防守 (来自 EklipZ)
```
当前问题: 我们的防守雷达只检测"大股敌军", 不识别关键地形
改进方案:
  - BFS 计算从敌方可能的出生点到我方将军的所有路径
  - 找出"必经之路" (咽喉点)
  - 在咽喉点优先集结兵力
```

### 3.3 迷雾预测增强 (来自 EklipZ)
```
当前问题: 我们的 BeliefState 只推断将军位置, 不推断城市/兵力分布
改进方案:
  - 记录每个迷雾格子的"可能最小兵力"
  - 如果某格子长时间未被更新, 估计值降为 1
  - 攻击时优先选择"估计兵力最低"的目标
```

### 3.4 Reward Shaping (来自 strakam)
```
当前问题: 我们的 RL 只有终局奖励 (赢/输)
改进方案:
  - 加入每步的"领地变化"奖励
  - 加入"兵力增长"奖励
  - 加入"探索奖励" (点亮新格子)
  - 使用 potential-based shaping 保证最优策略不变
```

### 3.5 分兵策略 (来自观察)
```
当前问题: 我们的 ScriptBot 倾向于集中兵力, 不善于多线作战
改进方案:
  - 当主力军团 > 50 兵时, 分出 20% 兵力作为"斥候/骚扰队"
  - 斥候队执行: 探迷雾 / 骚扰敌方侧翼 / 占领侧翼城市
  - 主力队执行: 正面推进 / 防守咽喉点
```

### 3.6 攻击时机优化 (来自 EklipZ)
```
当前问题: 我们的 ScriptBot 只要 arm > target+1 就攻击
改进方案:
  - 考虑"兵力周期": 每 25 步有一次兵力增长
  - 如果我方即将获得增长, 而敌方不在攻击范围内, 等待增长后再攻击
  - 如果我方即将获得增长, 且可以卡点攻击, 优先攻击 (最大化伤害/周期)
```

---

## 4. 下一步行动建议

### 短期 (1-2天)
1. 将 EklipZ 的"咽喉点识别"加入我们的 ScriptBot 防守模式
2. 实现"分兵策略": 主力 + 斥候队

### 中期 (1周)
1. 用 strakam 的 JAX 模拟器加速我们的自对弈
2. 实现 reward shaping (领地变化 + 探索奖励)

### 长期 (2周+)
1. 训练 Cycle 4: 用新地图生成器 + 新 ScriptBot 陪练
2. 评估 v4_attention 在新环境下的胜率
