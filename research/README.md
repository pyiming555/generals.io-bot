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

## 3. harrischristiansen/generals-bot (原始版本)

**仓库**: https://github.com/harrischristiansen/generals-bot
**语言**: Python | **类型**: 规则 AI (客户端 Bot)
**作者**: Harris Christiansen

### 代码结构 (精简版)

```
generals-bot-harris/
├── base/
│   ├── bot_base.py       # 主循环, WebSocket 通信, 地图解析
│   ├── bot_moves.py      # 核心移动逻辑 (所有策略的底座)
│   └── viewer.py         # 可视化
├── bot_blob.py           # 策略1: 暴兵推土机
├── bot_path_collect.py   # 策略2: 路径收集 + 主攻
├── bot_control.py        # 策略3: 人类控制 (调试用)
└── startup.py            # 启动入口
```

### 核心算法解析

#### 3.1 `move_priority` — 优先级捕获 (将军/城市防守)
```python
# 遍历所有将军和城市邻居, 如果我方兵力 > 敌方兵力 + 1, 优先攻击
for tile in generals_and_cities:
    for neighbor in tile.neighbors():
        if neighbor.isSelf() and neighbor.army > tile.army + 1:
            return (neighbor, tile)  # 攻击!
```
**核心思想**: 将军和城市是最关键的防守点, 优先确保它们不被偷袭.

#### 3.2 `move_outward` — 向外扩张
```python
# 遍历所有己方格子, 找兵力 >= 2 的, 尝试攻击路径上的邻居
for source in gamemap.tiles[player_index]:
    if source.army >= 2:
        target = source.neighbor_to_attack(path)
        if target and not target.isSwamp:
            return (source, target)
```
**核心思想**: 只要邻居可以攻击(兵力足够), 就向外推进.

#### 3.3 `path_proximity_target` — 最近目标路径
```python
# 从最大兵力格子出发, 找最近的目标, 生成路径
source = gamemap.find_largest_tile(includeGeneral=0.5)
target = source.nearest_target_tile()
path = source.path_to(target)
```
**核心思想**: 大兵团不盲目移动, 始终朝向最近的目标.

#### 3.4 `_move_path_capture` — 路径兵力累积
```python
# 沿路径从终点向起点累积兵力
# 当累积兵力 > 0 时, 从该点向终点移动
for i, tile in reversed(list(enumerate(path))):
    if tile.tile == source.tile:
        capture_army += (tile.army - 1)  # 己方: 累加
    else:
        capture_army -= tile.army        # 敌方: 减去
    if capture_army > 0 and path[i].army > 1:
        return (path[i], path[i+1])  # 从这点出发能赢!
```
**核心思想**: 路径上的兵力可以"接力"累积, 即使单个格子打不过, 整条路径的兵力汇总后可能打穿!

#### 3.5 `should_move_half` — 半兵策略
```python
# 250 步后, 将军偶尔出半兵 (增加机动性)
if gamemap.turn > 250:
    if source.isGeneral:
        return random.choice([True, True, True, False])  # 75% 出半兵
    elif source.isCity and gamemap.turn - source.turn_captured < 16:
        return True  # 刚占领的城市出半兵扩张
```
**核心思想**: 后期将军需要快速机动, 半兵可以提高频率.

### 与我们的 V6 对比

| 特性 | Harris (原版) | 我们的 V6 |
|:--|:--|:--|
| **兵力收集** | `path_gather` (最大→第二大) | BFS 有向树 (多对一) |
| **主攻方向** | `path_proximity_target` (最近目标) | BFS 有向树 (全局梯度) |
| **路径累积** | `_move_path_capture` (兵力接力) | 无 (单步移动) |
| **防守** | `move_priority` (将军/城市优先) | 危机雷达 + 咽喉点 |
| **迷雾** | 无 | ghost_grid 残影追踪 |
| **扩张** | `move_outward` (任意方向) | 有向树引导 |

### 可借鉴的关键点
1. **`_move_path_capture` 兵力接力** — 我们的 V6 是单步有向树, 没有"整条路径汇总兵力"的概念
2. **`move_priority` 防守优先** — 将军/城市的防守优先级应该高于一切扩张
3. **`should_move_half` 后期半兵** — 250步后将军需要机动性

---

## 4. 可借鉴的算法点子

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
