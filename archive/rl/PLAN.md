# 项目重建计划：模仿学习路线

## 核心理念
规则脚本生成数据 → FCN 学生学策略 → A* 执行步数

## 文件结构
```
project/generals.io/
├── script_agent.py           # 多性格脚本智能体（老师）
│   ├── BFS 寻路引擎
│   ├── 性格A: 疯狂扩张流
│   ├── 性格B: 憋兵城市流
│   └── 性格C: 暴力进攻流
├── generate_training_data.py  # 脚本对战 → 数据采集器
├── fcn_model.py               # 轻量全卷积网络（学生）
├── train_fcn.py               # 监督学习训练
├── inference.py               # 推理管线（FCN选点 + A*走路）
└── generals_gym_v6.py         # 环境封装
```

## 分步实施
- Step 1: script_agent.py（BFS + 多性格）
- Step 2: generals_gym_v6.py（环境+数据采集钩子）
- Step 3: generate_training_data.py（产生样本）
- Step 4: fcn_model.py（模型定义）
- Step 5: train_fcn.py（训练）
- Step 6: inference.py（线上推理）
