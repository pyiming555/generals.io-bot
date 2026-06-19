"""
rl_pipeline.py — 流场MCTS + 强化学习 (Expert Iteration)

流程：
  1. MCTS 自对弈 → 收集 (7ch观测, 策略分布, 胜负) 数据
  2. 训练小型策略-价值网络
  3. 网络作为 MCTS 先验（后续迭代）

用法:
  python rl_pipeline.py --generate 200   # 生成200局自对弈数据
  python rl_pipeline.py --train 1000     # 训练1000步
  python rl_pipeline.py --evaluate 20    # 用网络评估 vs 脚本AI
"""
import ctypes
import time
import os
import sys
import numpy as np
from pathlib import Path

# ================================================================
# C++ 引擎绑定
# ================================================================
_lib_path = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'libgenerals.so')
lib = ctypes.cdll.LoadLibrary(_lib_path)

lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.generals_create.restype = ctypes.c_void_p
lib.generals_destroy.argtypes = [ctypes.c_void_p]
lib.generals_destroy.restype = None
lib.generals_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
lib.generals_reset.restype = None
lib.generals_get_winner.argtypes = [ctypes.c_void_p]
lib.generals_get_winner.restype = ctypes.c_int
lib.generals_get_step.argtypes = [ctypes.c_void_p]
lib.generals_get_step.restype = ctypes.c_int
lib.generals_get_width.argtypes = [ctypes.c_void_p]
lib.generals_get_width.restype = ctypes.c_int
lib.generals_get_height.argtypes = [ctypes.c_void_p]
lib.generals_get_height.restype = ctypes.c_int
lib.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.generals_get_obs.restype = None
lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.generals_step_dual.restype = ctypes.c_int

lib.mcts_create.argtypes = [ctypes.c_uint]
lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]
lib.mcts_destroy.restype = None
lib.mcts_search_flow_with_policy.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
lib.mcts_search_flow_with_policy.restype = ctypes.c_int

lib.mcts_search_flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.mcts_search_flow.restype = ctypes.c_int

lib.mcts_search_nn.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
]
lib.mcts_search_nn.restype = ctypes.c_int

lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.belief_create.restype = ctypes.c_void_p
lib.belief_destroy.argtypes = [ctypes.c_void_p]
lib.belief_destroy.restype = None
lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.belief_observe.restype = None

# ================================================================
# 参数
# ================================================================
MAP_SIZE = 12
MAX_STEPS = 300
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1  # 1153
N_OBS_CH = 7
DATA_DIR = Path(__file__).parent / 'rl_data'
MODEL_DIR = Path(__file__).parent / 'rl_models'


# ================================================================
# 数据收集：自对弈
# ================================================================
def collect_selfplay_data(n_games=50, det=4, mcts_iter=200):
    """运行自对弈，收集 (obs, policy, value) 训练数据"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_obs = []
    all_policies = []
    all_values = []
    
    for g in range(n_games):
        seed = 42 + g * 13
        # 红方 = flow MCTS (用 policy 输出), 蓝方 = flow MCTS (不用 policy)
        gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, seed)
        bs0 = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
        bs1 = lib.belief_create(MAP_SIZE, MAP_SIZE, 1)
        m0 = lib.mcts_create(seed)
        m1 = lib.mcts_create(seed + 999)
        
        game_obs = []
        game_policies = []
        
        step = 0
        winner = -1
        
        while step < MAX_STEPS and winner == -1:
            # 观测 + MCTS 搜索 (红方带 policy 输出)
            lib.belief_observe(bs0, gs, step)
            obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf, 0)
            
            policy_buf = (ctypes.c_float * N_ACTIONS)()
            act0 = lib.mcts_search_flow_with_policy(m0, bs0, 0, det, mcts_iter, policy_buf)
            
            # 蓝方搜索
            lib.belief_observe(bs1, gs, step)
            act1 = lib.mcts_search_flow(m1, bs1, 1, det, mcts_iter)
            
            # 保存 (obs, policy)
            obs_arr = np.frombuffer(obs_buf, dtype=np.float32).copy().reshape(N_OBS_CH, MAP_SIZE, MAP_SIZE)
            policy_arr = np.frombuffer(policy_buf, dtype=np.float32).copy()
            
            game_obs.append(obs_arr)
            game_policies.append(policy_arr)
            
            # 执行
            winner = lib.generals_step_dual(gs, act0, act1)
            step += 1
        
        # 价值标签: 红方胜=+1, 负=-1, 平=0
        value = 0.0
        if winner == 0:
            value = 1.0
        elif winner == 1:
            value = -1.0
        
        all_obs.extend(game_obs)
        all_policies.extend(game_policies)
        all_values.extend([value] * len(game_obs))
        
        lib.belief_destroy(bs0)
        lib.belief_destroy(bs1)
        lib.mcts_destroy(m0)
        lib.mcts_destroy(m1)
        lib.generals_destroy(gs)
        
        if (g + 1) % 10 == 0:
            print(f"  收集 {g+1}/{n_games} 局, {len(all_obs)} 样本")
    
    # 保存
    data = {
        'obs': np.array(all_obs, dtype=np.float32),
        'policy': np.array(all_policies, dtype=np.float32),
        'value': np.array(all_values, dtype=np.float32),
    }
    fname = DATA_DIR / f'selfplay_{n_games}games_{int(time.time())}.npz'
    np.savez_compressed(fname, **data)
    print(f"  保存 {fname} ({len(all_obs)} 样本)")
    return data


# ================================================================
# 策略-价值网络 (小CNN, CPU可用)
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """小型策略-价值网络
    
    输入: 7×12×12 观测
    输出: policy (1153), value (1)
    """
    def __init__(self, ch_in=7, h=12, w=12, n_actions=1153):
        super().__init__()
        self.h, self.w = h, w
        
        # 卷积主干
        self.conv1 = nn.Conv2d(ch_in, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 全连接
        conv_out = 64 * h * w
        self.fc = nn.Linear(conv_out, 256)
        
        # 策略头
        self.policy_head = nn.Linear(256, n_actions)
        
        # 价值头
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x, mask=None):
        # x: (B, 7, 12, 12)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        
        # 策略
        policy_logits = self.policy_head(x)
        if mask is not None:
            # 无效动作设为 -inf
            policy_logits = policy_logits.masked_fill(~mask, -1e9)
        policy = F.softmax(policy_logits, dim=-1)
        
        # 价值
        value = torch.tanh(self.value_head(x))
        
        return policy, value


# ================================================================
# 训练
# ================================================================
def train(n_steps=1000, batch_size=64, lr=1e-3, model_name='policy_value_v1'):
    """训练策略-价值网络"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    data_files = list(DATA_DIR.glob('selfplay_*.npz'))
    if not data_files:
        print("没有训练数据！先运行 --generate")
        return
    
    all_obs, all_policy, all_value = [], [], []
    for f in data_files:
        d = np.load(f)
        all_obs.append(d['obs'])
        all_policy.append(d['policy'])
        all_value.append(d['value'])
    
    obs = np.concatenate(all_obs, axis=0)
    policy_target = np.concatenate(all_policy, axis=0)
    value_target = np.concatenate(all_value, axis=0)
    
    print(f"加载 {obs.shape[0]} 样本 ({len(data_files)} 文件)")
    
    # 创建网络
    device = torch.device('cpu')
    net = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
    n_samples = obs.shape[0]
    indices = np.arange(n_samples)
    
    for step in range(n_steps):
        # 随机采样
        batch_idx = np.random.choice(indices, batch_size, replace=(n_samples < batch_size))
        
        obs_batch = torch.FloatTensor(obs[batch_idx]).to(device)
        policy_batch = torch.FloatTensor(policy_target[batch_idx]).to(device)
        value_batch = torch.FloatTensor(value_target[batch_idx]).to(device).unsqueeze(1)
        
        # 前向
        pred_policy, pred_value = net(obs_batch)
        
        # 损失: 策略交叉熵 + 价值 MSE
        policy_loss = -(policy_batch * torch.log(pred_policy + 1e-10)).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_value, value_batch)
        loss = policy_loss + value_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            # 计算策略准确率 (top-1)
            pred_actions = pred_policy.argmax(dim=1)
            true_actions = policy_batch.argmax(dim=1)
            accuracy = (pred_actions == true_actions).float().mean().item()
            
            print(f"  步骤 {step+1:4d}/{n_steps}  loss={loss.item():.4f}  "
                  f"p_loss={policy_loss.item():.4f}  v_loss={value_loss.item():.4f}  "
                  f"acc={accuracy:.3f}")
    
    # 保存
    model_path = MODEL_DIR / f'{model_name}.pt'
    torch.save(net.state_dict(), model_path)
    print(f"  模型保存: {model_path}")
    
    return net


# ================================================================
# 评估
# ================================================================
def evaluate(n_games=20, model_name='policy_value_v1'):
    """用网络 + MCTS 评估 vs 脚本AI"""
    device = torch.device('cpu')
    net = PolicyValueNet()
    model_path = MODEL_DIR / f'{model_name}.pt'
    if not model_path.exists():
        print(f"模型 {model_path} 不存在，使用纯 MCTS 评估")
        net = None
    else:
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        print(f"加载模型: {model_path}")
    
    for personality in [0, 1, 2]:
        name = ['A-扩张流', 'B-城市流', 'C-进攻流'][personality]
        wins = 0
        for g in range(n_games):
            gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, 42 + g * 7)
            bs = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
            m = lib.mcts_create(42 + g * 7 + 999)
            
            step, done, w = 0, False, -1
            while not done and step < MAX_STEPS:
                lib.belief_observe(bs, gs, step)
                act = lib.mcts_search_flow(m, bs, 0, 4, 200)
                w = lib.generals_script_step(gs, act, personality)
                step += 1
                if w != -1:
                    done = True
                    if w == 0:
                        wins += 1
            
            lib.belief_destroy(bs)
            lib.mcts_destroy(m)
            lib.generals_destroy(gs)
        
        print(f"  vs {name}: {wins}/{n_games} ({wins*100//n_games}%)")


def evaluate_nn(n_games=20, model_name='policy_value_v1', det=4, mcts_iter=200):
    """用 NN 先验 + MCTS 评估 vs 脚本AI"""
    device = torch.device('cpu')
    net = PolicyValueNet()
    model_path = MODEL_DIR / f'{model_name}.pt'
    if not model_path.exists():
        print(f"模型 {model_path} 不存在")
        return
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print(f"加载模型: {model_path}")
    
    nn_time = 0.0
    
    for personality in [0, 1, 2]:
        name = ['A-扩张流', 'B-城市流', 'C-进攻流'][personality]
        wins = 0
        for g in range(n_games):
            gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, 42 + g * 7)
            bs = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
            m = lib.mcts_create(42 + g * 7 + 999)
            
            step, done, w = 0, False, -1
            while not done and step < MAX_STEPS:
                lib.belief_observe(bs, gs, step)
                
                # 获取观测
                obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
                lib.generals_get_obs(gs, obs_buf, 0)
                obs = torch.FloatTensor(np.frombuffer(obs_buf, dtype=np.float32)
                                        .copy().reshape(1, N_OBS_CH, MAP_SIZE, MAP_SIZE))
                
                # NN 推理
                t0 = time.time()
                with torch.no_grad():
                    policy_pred, value_pred = net(obs)
                t1 = time.time()
                nn_time += (t1 - t0)
                
                # 传给 C API
                policy_arr = policy_pred.cpu().numpy().flatten().astype(np.float32)
                policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                value_float = float(value_pred.item())
                
                act = lib.mcts_search_nn(m, bs, 0, det, mcts_iter, policy_ptr, value_float)
                
                w = lib.generals_script_step(gs, act, personality)
                step += 1
                if w != -1:
                    done = True
                    if w == 0:
                        wins += 1
            
            lib.belief_destroy(bs)
            lib.mcts_destroy(m)
            lib.generals_destroy(gs)
        
        print(f"  NN-MCTS vs {name}: {wins}/{n_games} ({wins*100//n_games}%)")
    
    print(f"  NN 推理时间: {nn_time*1000/max(1,wins):.1f}ms/步")


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', type=int, default=0, help='生成N局自对弈数据')
    parser.add_argument('--train', type=int, default=0, help='训练N步')
    parser.add_argument('--evaluate', type=int, default=0, help='评估N局（纯MCTS）')
    parser.add_argument('--evaluate-nn', type=int, default=0, help='评估N局（NN + MCTS）')
    parser.add_argument('--det', type=int, default=4)
    parser.add_argument('--mcts', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--model', type=str, default='policy_value_v1')
    args = parser.parse_args()
    
    if args.generate:
        print(f"\n=== 生成 {args.generate} 局自对弈数据 ===")
        t0 = time.time()
        collect_selfplay_data(n_games=args.generate, det=args.det, mcts_iter=args.mcts)
        print(f"耗时: {time.time() - t0:.1f}s")
    
    if args.train:
        print(f"\n=== 训练 {args.train} 步 ===")
        t0 = time.time()
        train(n_steps=args.train, batch_size=args.batch, lr=args.lr, model_name=args.model)
        print(f"耗时: {time.time() - t0:.1f}s")
    
    if args.evaluate:
        print(f"\n=== 评估 {args.evaluate} 局（纯MCTS）===")
        evaluate(n_games=args.evaluate, model_name=args.model)
    
    if args.evaluate_nn:
        print(f"\n=== 评估 {args.evaluate_nn} 局（NN + MCTS）===")
        evaluate_nn(n_games=args.evaluate_nn, model_name=args.model, det=args.det, mcts_iter=args.mcts)
