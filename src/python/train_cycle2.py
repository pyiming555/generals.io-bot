"""
train_cycle2.py — Cycle 2 训练 v3 模型

用法: python train_cycle2.py [--steps 8000] [--batch 256] [--lr 1e-3]
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys

# ================================================================
# 参数
# ================================================================
MAP_SIZE = 12
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1  # 1153
N_OBS_CH = 7
DATA_DIR = os.path.join(os.path.dirname(__file__), 'rl_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'rl_models')
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# 网络 (与之前相同架构)
# ================================================================
class PolicyValueNet(nn.Module):
    def __init__(self, ch_in=7, h=12, w=12, n_actions=1153):
        super().__init__()
        self.h, self.w = h, w
        self.conv1 = nn.Conv2d(ch_in, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        conv_out = 64 * h * w
        self.fc = nn.Linear(conv_out, 256)
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x, mask=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        policy_logits = self.policy_head(x)
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)
        policy = F.softmax(policy_logits, dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value


# ================================================================
# 数据加载（增量方式防止 OOM）
# ================================================================
def load_data():
    """加载所有 cycle2_batch* 文件（预分配数组，避免 OOM）"""
    batch_files = sorted([f for f in os.listdir(DATA_DIR)
                          if f.startswith('cycle2_batch') and f.endswith('.npz')])
    if not batch_files:
        batch_files = sorted([f for f in os.listdir(DATA_DIR)
                              if 'cycle2' in f and 'aug8' in f and f.endswith('.npz')])
    if not batch_files:
        print("❌ 没有找到 Cycle 2 数据文件！")
        sys.exit(1)
    
    print(f"加载 {len(batch_files)} 个数据文件...")
    t0 = time.time()
    
    # 先计算总样本数（读第一行的 shape）
    total_samples = 0
    example_shape = None
    policy_shape = None
    for fname in batch_files:
        with np.load(os.path.join(DATA_DIR, fname)) as d:
            n = d['obs'].shape[0]
            total_samples += n
            if example_shape is None:
                example_shape = d['obs'].shape[1:]
                policy_shape = d['policy'].shape[1:]
    
    # 预分配大数组
    obs = np.empty((total_samples,) + example_shape, dtype=np.float32)
    policy = np.empty((total_samples,) + policy_shape, dtype=np.float32)
    value = np.empty(total_samples, dtype=np.float32)
    
    # 逐批复制
    offset = 0
    for fname in batch_files:
        fpath = os.path.join(DATA_DIR, fname)
        with np.load(fpath) as d:
            n = d['obs'].shape[0]
            obs[offset:offset+n] = d['obs']
            policy[offset:offset+n] = d['policy']
            value[offset:offset+n] = d['value']
            offset += n
    
    t1 = time.time()
    print(f"  加载完成: {total_samples:,} 样本, 耗时 {t1-t0:.1f}s")
    print(f"  内存: obs={obs.nbytes/1024**3:.2f}GB, policy={policy.nbytes/1024**3:.2f}GB")
    
    return obs, policy, value


# ================================================================
# 训练
# ================================================================
def train(n_steps=8000, batch_size=256, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"🚀 Cycle 2 训练 v3 模型")
    print(f"{'='*60}")
    print(f"参数: steps={n_steps}, batch={batch_size}, lr={lr}")
    
    # 加载数据
    obs, policy_target, value_target = load_data()
    n_samples = obs.shape[0]
    
    # 创建网络
    device = torch.device('cpu')
    net = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"网络参数量: {sum(p.numel() for p in net.parameters()):,}")
    print(f"训练样本: {n_samples:,}")
    print(f"批次: {batch_size}, 步数: {n_steps}")
    print()
    
    # 训练循环
    indices = np.arange(n_samples)
    best_loss = float('inf')
    t_start = time.time()
    
    for step in range(n_steps):
        # 随机采样
        batch_idx = np.random.choice(indices, batch_size, replace=False)
        
        obs_batch = torch.FloatTensor(obs[batch_idx]).to(device)
        policy_batch = torch.FloatTensor(policy_target[batch_idx]).to(device)
        value_batch = torch.FloatTensor(value_target[batch_idx]).to(device).unsqueeze(1)
        
        net.train()
        pred_policy, pred_value = net(obs_batch)
        
        # 损失: 策略交叉熵 + 价值 MSE
        policy_loss = -(policy_batch * torch.log(pred_policy + 1e-10)).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_value, value_batch)
        loss = policy_loss + value_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 200 == 0:
            # 策略准确率 (top-1)
            pred_actions = pred_policy.argmax(dim=1)
            true_actions = policy_batch.argmax(dim=1)
            accuracy = (pred_actions == true_actions).float().mean().item()
            
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed * 60  # 步/分
            
            print(f"  [{step+1:5d}/{n_steps}]  loss={loss:.4f}  "
                  f"p_loss={policy_loss:.4f}  v_loss={value_loss:.6f}  "
                  f"acc={accuracy:.3f}  {rate:.0f}步/分")
            
            # 保存最佳模型
            if loss < best_loss:
                best_loss = loss
                model_path = os.path.join(MODEL_DIR, 'policy_value_v3_best.pt')
                torch.save(net.state_dict(), model_path)
    
    # 保存最终模型
    model_path = os.path.join(MODEL_DIR, 'policy_value_v3.pt')
    torch.save(net.state_dict(), model_path)
    
    t_end = time.time()
    total_time = t_end - t_start
    
    print(f"\n{'='*60}")
    print(f"✅ Cycle 2 训练完成!")
    print(f"{'='*60}")
    print(f"  模型: {model_path}")
    print(f"  最佳 loss: {best_loss:.4f}")
    print(f"  耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  速度: {n_steps/total_time:.0f}步/s")
    
    return net


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=8000, help='训练步数')
    parser.add_argument('--batch', type=int, default=256, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    args = parser.parse_args()
    
    train(n_steps=args.steps, batch_size=args.batch, lr=args.lr)
