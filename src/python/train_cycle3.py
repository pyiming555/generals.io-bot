"""
train_cycle3.py — Cycle 3 训练脚本

架构选项:
  - FCN: 3Conv+FC (2.7M params) — 与 v3 相同
  - ResNet: 128f × 8~10 ResBlocks (3.2~3.8M params) — 主力候选

用法: python train_cycle3.py [--arch resnet] [--blocks 8] [--epochs 10000]
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
# ResNet 架构 (128 filters × N ResBlocks)
# ================================================================
class ResBlock(nn.Module):
    """残差块: Conv→BN→ReLU→Conv→BN→Skip+ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class ResNetPolicyValue(nn.Module):
    """
    ResNet 策略-价值网络 (128 filters)

    输入: 7×12×12 观测
    输出: policy (1153), value (1)

    参数量 (n_blocks=8): ~3.2M
    参数量 (n_blocks=10): ~3.8M
    """
    def __init__(self, ch_in=7, h=12, w=12, n_actions=1153, n_blocks=8, n_filters=128):
        super().__init__()
        self.h, self.w = h, w

        # 输入卷积: 7→128
        self.conv_input = nn.Conv2d(ch_in, n_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(n_filters)

        # 残差塔: N blocks × 128 filters
        self.res_blocks = nn.ModuleList([
            ResBlock(n_filters) for _ in range(n_blocks)
        ])

        # 策略头: 128→2→FC→1153
        self.policy_conv = nn.Conv2d(n_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * h * w, n_actions)

        # 价值头: 128→1→FC(256)→tanh
        self.value_conv = nn.Conv2d(n_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * h * w, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x, mask=None):
        # 输入卷积
        x = F.relu(self.bn_input(self.conv_input(x)))

        # 残差塔
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)
        policy = F.softmax(policy_logits, dim=-1)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value


# ================================================================
# FCN 架构 (与 v3 相同，作为对比基线)
# ================================================================
class FCNPolicyValue(nn.Module):
    """3Conv+FC, 2.7M params — Cycle 1/2 使用的架构"""
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
# 数据加载
# ================================================================
def load_cycle3_data():
    """加载 Cycle 3 数据文件 (cycle3_batch_*.npz)"""
    batch_files = sorted([f for f in os.listdir(DATA_DIR)
                          if f.startswith('cycle3_batch') and f.endswith('.npz')])
    if not batch_files:
        # 也尝试找已合并的文件
        batch_files = sorted([f for f in os.listdir(DATA_DIR)
                              if 'cycle3' in f and 'aug8' in f and f.endswith('.npz')])
    if not batch_files:
        print(f"❌ 没有找到 Cycle 3 数据文件！({DATA_DIR})")
        print(f"   请先运行: python generate_cycle3.py --games 1000")
        sys.exit(1)

    print(f"加载 {len(batch_files)} 个数据文件...")
    t0 = time.time()

    all_obs, all_policy, all_value = [], [], []
    for fname in batch_files:
        fp = os.path.join(DATA_DIR, fname)
        with np.load(fp) as d:
            n = d['obs'].shape[0]
            all_obs.append(d['obs'])
            all_policy.append(d['policy'])
            all_value.append(d['value'])
        print(f"  {fname}: {n:,} 样本")

    obs = np.concatenate(all_obs, axis=0)
    policy = np.concatenate(all_policy, axis=0)
    value = np.concatenate(all_value, axis=0)

    print(f"  总计: {obs.shape[0]:,} 样本 ({len(batch_files)} 文件)")
    print(f"  耗时: {time.time()-t0:.1f}s")
    return obs, policy, value


# ================================================================
# 训练
# ================================================================
def train(model, obs, policy_target, value_target,
          n_epochs=10000, batch_size=512, lr=3e-4, save_prefix='resnet_v5'):
    """策略-价值联合训练"""
    device = torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = obs.shape[0]
    indices = np.arange(n)

    obs_t = torch.FloatTensor(obs)
    policy_t = torch.FloatTensor(policy_target)
    value_t = torch.FloatTensor(value_target).unsqueeze(1)

    best_loss = float('inf')
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"🚀 训练: {save_prefix}")
    print(f"{'='*60}")
    print(f"  样本:     {n:,}")
    print(f"  架构:     {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Epochs:   {n_epochs}")
    print(f"  Batch:    {batch_size}")
    print(f"  LR:       {lr}")
    print(f"  Device:   {device}")
    print()

    step = 0
    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(indices)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            obs_b = obs_t[batch_idx]
            policy_b = policy_t[batch_idx]
            value_b = value_t[batch_idx]

            optimizer.zero_grad()
            policy_pred, value_pred = model(obs_b)

            # 策略损失: KL散度 (交叉熵)
            policy_loss = -(policy_b * torch.log(policy_pred + 1e-10)).sum(dim=1).mean()
            # 价值损失: MSE
            value_loss = F.mse_loss(value_pred, value_b)
            # 总损失
            total_loss = policy_loss + value_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1
            step += 1

        avg_policy = epoch_policy_loss / n_batches
        avg_value = epoch_value_loss / n_batches
        avg_total = epoch_total_loss / n_batches

        if epoch % 500 == 0 or epoch == 1 or avg_total < best_loss:
            elapsed = time.time() - t_start
            speed = step / elapsed
            print(f"  Epoch {epoch:5d}/{n_epochs}: "
                  f"loss={avg_total:.4f}  "
                  f"policy={avg_policy:.4f}  "
                  f"value={avg_value:.6f}  "
                  f"{speed:.0f}步/s")

            if avg_total < best_loss:
                best_loss = avg_total
                model_path = os.path.join(MODEL_DIR, f'{save_prefix}_best.pt')
                torch.save(model.state_dict(), model_path)
                print(f"    → 新最佳模型 saved: {model_path}")

    # 保存最终模型
    final_path = os.path.join(MODEL_DIR, f'{save_prefix}.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ 训练完成!")
    print(f"  最佳模型: {save_prefix}_best.pt (loss={best_loss:.4f})")
    print(f"  最终模型: {save_prefix}.pt")
    print(f"  总耗时:   {time.time()-t_start:.0f}s ({ (time.time()-t_start)/60:.1f}min)")

    return model


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cycle 3 训练')
    parser.add_argument('--arch', type=str, default='resnet', choices=['fcn', 'resnet'],
                        help='网络架构: fcn(2.7M) 或 resnet(3.2~3.8M)')
    parser.add_argument('--blocks', type=int, default=8,
                        help='ResNet残差块数 (仅resnet, 8~10)')
    parser.add_argument('--filters', type=int, default=128,
                        help='ResNet卷积核数 (仅resnet)')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    # 加载数据
    obs, policy, value = load_cycle3_data()

    # 创建模型
    if args.arch == 'fcn':
        model = FCNPolicyValue()
        save_prefix = 'fcn_v4'
    else:
        model = ResNetPolicyValue(n_blocks=args.blocks, n_filters=args.filters)
        save_prefix = f'resnet_v5_{args.filters}f_{args.blocks}b'

    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    train(model, obs, policy, value,
          n_epochs=args.epochs,
          batch_size=args.batch,
          lr=args.lr,
          save_prefix=save_prefix)
