"""
train_resnet.py — ResNet 训练 v4 模型

架构: 7→64 Conv → 6×ResBlock(64) → FC(256) → Policy+Value
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
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1
N_OBS_CH = 7
DATA_DIR = os.path.join(os.path.dirname(__file__), 'rl_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'rl_models')
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# ResNet 架构
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


class ResNet(nn.Module):
    """ResNet 策略-价值网络

    输入: 7×12×12 观测
    输出: policy (1153), value (1)
    """
    def __init__(self, ch_in=7, h=12, w=12, n_actions=1153, n_blocks=6, n_filters=128):
        super().__init__()
        self.h, self.w = h, w

        # 输入卷积
        self.conv_input = nn.Conv2d(ch_in, n_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(n_filters)

        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResBlock(n_filters) for _ in range(n_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(n_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        conv_out = 2 * h * w
        self.policy_fc = nn.Linear(conv_out, n_actions)

        # 价值头
        self.value_conv = nn.Conv2d(n_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        value_in = 1 * h * w
        self.value_fc1 = nn.Linear(value_in, 256)
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
# 数据加载
# ================================================================
def load_data():
    batch_files = sorted([f for f in os.listdir(DATA_DIR)
                          if f.startswith('cycle2_batch') and f.endswith('.npz')])
    if not batch_files:
        print("❌ 没有找到 Cycle 2 数据文件！")
        sys.exit(1)

    print(f"加载 {len(batch_files)} 个数据文件...")
    t0 = time.time()

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

    obs = np.empty((total_samples,) + example_shape, dtype=np.float32)
    policy = np.empty((total_samples,) + policy_shape, dtype=np.float32)
    value = np.empty(total_samples, dtype=np.float32)

    offset = 0
    for fname in batch_files:
        with np.load(os.path.join(DATA_DIR, fname)) as d:
            n = d['obs'].shape[0]
            obs[offset:offset+n] = d['obs']
            policy[offset:offset+n] = d['policy']
            value[offset:offset+n] = d['value']
            offset += n

    t1 = time.time()
    print(f"  加载完成: {total_samples:,} 样本, 耗时 {t1-t0:.1f}s")
    return obs, policy, value


# ================================================================
# 训练
# ================================================================
def train(n_steps=8000, batch_size=128, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"🚀 ResNet v4 训练")
    print(f"{'='*60}")
    print(f"架构: 6 ResBlocks(64filters), 参数量待计算")
    print(f"参数: steps={n_steps}, batch={batch_size}, lr={lr}")

    obs, policy_target, value_target = load_data()
    n_samples = obs.shape[0]

    device = torch.device('cpu')
    net = ResNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"网络参数量: {n_params:,}")
    print(f"训练样本: {n_samples:,}")
    print()

    indices = np.arange(n_samples)
    best_loss = float('inf')
    t_start = time.time()

    for step in range(n_steps):
        batch_idx = np.random.choice(indices, batch_size, replace=False)

        obs_batch = torch.FloatTensor(obs[batch_idx]).to(device)
        policy_batch = torch.FloatTensor(policy_target[batch_idx]).to(device)
        value_batch = torch.FloatTensor(value_target[batch_idx]).to(device).unsqueeze(1)

        net.train()
        pred_policy, pred_value = net(obs_batch)

        policy_loss = -(policy_batch * torch.log(pred_policy + 1e-10)).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_value, value_batch)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 200 == 0:
            pred_actions = pred_policy.argmax(dim=1)
            true_actions = policy_batch.argmax(dim=1)
            accuracy = (pred_actions == true_actions).float().mean().item()

            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed * 60

            print(f"  [{step+1:5d}/{n_steps}]  loss={loss:.4f}  "
                  f"p_loss={policy_loss:.4f}  v_loss={value_loss:.6f}  "
                  f"acc={accuracy:.3f}  {rate:.0f}步/分")

            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(),
                           os.path.join(MODEL_DIR, 'resnet_v4_best.pt'))

    model_path = os.path.join(MODEL_DIR, 'resnet_v4.pt')
    torch.save(net.state_dict(), model_path)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✅ ResNet v4 训练完成!")
    print(f"{'='*60}")
    print(f"  模型: {model_path}")
    print(f"  最佳 loss: {best_loss:.4f}")
    print(f"  耗时: {total_time:.0f}s ({total_time/60:.1f}min)")

    return net


# ================================================================
# TorchScript 导出
# ================================================================
def export_torchscript():
    """导出 ResNet 到 TorchScript 供 LibTorch C++ 使用"""
    device = torch.device('cpu')
    net = ResNet()
    model_path = os.path.join(MODEL_DIR, 'resnet_v4.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型 {model_path} 不存在，跳过导出")
        return

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # 用 tracing 导出
    example = torch.randn(1, 7, 12, 12)
    traced = torch.jit.trace(net, example)

    ts_path = os.path.join(MODEL_DIR, 'resnet_v4.ptl')
    traced.save(ts_path)
    print(f"✅ TorchScript 模型导出: {ts_path} ({os.path.getsize(ts_path)/1024:.0f}KB)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=6000, help='训练步数')
    parser.add_argument('--batch', type=int, default=128, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--export-only', action='store_true', help='仅导出 TorchScript')
    args = parser.parse_args()

    if args.export_only:
        export_torchscript()
    else:
        net = train(n_steps=args.steps, batch_size=args.batch, lr=args.lr)
        export_torchscript()
