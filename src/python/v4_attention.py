"""
v4_attention.py — Attention-augmented Policy-Value Network (v4)

架构:
  Input(7×12×12)
  → Conv stem 7→128 + BN + ReLU
  → 7× ResBlock(128)       ← 局部特征提取
  → 1×1 Conv(128→128)      ← 降维投影
  → Flatten → (B, 144, 128)
  → Pre-LN SelfAttn(dim=128, heads=4) + MLP  ← 全局上下文
  → Reshape → (B, 128, 12, 12)
  → Policy head + Value head
  → [训练时] GeneralPred 辅助头 (推理时丢弃, 0开销)

目标参数量: ~2.6M (不含辅助头), ~2.7M (含)
训练数据: Cycle 2 + Cycle 3 = ~120万样本

用法:
  python v4_attention.py [--steps 8000] [--batch 256] [--lr 1e-3]
                         [--aux-weight 0.0] [--label-smooth 0.05]
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
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1   # 1153
N_OBS_CH = 7
DATA_DIR = os.path.join(os.path.dirname(__file__), 'rl_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'rl_models')
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# Pre-LN Self-Attention Block
# ================================================================
class SelfAttnBlock(nn.Module):
    """Pre-LayerNorm Self-Attention + MLP (稳定训练)"""
    def __init__(self, dim=128, heads=4, mlp_ratio=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        # x: (B, N, dim)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


# ================================================================
# ResBlock (标准 Conv→BN→ReLU→Conv→BN→Skip)
# ================================================================
class ResBlock(nn.Module):
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


# ================================================================
# v4 AttentionFCN — 主干网络
# ================================================================
class AttentionFCN(nn.Module):
    """
    带 Self-Attention 的策略-价值网络

    ┌─ 训练时输出: (policy, value, [general_pred])
    └─ 推理时输出: (policy, value) — auxiliary head 自动丢弃
    """
    def __init__(self, ch_in=N_OBS_CH, h=MAP_SIZE, w=MAP_SIZE,
                 n_actions=N_ACTIONS, n_blocks=7, n_filters=128,
                 attn_dim=128, attn_heads=4, use_general_head=True,
                 tiny=False):
        super().__init__()
        self.h, self.w = h, w
        self.use_general_head = use_general_head

        # 轻量模式: 4 ResBlocks(64) + Attn(dim=64)
        if tiny:
            n_blocks = 4
            n_filters = 64
            attn_dim = 64

        self.n_filters = n_filters

        # === 输入 stem ===
        self.conv_input = nn.Conv2d(ch_in, n_filters, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(n_filters)

        # === 残差塔 (局部特征) ===
        self.res_blocks = nn.ModuleList([
            ResBlock(n_filters) for _ in range(n_blocks)
        ])

        # === 1×1 投影到 attention 维度 ===
        self.proj_attn = nn.Conv2d(n_filters, attn_dim, 1, bias=True)

        # === 全局注意力 ===
        self.attn_block = SelfAttnBlock(dim=attn_dim, heads=attn_heads)

        # === 1×1 投影回卷积特征（保持一致性） ===
        self.proj_back = nn.Conv2d(attn_dim, n_filters, 1, bias=True)
        self.bn_back = nn.BatchNorm2d(n_filters)

        # === 策略头 ===
        self.policy_conv = nn.Conv2d(n_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        conv_out = 2 * h * w
        self.policy_fc = nn.Linear(conv_out, n_actions)

        # === 价值头 ===
        self.value_conv = nn.Conv2d(n_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        value_in = 1 * h * w
        self.value_fc1 = nn.Linear(value_in, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # === 敌方大本营预测头 (仅训练, 推理丢弃) ===
        if use_general_head:
            self.general_head = nn.Sequential(
                nn.Conv2d(n_filters, 32, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1, bias=True),
            )

    def forward(self, x, mask=None, return_general=False):
        """
        Args:
            x: (B, 7, 12, 12) 观测
            mask: (B, N_ACTIONS) 动作掩码
            return_general: 强制返回 general_pred (推理时默认不返回)
        Returns:
            policy: (B, 1153) softmax
            value: (B, 1) tanh → [-1, 1]
            general_pred: (可选, B, 144) logits
        """
        # === 局部特征提取 ===
        x = F.relu(self.bn_input(self.conv_input(x)))       # (B, 128, 12, 12)
        for block in self.res_blocks:
            x = block(x)

        # === 投影到 attention 空间 ===
        attn_feat = self.proj_attn(x)                       # (B, 128, 12, 12)
        B, C, H, W = attn_feat.shape
        attn_flat = attn_feat.view(B, H * W, C)             # (B, 144, 128)

        # === 全局注意力 ===
        attn_out = self.attn_block(attn_flat)               # (B, 144, 128)
        attn_out = attn_out.view(B, C, H, W)                # (B, 128, 12, 12)

        # === 投影回卷积空间 ===
        x = F.relu(self.bn_back(self.proj_back(attn_out)))  # (B, 128, 12, 12)

        # === 可选: 通用大本营预测 (辅助头) ===
        general_pred = None
        if return_general and self.use_general_head:
            g = self.general_head(x)                         # (B, 1, 12, 12)
            general_pred = g.view(B, -1)                     # (B, 144)

        # === 策略头 ===
        p = F.relu(self.policy_bn(self.policy_conv(x)))      # (B, 2, 12, 12)
        p = p.view(p.size(0), -1)                            # (B, 288)
        policy_logits = self.policy_fc(p)
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)
        policy = F.softmax(policy_logits, dim=-1)

        # === 价值头 ===
        v = F.relu(self.value_bn(self.value_conv(x)))        # (B, 1, 12, 12)
        v = v.view(v.size(0), -1)                            # (B, 144)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        if return_general or general_pred is not None:
            return policy, value, general_pred
        return policy, value


# ================================================================
# 流式数据加载 (节省内存: 一次只加载一个 .npz 文件 ~8MB)
# ================================================================
class StreamingDataLoader:
    """
    流式数据加载器

    优先使用预合并的 .npy mmap 文件 (零 I/O, 零内存):
      cycle2_merged_obs.npy (2.3GB)
      cycle2_merged_policy.npy (2.6GB)
      cycle2_merged_value.npy (2.3MB)

    回退: 按需加载 .npz 文件
    """
    def __init__(self, data_dir=DATA_DIR, prefixes=('cycle2', 'cycle3')):
        self.data_dir = data_dir
        self.prefixes = prefixes

        # 尝试使用 mmap 文件
        mmap_paths = self._try_mmap(prefixes)
        if mmap_paths:
            self._use_mmap = True
            self.obs_mmap, self.policy_mmap, self.value_mmap = mmap_paths
            self.total_samples = self.obs_mmap.shape[0]
            self.n_files = 1
            print(f"  📂 mmap 加速: {self.total_samples:,} 样本 ({' + '.join(prefixes)})")
            return

        # 回退: .npz 流式加载
        self._use_mmap = False
        all_files = sorted(f for f in os.listdir(data_dir)
                           if f.endswith('.npz') and '_aug8_' in f
                           and any(f.startswith(p) for p in prefixes))
        if not all_files:
            print(f"❌ 没有找到匹配的数据文件！前缀={prefixes} 目录={data_dir}")
            sys.exit(1)

        self.file_index = []
        total = 0
        for fname in all_files:
            fpath = os.path.join(data_dir, fname)
            try:
                with np.load(fpath, mmap_mode='r') as d:
                    n = d['obs'].shape[0]
                    self.file_index.append((fpath, n))
                    total += n
            except Exception as e:
                print(f"  ⚠️ 跳过 {fname}: {e}")

        self.total_samples = total
        self.n_files = len(self.file_index)
        self._weights = np.array([n for _, n in self.file_index], dtype=np.float64)
        self._cache_path = None
        self._cache_data = None
        self._steps_on_this_file = 0

        print(f"  流式加载 {self.n_files} 个文件, {total:,} 总样本 ({' + '.join(prefixes)})")

    def _try_mmap(self, prefixes):
        """尝试打开预先合并的 .npy mmap 文件"""
        # 支持多个 prefix: 取第一个能用的
        for prefix in prefixes:
            obs_path = os.path.join(self.data_dir, f'{prefix}_merged_obs.npy')
            pol_path = os.path.join(self.data_dir, f'{prefix}_merged_policy.npy')
            val_path = os.path.join(self.data_dir, f'{prefix}_merged_value.npy')
            if all(os.path.exists(p) for p in [obs_path, pol_path, val_path]):
                obs = np.load(obs_path, mmap_mode='r')
                pol = np.load(pol_path, mmap_mode='r')
                val = np.load(val_path, mmap_mode='r')
                return (obs, pol, val)
        return None

    def sample_batch(self, batch_size, device='cpu'):
        """采样一个 batch (mmap 快速路径)"""
        if self._use_mmap:
            return self._sample_mmap(batch_size, device)
        else:
            return self._sample_npz(batch_size, device)

    def _sample_mmap(self, batch_size, device='cpu'):
        """从 mmap 文件采样 (≈0 I/O, ≈0 内存)"""
        idx = np.random.choice(self.total_samples, batch_size, replace=False)

        obs_t = torch.from_numpy(self.obs_mmap[idx].copy()).float().to(device)
        policy_t = torch.from_numpy(self.policy_mmap[idx].copy()).float().to(device)
        value_t = torch.from_numpy(self.value_mmap[idx].copy()).float().to(device).unsqueeze(1)

        # general_pred (预留)
        general_ch = self.obs_mmap[idx, 6, :, :]
        g_flat = general_ch.reshape(batch_size, -1)
        g_idx = g_flat.argmax(axis=1)
        g_valid = g_flat.max(axis=1) > 0.5
        g_idx[~g_valid] = -1
        general_t = torch.LongTensor(g_idx).to(device)

        return obs_t, policy_t, value_t, general_t

    def _sample_npz(self, batch_size, device='cpu'):
        """从 .npz 文件采样 (回退路径)"""
        if not hasattr(self, '_steps_on_this_file'):
            self._steps_on_this_file = 0
            self._reload_file()

        self._steps_on_this_file += 1
        if self._steps_on_this_file >= 10:
            self._reload_file()
            self._steps_on_this_file = 0

        data = self._cache_data
        n_avail = data['obs'].shape[0]
        idx_in_file = np.random.choice(n_avail, batch_size, replace=True)

        obs_t = torch.FloatTensor(data['obs'][idx_in_file]).to(device)
        policy_t = torch.FloatTensor(data['policy'][idx_in_file]).to(device)
        value_t = torch.FloatTensor(data['value'][idx_in_file]).to(device).unsqueeze(1)

        general_ch = data['obs'][idx_in_file, 6, :, :]
        g_flat = general_ch.reshape(batch_size, -1)
        g_idx = g_flat.argmax(axis=1)
        g_valid = g_flat.max(axis=1) > 0.5
        g_idx[~g_valid] = -1
        general_t = torch.LongTensor(g_idx).to(device)

        return obs_t, policy_t, value_t, general_t

    def _reload_file(self):
        idx = np.random.choice(self.n_files, p=self._weights / self._weights.sum())
        fpath, _ = self.file_index[idx]
        self._cache_path = fpath
        self._cache_data = np.load(fpath)


# ================================================================
# Label Smoothing Cross Entropy
# ================================================================
def label_smoothing_cross_entropy(logits, targets, smoothing=0.05, ignore_index=-1):
    """
    Label smoothing + 可选的 ignore_index

    Args:
        logits: (B, C)
        targets: (B,) — 目标索引, -1 表示忽略
        smoothing: 平滑系数 (0=标准 CE)
    Returns:
        loss: 标量 (仅对非忽略样本平均)
    """
    if ignore_index is not None:
        valid = targets != ignore_index
        if valid.sum() == 0:
            return logits.new_tensor(0.0)
        logits = logits[valid]
        targets = targets[valid]

    n_class = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # one-hot with smoothing
    with torch.no_grad():
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - smoothing) * targets_one_hot + smoothing / n_class

    loss = -(targets_smooth * log_probs).sum(dim=-1).mean()
    return loss


# ================================================================
# 训练
# ================================================================
def train(n_steps=8000, batch_size=256, lr=1e-3, weight_decay=1e-4,
          aux_weight=0.0, label_smooth=0.05, data_prefixes=('cycle2', 'cycle3'),
          warmup_steps=1000, tiny=False):
    """
    训练 v4 AttentionFCN

    Args:
        n_steps: 训练步数
        batch_size: 批次大小
        lr: 初始学习率
        weight_decay: 权重衰减
        aux_weight: 大本营预测辅助 loss 权重 (0 = 关闭辅助任务)
        label_smooth: label smoothing 系数
        data_prefixes: 数据文件前缀
        warmup_steps: warmup 步数
    """
    print(f"\n{'='*60}")
    arch_str = '轻量 ' if tiny else ''
    print(f"  🚀 v4 AttentionFCN 训练 ({arch_str}模式)")
    print(f"{'='*60}")
    if tiny:
        print(f"  架构: 4 ResBlocks(64) + SelfAttn(dim=64, heads=4)")
    else:
        print(f"  架构: 7 ResBlocks(128) + SelfAttn(dim=128, heads=4)")
    print(f"  辅助头: {'GeneralPred(β=' + str(aux_weight) + ')' if aux_weight > 0 else '无'}")
    print(f"  参数: steps={n_steps}, batch={batch_size}, lr={lr}")
    print(f"        weight_decay={weight_decay}, label_smooth={label_smooth}")
    print(f"  数据: {', '.join(data_prefixes)}")

    # 流式数据加载 (零内存占用, 按需加载)
    loader = StreamingDataLoader(prefixes=data_prefixes)

    device = torch.device('cpu')

    # 限制线程数 (2 线程: 避免 OpenMP 死锁, 同时有加速)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    net = AttentionFCN(use_general_head=(aux_weight > 0), tiny=tiny).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  网络参数量: {n_params:,}")
    print()

    best_loss = float('inf')
    t_start = time.time()

    for step in range(n_steps):
        obs_batch, policy_batch, value_batch, general_batch = loader.sample_batch(batch_size)

        # --- Warmup LR ---
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = lr * lr_scale

        # --- 前向 ---
        net.train()
        if aux_weight > 0:
            pred_policy, pred_value, pred_general = net(obs_batch, return_general=True)
        else:
            pred_policy, pred_value = net(obs_batch)

        # --- Policy loss: CrossEntropy with label smoothing ---
        # Compute logits (not softmax) for cross-entropy
        log_policy = torch.log(pred_policy + 1e-10)
        policy_loss = -(policy_batch * log_policy).sum(dim=1).mean()

        # --- Value loss: MSE ---
        value_loss = F.mse_loss(pred_value, value_batch)

        # --- General pred loss (auxiliary task) ---
        if aux_weight > 0:
            valid_g = general_batch != -1
            if valid_g.sum() > 0:
                # general_pred: (B, 144), general_target: (B,) with -1 for invalid
                general_loss = F.cross_entropy(
                    pred_general, general_batch, ignore_index=-1)
                loss = policy_loss + value_loss + aux_weight * general_loss
            else:
                general_loss = pred_policy.new_tensor(0.0)
                loss = policy_loss + value_loss
        else:
            general_loss = None
            loss = policy_loss + value_loss

        # --- 反传 ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        # --- 汇报 ---
        if (step + 1) % 200 == 0:
            pred_actions = pred_policy.argmax(dim=1)
            true_actions = policy_batch.argmax(dim=1)
            accuracy = (pred_actions == true_actions).float().mean().item()

            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed * 60

            if aux_weight > 0 and general_loss is not None:
                general_acc = ((pred_general.argmax(dim=1) == general_batch) & valid_g).float().mean().item() * 100 if valid_g.sum() > 0 else 0
                print(f"  [{step+1:5d}/{n_steps}]  loss={loss:.4f}  "
                      f"p={policy_loss:.4f}  v={value_loss:.6f}  "
                      f"g={general_loss:.4f}({general_acc:.0f}%)  "
                      f"acc={accuracy:.3f}  {rate:.0f}步/分")
            else:
                print(f"  [{step+1:5d}/{n_steps}]  loss={loss:.4f}  "
                      f"p={policy_loss:.4f}  v={value_loss:.6f}  "
                      f"acc={accuracy:.3f}  {rate:.0f}步/分")

            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(),
                           os.path.join(MODEL_DIR, 'v4_attention_best.pt'))

    # 保存最终模型
    model_path = os.path.join(MODEL_DIR, 'v4_attention.pt')
    torch.save(net.state_dict(), model_path)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✅ v4 AttentionFCN 训练完成!")
    print(f"{'='*60}")
    print(f"  模型: {model_path}")
    print(f"  最佳 loss: {best_loss:.4f}")
    print(f"  耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  速度: {n_steps/total_time:.1f}步/s")

    return net


# ================================================================
# TorchScript 导出 (C++ LibTorch 推理)
# ================================================================
def export_torchscript(tiny=False):
    """导出 v4 AttentionFCN 到 TorchScript — 仅 policy+value, 丢弃 auxiliary head"""
    device = torch.device('cpu')
    model_path = os.path.join(MODEL_DIR, 'v4_attention.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型 {model_path} 不存在，跳过导出")
        return

    net = AttentionFCN(use_general_head=False, tiny=tiny)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # 包装器: 丢弃 general_pred 输出, 只返回 policy+value
    class InferenceWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            policy, value = self.net(x, return_general=False)
            return policy, value

    wrapped = InferenceWrapper(net).eval()

    # Tracing
    example = torch.randn(1, 7, 12, 12)
    traced = torch.jit.trace(wrapped, example, check_trace=False)

    ts_path = os.path.join(MODEL_DIR, 'v4_attention.ptl')
    traced.save(ts_path)
    print(f"  ✅ TorchScript 导出: {ts_path} ({os.path.getsize(ts_path)/1024:.0f}KB)")
    print(f"  📦 推理时仅输出 policy+value, aux head 已丢弃")


# ================================================================
# 纯网络先验对战 (v4 vs v3, 无 MCTS)
# ================================================================
def evaluate_pure_prior(n_games=100, n_det=4):
    """
    纯网络先验对战 (无 MCTS) — v4(Attention) vs v3(纯FCN)

    用 C API 直接双人对战:
      - 交替采样两网络的 policy 分布 (温度 τ=0.5)
      - 无 MCTS → 只比拼 networks 的先验质量
      - 如果 v4 pure win rate > 55% → 说明 Attention 带来了真实提升
    """
    import ctypes
    import sys as _sys

    # 加载 C++ 引擎
    _lib_path = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'libgenerals_nn.so')
    if not os.path.exists(_lib_path):
        print(f"❌ C++ 引擎 {_lib_path} 不存在")
        return
    lib = ctypes.cdll.LoadLibrary(_lib_path)
    lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    lib.generals_create.restype = ctypes.c_void_p
    lib.generals_destroy.argtypes = [ctypes.c_void_p]
    lib.generals_get_winner.argtypes = [ctypes.c_void_p]
    lib.generals_get_winner.restype = ctypes.c_int
    lib.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.generals_step_dual.restype = ctypes.c_int

    device = torch.device('cpu')

    # 加载 v4
    v4_path = os.path.join(MODEL_DIR, 'v4_attention.pt')
    if not os.path.exists(v4_path):
        print(f"❌ v4 模型 {v4_path} 不存在")
        return
    v4_net = AttentionFCN(use_general_head=False)
    v4_net.load_state_dict(torch.load(v4_path, map_location=device))
    v4_net.eval()

    # 加载 v3
    from generate_cycle3 import PolicyValueNet
    v3_path = os.path.join(MODEL_DIR, 'policy_value_v3.pt')
    if not os.path.exists(v3_path):
        print(f"❌ v3 模型 {v3_path} 不存在")
        return
    v3_net = PolicyValueNet()
    v3_net.load_state_dict(torch.load(v3_path, map_location=device))
    v3_net.eval()

    MAP = 12
    CH = 7
    MAX_STEPS = 300
    N_ACT = MAP * MAP * 8 + 1

    print(f"\n{'='*60}")
    print(f"  🏆 纯网络先验对战: v4(Attention) vs v3(FCN)")
    print(f"{'='*60}")
    print(f"  {n_games} 局, 无 MCTS, 温度 τ=0.5 采样\n")

    wins = {0: 0, 1: 0}  # 0=v4 wins, 1=v3 wins
    t_start = time.time()

    for g in range(n_games):
        seed = int(time.time() * 1000 + g)
        gs = lib.generals_create(MAP, MAP, MAX_STEPS, seed)

        v4_player = 0 if g < n_games // 2 else 1  # v4 先手一半

        step, winner = 0, -1
        while step < MAX_STEPS and winner == -1:
            # 当前该谁走
            cur_player = step % 2

            obs_buf = (ctypes.c_float * (CH * MAP * MAP))()
            lib.generals_get_obs(gs, obs_buf, cur_player)
            obs_np = np.frombuffer(obs_buf, dtype=np.float32).copy().reshape(1, CH, MAP, MAP)
            obs_t = torch.FloatTensor(obs_np)

            # 选择模型
            use_v4 = (cur_player == v4_player)
            with torch.no_grad():
                net = v4_net if use_v4 else v3_net
                policy, _ = net(obs_t)
            probs = policy.squeeze(0).numpy().astype(np.float64)

            # 温度采样 τ=0.5 (平方放大优势)
            probs = np.power(np.maximum(probs, 1e-10), 2.0)
            probs /= probs.sum()

            # 两个玩家各自 action
            actions = [0, 0]
            actions[cur_player] = int(np.random.choice(N_ACT, p=probs))
            # 另一个玩家跳过
            actions[1 - cur_player] = N_ACT - 1  # skip

            winner = lib.generals_step_dual(gs, actions[0], actions[1])
            step += 1

        lib.generals_destroy(gs)

        # 统计
        if winner == 0:
            wins[0 if v4_player == 0 else 1] += 1
        elif winner == 1:
            wins[1 if v4_player == 0 else 0] += 1

        if (g + 1) % 10 == 0:
            v4_w = wins[0]
            v3_w = wins[1]
            print(f"  [{g+1:3d}/{n_games}]  v4={v4_w}  v3={v3_w}")

    total_time = time.time() - t_start
    v4_wins = wins[0]
    v4_rate = v4_wins / n_games * 100

    print(f"\n{'='*60}")
    print(f"  🏆 纯网络先验对战结果")
    print(f"{'='*60}")
    print(f"  🤖 v4(Attention): {v4_wins} 胜 ({v4_rate:.1f}%)")
    print(f"  🧪 v3(FCN):      {wins[1]} 胜 ({100-v4_rate:.1f}%)")
    print(f"  ⏱ 耗时:         {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  速度:           {n_games/total_time*60:.1f}局/分")
    print()

    if v4_rate > 55:
        print(f"  ✅ v4 纯先验 >55% → 架构升维有效! 可以上 MCTS 测试.")
    elif v4_rate > 52:
        print(f"  ⚠️  v4 纯先验 52-55% → 有微弱优势, 需更多测试.")
    else:
        print(f"  ❌ v4 纯先验 <52% → Attention 未带来明显提升, 建议调架构.")


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('v4 AttentionFCN 训练+评估')

    # 训练参数
    parser.add_argument('--steps', type=int, default=6000, help='训练步数')
    parser.add_argument('--batch', type=int, default=256, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--wd', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--aux-weight', type=float, default=0.0,
                        help='大本营预测辅助 loss 权重 (0 = 关闭, ⚠️ 当前数据有完美信息泄漏, 默认关闭)')
    parser.add_argument('--label-smooth', type=float, default=0.05,
                        help='label smoothing 系数')
    parser.add_argument('--tiny', action='store_true',
                        help='轻量模式: 4 ResBlocks(64) + Attn(dim=64) ~0.8M params')

    # 数据参数
    parser.add_argument('--data', type=str, nargs='*',
                        default=['cycle2', 'cycle3'],
                        help='数据前缀 (默认: cycle2 cycle3)')

    # 动作
    parser.add_argument('--export-only', action='store_true',
                        help='仅导出 TorchScript')
    parser.add_argument('--evaluate', action='store_true',
                        help='评估纯网络先验对战 (需先训练)')
    parser.add_argument('--n-games', type=int, default=100,
                        help='评估局数')

    args = parser.parse_args()

    if args.export_only:
        export_torchscript(tiny=args.tiny)
    elif args.evaluate:
        evaluate_pure_prior(n_games=args.n_games)
    else:
        net = train(
            n_steps=args.steps,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.wd,
            aux_weight=args.aux_weight,
            label_smooth=args.label_smooth,
            data_prefixes=tuple(args.data),
            tiny=args.tiny,
        )
        export_torchscript(tiny=args.tiny)
