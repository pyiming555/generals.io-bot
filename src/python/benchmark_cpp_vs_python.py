"""
benchmark_cpp_vs_python.py — C++ vs Python 推理速度严谨对比

测量维度:
  1. NN推理纯时间 (CPU forward)
  2. 全步NN+MCTS总时间
  3. 纯MCTS搜索时间 (流场)
  4. determinization时间

每次测 N 轮，输出: mean, std, min, p50, p95, p99, max
"""
import ctypes
import time
import os
import sys
import numpy as np

# ================================================================
# 加载库
# ================================================================
_PROJECT = os.path.dirname(os.path.abspath(__file__))
_LIB_CPP = os.path.join(_PROJECT, '..', 'cpp', 'libgenerals_nn.so')  # 含 NN
_LIB_PURE = os.path.join(_PROJECT, '..', 'cpp', 'libgenerals.so')    # 不含 NN

lib_nn = ctypes.cdll.LoadLibrary(_LIB_CPP)
lib_pure = ctypes.cdll.LoadLibrary(_LIB_PURE)

# --- 通用 C API 类型 ---
lib_nn.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib_nn.generals_create.restype = ctypes.c_void_p
lib_nn.generals_destroy.argtypes = [ctypes.c_void_p]
lib_nn.generals_destroy.restype = None
lib_nn.generals_get_winner.argtypes = [ctypes.c_void_p]
lib_nn.generals_get_winner.restype = ctypes.c_int
lib_nn.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib_nn.generals_get_obs.restype = None
lib_nn.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib_nn.generals_step_dual.restype = ctypes.c_int
lib_nn.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib_nn.belief_create.restype = ctypes.c_void_p
lib_nn.belief_destroy.argtypes = [ctypes.c_void_p]
lib_nn.belief_destroy.restype = None
lib_nn.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib_nn.belief_observe.restype = None
lib_nn.mcts_create.argtypes = [ctypes.c_uint]
lib_nn.mcts_create.restype = ctypes.c_void_p
lib_nn.mcts_destroy.argtypes = [ctypes.c_void_p]
lib_nn.mcts_destroy.restype = None
lib_nn.mcts_search_nn_auto.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib_nn.mcts_search_nn_auto.restype = ctypes.c_int
lib_nn.mcts_search_flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib_nn.mcts_search_flow.restype = ctypes.c_int
lib_nn.nn_create.argtypes = [ctypes.c_char_p]
lib_nn.nn_create.restype = ctypes.c_void_p
lib_nn.nn_destroy.argtypes = [ctypes.c_void_p]
lib_nn.nn_destroy.restype = None

# 纯引擎也注册通用的
lib_pure.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib_pure.generals_create.restype = ctypes.c_void_p
lib_pure.generals_destroy.argtypes = [ctypes.c_void_p]
lib_pure.generals_destroy.restype = None
lib_pure.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib_pure.generals_get_obs.restype = None
lib_pure.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib_pure.generals_step_dual.restype = ctypes.c_int
lib_pure.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib_pure.belief_create.restype = ctypes.c_void_p
lib_pure.belief_destroy.argtypes = [ctypes.c_void_p]
lib_pure.belief_destroy.restype = None
lib_pure.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib_pure.belief_observe.restype = None
lib_pure.mcts_create.argtypes = [ctypes.c_uint]
lib_pure.mcts_create.restype = ctypes.c_void_p
lib_pure.mcts_destroy.argtypes = [ctypes.c_void_p]
lib_pure.mcts_destroy.restype = None
lib_pure.mcts_search_flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib_pure.mcts_search_flow.restype = ctypes.c_int

# ================================================================
# 参数
# ================================================================
MAP_SIZE = 12
MAX_STEPS = 300
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1  # 1153
N_OBS_CH = 7
MODEL_DIR = os.path.join(_PROJECT, 'rl_models')
SEED = 42

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# Python 网络定义 (与 train_cycle2.py 一致)
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
# 准备: 创建固定场景 + 加载模型
# ================================================================
print("=" * 65)
print("  🏋️  准备阶段：创建场景 + 加载模型")
print("=" * 65)

# 创建一个固定的游戏状态 (用于可重复)
gs = lib_nn.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, SEED)
bs0 = lib_nn.belief_create(MAP_SIZE, MAP_SIZE, 0)
lib_nn.belief_observe(bs0, gs, 0)

# 提取观测
obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
lib_nn.generals_get_obs(gs, obs_buf, 0)
obs_np = np.frombuffer(obs_buf, dtype=np.float32).copy().reshape(1, N_OBS_CH, MAP_SIZE, MAP_SIZE)
obs_torch = torch.FloatTensor(obs_np)

# 加载模型 (Python)
device = torch.device('cpu')
net = PolicyValueNet()
model_path = os.path.join(MODEL_DIR, 'policy_value_v3.pt')
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()
print(f"  ✓ Python 模型: {model_path} ({sum(p.numel() for p in net.parameters()):,} params)")

# 加载模型 (C++ LibTorch)
cpp_model = os.path.join(MODEL_DIR, 'policy_value_v3.ptl')
nn_handle = lib_nn.nn_create(cpp_model.encode())
if not nn_handle:
    print("  ❌ C++ 模型加载失败!")
    sys.exit(1)
print(f"  ✓ C++  模型: {cpp_model}")

# 创建 MCTS 引擎
mcts_nn = lib_nn.mcts_create(SEED)
mcts_pure = lib_nn.mcts_create(SEED + 999)

print()


# ================================================================
# 统计工具
# ================================================================
def stats(times_ms, label):
    """打印统计摘要"""
    t = np.array(times_ms)
    print(f"  {label:28s}:  {np.mean(t):7.2f} ± {np.std(t):5.2f} ms")
    print(f"   {'':28s}  min={np.min(t):6.2f}  p50={np.median(t):6.2f}  p95={np.percentile(t,95):6.2f}  p99={np.percentile(t,99):6.2f}  max={np.max(t):6.2f}")
    print(f"   {'':28s}  N={len(t)}")
    return t


# ================================================================
# 基准 1: Python NN 纯前向推理
# ================================================================
print("=" * 65)
print("  🔬 基准 1: Python NN 推理 (纯 forward)")
print("=" * 65)

N_WARMUP = 50
N_BENCH = 2000
py_nn_times = []

# warmup
for _ in range(N_WARMUP):
    with torch.no_grad():
        p, v = net(obs_torch)

# benchmark
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    with torch.no_grad():
        p, v = net(obs_torch)
    t1 = time.perf_counter()
    py_nn_times.append((t1 - t0) * 1000)

t_py_nn = stats(py_nn_times, "Python NN forward")


# ================================================================
# 基准 2: C++ NN 纯前向推理 (通过 n_det=1, n_mcts=1 近似)
# ================================================================
print()
print("=" * 65)
print("  🔬 基准 2: C++ NN 推理 (n_det=1, n_mcts=1)")
print("=" * 65)

cpp_nn_times = []

# warmup
for _ in range(N_WARMUP):
    lib_nn.mcts_search_nn_auto(mcts_nn, bs0, 0, 1, 1, nn_handle)

# benchmark
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    lib_nn.mcts_search_nn_auto(mcts_nn, bs0, 0, 1, 1, nn_handle)
    t1 = time.perf_counter()
    cpp_nn_times.append((t1 - t0) * 1000)

t_cpp_nn = stats(cpp_nn_times, "C++ NN forward")


# ================================================================
# 基准 3: C++ 纯 determinization 开销 (n_mcts=1 但 nn已预热)
#   用无NN的 mcts_search_flow (n_det=1, n_mcts=1) 减去纯NN时间
# ================================================================
print()
print("=" * 65)
print("  🔬 基准 3: C++ determinization 开销")
print("=" * 65)

cpp_det_times = []

# warmup
for _ in range(N_WARMUP):
    lib_nn.mcts_search_flow(mcts_pure, bs0, 0, 1, 1)

for _ in range(N_BENCH):
    t0 = time.perf_counter()
    lib_nn.mcts_search_flow(mcts_pure, bs0, 0, 1, 1)
    t1 = time.perf_counter()
    cpp_det_times.append((t1 - t0) * 1000)

t_cpp_det = stats(cpp_det_times, "C++ determinize(n_det=1)")


# ================================================================
# 基准 4: C++ 全步 NN+MCTS (n_det=4, n_mcts=200)
# ================================================================
print()
print("=" * 65)
print("  🔬 基准 4: C++ 全步 NN+MCTS (n_det=4, n_mcts=200)")
print("=" * 65)

N_CPP_FULL = 200
cpp_full_times = []

for _ in range(N_WARMUP):
    lib_nn.mcts_search_nn_auto(mcts_nn, bs0, 0, 4, 200, nn_handle)

for _ in range(N_CPP_FULL):
    t0 = time.perf_counter()
    lib_nn.mcts_search_nn_auto(mcts_nn, bs0, 0, 4, 200, nn_handle)
    t1 = time.perf_counter()
    cpp_full_times.append((t1 - t0) * 1000)

t_cpp_full = stats(cpp_full_times, "C++ NN+MCTS (4/200)")


# ================================================================
# 基准 5: Python NN + C API MCTS (Python 典型流程)
#   模仿 evaluate_nn_vs_pure.py 的计时方式
# ================================================================
print()
print("=" * 65)
print("  🔬 基准 5: Python NN + C API MCTS (n_det=4, n_mcts=200)")
print("=" * 65)

N_PY_FULL = 200
py_full_times = []
py_nn_only_times_full = []

# 先设置 argtypes，再调用
lib_nn.mcts_search_nn.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
]
lib_nn.mcts_search_nn.restype = ctypes.c_int

for _ in range(N_WARMUP):
    with torch.no_grad():
        p_pred, v_pred = net(obs_torch)
    policy_arr = p_pred.cpu().numpy().flatten().astype(np.float32)
    policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    value_float = ctypes.c_float(float(v_pred.item()))
    lib_nn.mcts_search_nn(mcts_pure, bs0, 0, 4, 200, policy_ptr, value_float)

for _ in range(N_PY_FULL):
    with torch.no_grad():
        p_pred, v_pred = net(obs_torch)
    t0 = time.perf_counter()
    policy_arr = p_pred.cpu().numpy().flatten().astype(np.float32)
    policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    value_float = ctypes.c_float(float(v_pred.item()))
    act = lib_nn.mcts_search_nn(mcts_pure, bs0, 0, 4, 200, policy_ptr, value_float)
    t1 = time.perf_counter()
    py_full_times.append((t1 - t0) * 1000)

t_py_full = stats(py_full_times, "Py NN→C MCTS (4/200)")


# ================================================================
# 基准 6: C++ 纯 MCTS 流场搜索 (无 NN)
# ================================================================
print()
print("=" * 65)
print("  🔬 基准 6: C++ 纯MCTS流场 (n_det=4, n_mcts=200)")
print("=" * 65)

cpp_mcts_only_times = []

for _ in range(N_WARMUP):
    lib_nn.mcts_search_flow(mcts_pure, bs0, 0, 4, 200)

for _ in range(N_CPP_FULL):
    t0 = time.perf_counter()
    lib_nn.mcts_search_flow(mcts_pure, bs0, 0, 4, 200)
    t1 = time.perf_counter()
    cpp_mcts_only_times.append((t1 - t0) * 1000)

t_cpp_mcts_only = stats(cpp_mcts_only_times, "C++ 纯MCTS流场 (4/200)")


# ================================================================
# 总结
# ================================================================
print()
print("=" * 65)
print("  📊 最终对比汇总")
print("=" * 65)
print()
print(f"  {'组件':20s} {'Python':>14s} {'C++':>14s} {'比例':>10s}")
sep20 = '-' * 20
sep14 = '-' * 14
sep10 = '-' * 10
print(f"  {sep20:20s} {sep14:>14s} {sep14:>14s} {sep10:>10s}")

# NN纯推理
py_nn = np.mean(py_nn_times)
cpp_nn = np.mean(cpp_nn_times)
py_to_cpp_nn = py_nn / cpp_nn
print(f"  {'NN 推理':20s} {py_nn:>11.2f}ms {'':>3s} {cpp_nn:>11.2f}ms {py_to_cpp_nn:>8.2f}x")

# 全步
cpp_full = np.mean(cpp_full_times)
py_full = np.mean(py_full_times) if len(py_full_times) > 0 else 0.0
py_to_cpp_full = py_full / cpp_full if py_full > 0 else 0.0
print(f"  {'NN+MCTS (全步)':20s} {py_full:>11.2f}ms {'':>3s} {cpp_full:>11.2f}ms {py_to_cpp_full:>8.2f}x")

# MCTS纯搜索
cpp_mcts = np.mean(cpp_mcts_only_times)
print(f"  {'MCTS 流场搜索':20s} {'(C API)':>14s} {cpp_mcts:>11.2f}ms {'':>10s}")

print()
print("  🔑 关键结论:")
print(f"  1) NN推理纯速度: C++ 是 Python 的 {py_to_cpp_nn:.2f}x")
print(f"  2) 全步 NN+MCTS: C++ vs Python 流程")
if len(py_full_times) > 0:
    diff_pct = (py_full - cpp_full) / py_full * 100
    print(f"     C++ {'快' if diff_pct > 0 else '慢'}了 {abs(diff_pct):.1f}%")
print(f"  3) MCTS搜索占总时间 {cpp_mcts / cpp_full * 100:.0f}%")
print(f"  4) NN推理占总时间 {cpp_nn / cpp_full * 100:.0f}%")

# ================================================================
# 清理
# ================================================================
lib_nn.nn_destroy(nn_handle)
lib_nn.mcts_destroy(mcts_nn)
lib_nn.mcts_destroy(mcts_pure)
lib_nn.belief_destroy(bs0)
lib_nn.generals_destroy(gs)
print()
print("=" * 65)
print("  ✅ 清理完成")
print("=" * 65)
