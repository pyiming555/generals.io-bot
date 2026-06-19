"""Verify ResNet v4 Python vs C++ LibTorch inference match"""
import torch, numpy as np, ctypes, time, sys, os

workdir = '/media/pyiming/C22AA0E82AA0DB25/project/generals.io/src/python'
os.chdir(workdir)

# === Python 推理 ===
from train_resnet import ResNet

device = torch.device('cpu')
net = ResNet()
net.load_state_dict(torch.load('rl_models/resnet_v4.pt', map_location=device))
net.eval()

# 测试输入: 随机噪声
obs = np.random.randn(1, 7, 12, 12).astype(np.float32)
obs_t = torch.FloatTensor(obs)
with torch.no_grad():
    p_py, v_py = net(obs_t)
print(f"Python(随机): v={v_py.item():.6f}, p_sum={p_py.sum().item():.4f}")

# === C++ LibTorch 推理 ===
_lib_path = os.path.join('..', 'cpp', 'libgenerals_nn.so')
lib = ctypes.cdll.LoadLibrary(_lib_path)
lib.nn_create.argtypes = [ctypes.c_char_p]
lib.nn_create.restype = ctypes.c_void_p
lib.nn_destroy.argtypes = [ctypes.c_void_p]
lib.nn_destroy.restype = None

nn = lib.nn_create(b'rl_models/resnet_v4.ptl')
if not nn:
    print("❌ C++ 模型加载失败!")
    sys.exit(1)
print(f"✓ C++ 模型加载成功: {nn}")

# 创建游戏测试
lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.generals_create.restype = ctypes.c_void_p
lib.generals_destroy.argtypes = [ctypes.c_void_p]
lib.generals_destroy.restype = None
lib.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.generals_get_obs.restype = None
lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.belief_create.restype = ctypes.c_void_p
lib.belief_destroy.argtypes = [ctypes.c_void_p]
lib.belief_destroy.restype = None
lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.belief_observe.restype = None
lib.mcts_create.argtypes = [ctypes.c_uint]
lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]
lib.mcts_destroy.restype = None
lib.mcts_search_nn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_float]
lib.mcts_search_nn.restype = ctypes.c_int
lib.mcts_search_nn_auto.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.mcts_search_nn_auto.restype = ctypes.c_int

# 创建游戏
gs = lib.generals_create(12, 12, 300, 42)
bs = lib.belief_create(12, 12, 0)
lib.belief_observe(bs, gs, 0)

# 获取观测
obs_buf = (ctypes.c_float * (7 * 12 * 12))()
lib.generals_get_obs(gs, obs_buf, 0)
obs_arr = np.frombuffer(obs_buf, dtype=np.float32).copy().reshape(1, 7, 12, 12)

# Python 推理
with torch.no_grad():
    p_py2, v_py2 = net(torch.FloatTensor(obs_arr))
print(f"\n实际游戏状态:")
print(f"Python(真实): v={v_py2.item():.6f}, p_sum={p_py2.sum().item():.4f}")
top5 = torch.topk(p_py2[0], 5).indices.numpy()
print(f"  Policy top5动作ID: {top5}")

# C++ 自动推理 vs Python + MCTS (用相同seed确保确定性)
m_cpp = lib.mcts_create(42)
t0 = time.time()
act_cpp = lib.mcts_search_nn_auto(m_cpp, bs, 0, 1, 50, nn)
t1 = time.time()
time_cpp = (t1 - t0) * 1000

# Python NN + MCTS
policy_arr = p_py2[0].numpy().astype(np.float32)
v_val = float(v_py2.item())
policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
m_py = lib.mcts_create(42)
t2 = time.time()
act_py = lib.mcts_search_nn(m_py, bs, 0, 1, 50, policy_ptr, v_val)
t3 = time.time()
time_py = (t3 - t2) * 1000

print(f"\n结果对比:")
print(f"  C++ LibTorch MCTS: act={act_cpp}, {time_cpp:.1f}ms")
print(f"  Python NN + MCTS:  act={act_py},  {time_py:.1f}ms")
print(f"  动作一致? {'✅ 是' if act_cpp == act_py else '❌ 否'}")
if time_py > 0:
    print(f"  速度比: C++ 是 Python的 {time_py/time_cpp:.1f}x")

lib.mcts_destroy(m_cpp); lib.mcts_destroy(m_py)
lib.belief_destroy(bs); lib.generals_destroy(gs)
lib.nn_destroy(nn)
