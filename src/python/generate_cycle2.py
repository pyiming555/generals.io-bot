"""
generate_cycle2.py — Cycle 2 数据生成（增量保存版）

流程:
  1. 加载 v2 网络
  2. NN+MCTS vs NN+MCTS 自对弈（红方输出访问分布作为策略标签）
  3. 8倍对称增强 (4旋转 × 2镜像)
  4. 增量保存：每 N 局存一批，避免 OOM

用法: python generate_cycle2.py [--games 500] [--model policy_value_v2]
"""
import ctypes
import time
import os
import sys
import numpy as np
import torch

# ================================================================
# C++ 引擎绑定
# ================================================================
_lib_path = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'libgenerals.so')
lib = ctypes.cdll.LoadLibrary(_lib_path)

lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.generals_create.restype = ctypes.c_void_p
lib.generals_destroy.argtypes = [ctypes.c_void_p]
lib.generals_destroy.restype = None
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
lib.mcts_search_nn.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
]
lib.mcts_search_nn.restype = ctypes.c_int
lib.mcts_search_nn_with_policy.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
]
lib.mcts_search_nn_with_policy.restype = ctypes.c_int

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
DATA_DIR = os.path.join(os.path.dirname(__file__), 'rl_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'rl_models')

# 增量保存批大小（每批保存后清空内存）
BATCH_SIZE = 50  # 局


# ================================================================
# 策略-价值网络
# ================================================================
class PolicyValueNet(torch.nn.Module):
    def __init__(self, ch_in=7, h=12, w=12, n_actions=1153):
        super().__init__()
        self.h, self.w = h, w
        self.conv1 = torch.nn.Conv2d(ch_in, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        conv_out = 64 * h * w
        self.fc = torch.nn.Linear(conv_out, 256)
        self.policy_head = torch.nn.Linear(256, n_actions)
        self.value_head = torch.nn.Linear(256, 1)

    def forward(self, x, mask=None):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc(x))
        policy_logits = self.policy_head(x)
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)
        policy = torch.nn.functional.softmax(policy_logits, dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value


# ================================================================
# 8倍对称增强
# ================================================================
def augment_8fold(obs, policy, h=12, w=12):
    """
    对 (obs, policy) 进行 8 种对称变换。
    
    obs: (7, h, w) — 7通道棋盘特征
    policy: (n_actions,) — 1153维动作分布
    
    返回: list of (obs_aug, policy_aug)
    """
    results = []
    
    def rotate_obs(o, k):
        return np.rot90(o, k, axes=(1, 2)).copy()
    
    def flip_obs(o):
        return np.flip(o, axis=2).copy()
    
    # 方向映射 [上, 下, 左, 右]
    # 原始: 0=up(-1,0), 1=down(1,0), 2=left(0,-1), 3=right(0,1)
    # 旋转90°: up→right(3), right→down(1), down→left(2), left→up(0)
    # 旋转180°: up→down(1), down→up(0), left→right(3), right→left(2)
    # 旋转270°: up→left(2), left→down(1), down→right(3), right→up(0)
    rot_dir_map = [
        [3, 2, 1, 0],   # 旋转90°: 上→右, 下→左, 左→上, 右→下
        [1, 0, 3, 2],   # 旋转180°
        [2, 3, 0, 1],   # 旋转270°
    ]
    
    for rot_k in range(4):
        for flip_h in [False, True]:
            # 变换观察
            o = obs.copy()
            if flip_h:
                o = flip_obs(o)
            o = rotate_obs(o, rot_k)
            
            # 变换策略
            p = np.zeros_like(policy)
            for aid in range(N_ACTIONS - 1):  # 排除 wait
                if policy[aid] <= 0:
                    continue
                tile = aid // 8
                r = tile // w
                c = tile % w
                d = (aid // 2) % 4
                half = aid % 2
                
                # 水平镜像
                if flip_h:
                    c = w - 1 - c
                    if d == 2: d = 3
                    elif d == 3: d = 2
                
                # 旋转
                if rot_k > 0:
                    for _ in range(rot_k):
                        r, c = c, h - 1 - r
                    d = rot_dir_map[rot_k - 1][d]
                
                new_tile = r * w + c
                new_aid = new_tile * 8 + d * 2 + half
                p[new_aid] = policy[aid]
            
            # Wait 动作不变
            p[N_ACTIONS - 1] = policy[N_ACTIONS - 1]
            
            results.append((o, p))
    
    return results


# ================================================================
# NN 推理
# ================================================================
def nn_infer(net, obs_buf, map_size, n_obs_ch):
    """运行 NN 推理，返回 policy 和 value"""
    obs = torch.FloatTensor(np.frombuffer(obs_buf, dtype=np.float32)
                            .copy().reshape(1, n_obs_ch, map_size, map_size))
    with torch.no_grad():
        policy_pred, value_pred = net(obs)
    policy_arr = policy_pred.cpu().numpy().flatten().astype(np.float32)
    value_float = float(value_pred.item())
    return policy_arr, value_float


# ================================================================
# 增量保存
# ================================================================
def save_batch(obs_list, policy_list, value_list, batch_idx, timestamp):
    """保存一批数据并清空列表"""
    if len(obs_list) == 0:
        return
    
    obs_arr = np.array(obs_list, dtype=np.float32)
    policy_arr = np.array(policy_list, dtype=np.float32)
    value_arr = np.array(value_list, dtype=np.float32)
    
    fname = os.path.join(DATA_DIR, f'cycle2_batch{batch_idx:03d}_aug8_{timestamp}.npz')
    np.savez_compressed(fname,
                        obs=obs_arr,
                        policy=policy_arr,
                        value=value_arr)
    
    # 清空列表
    obs_list.clear()
    policy_list.clear()
    value_list.clear()
    
    return fname


# ================================================================
# 数据收集：NN+MCTS 自对弈（增量保存）
# ================================================================
def collect_nn_selfplay(net, n_games=500, det=4, mcts_iter=200):
    """运行 NN+MCTS vs NN+MCTS 自对弈，增量保存数据"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    obs_list = []
    policy_list = []
    value_list = []
    
    total_raw = 0
    total_aug = 0
    saved_files = []
    batch_counter = 0
    timestamp = int(time.time())
    
    print(f"\n{'='*60}")
    print(f"🚀 Cycle 2: NN+MCTS 自对弈 ({n_games} 局)")
    print(f"{'='*60}")
    print(f"参数: n_det={det}, n_mcts={mcts_iter}")
    print(f"增强: 8倍对称变换")
    print(f"增量保存: 每 {BATCH_SIZE} 局存一批\n")
    
    t_start = time.time()
    
    for g in range(n_games):
        seed = 42 + g * 13
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
            # ======= 红方: NN+MCTS + 输出策略 =======
            lib.belief_observe(bs0, gs, step)
            obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf, 0)
            
            policy_arr, value_float = nn_infer(net, obs_buf, MAP_SIZE, N_OBS_CH)
            policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            policy_out = (ctypes.c_float * N_ACTIONS)()
            act0 = lib.mcts_search_nn_with_policy(
                m0, bs0, 0, det, mcts_iter, policy_ptr, value_float, policy_out
            )
            
            # ======= 蓝方: NN+MCTS (不输出策略) =======
            lib.belief_observe(bs1, gs, step)
            obs_buf1 = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf1, 1)
            
            policy_arr1, value_float1 = nn_infer(net, obs_buf1, MAP_SIZE, N_OBS_CH)
            policy_ptr1 = policy_arr1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            act1 = lib.mcts_search_nn(m1, bs1, 1, det, mcts_iter, policy_ptr1, value_float1)
            
            # ======= 保存观测 + 策略 =======
            obs_arr = np.frombuffer(obs_buf, dtype=np.float32).copy().reshape(N_OBS_CH, MAP_SIZE, MAP_SIZE)
            policy_arr_out = np.frombuffer(policy_out, dtype=np.float32).copy()
            
            game_obs.append(obs_arr)
            game_policies.append(policy_arr_out)
            
            # ======= 执行 =======
            winner = lib.generals_step_dual(gs, act0, act1)
            step += 1
        
        # 价值标签
        if winner == 0:
            value = 1.0
        elif winner == 1:
            value = -1.0
        else:
            value = 0.0
        
        # 8倍增强
        game_raw = len(game_obs)
        for obs_arr, policy_arr in zip(game_obs, game_policies):
            for obs_aug, policy_aug in augment_8fold(obs_arr, policy_arr, MAP_SIZE, MAP_SIZE):
                obs_list.append(obs_aug)
                policy_list.append(policy_aug)
                value_list.append(value)
        
        total_raw += game_raw
        total_aug += game_raw * 8
        
        lib.belief_destroy(bs0); lib.belief_destroy(bs1)
        lib.mcts_destroy(m0); lib.mcts_destroy(m1)
        lib.generals_destroy(gs)
        
        # 每 BATCH_SIZE 局保存一批
        if (g + 1) % BATCH_SIZE == 0:
            fname = save_batch(obs_list, policy_list, value_list, batch_counter, timestamp)
            saved_files.append(fname)
            batch_counter += 1
            elapsed = time.time() - t_start
            rate = (g + 1) / elapsed * 60  # 局/分
            mem_kb = os.popen('grep VmRSS /proc/self/status 2>/dev/null').read()
            mem_str = f", {int(mem_kb.split()[1])//1024}MB" if mem_kb else ""
            print(f"  [{g+1:4d}/{n_games}] 保存 {fname}  |  "
                  f"累计: {total_raw:,}原始 → {total_aug:,}增强  |  "
                  f"{rate:.1f}局/分{mem_str}")
    
    # 最后一批
    if obs_list:
        fname = save_batch(obs_list, policy_list, value_list, batch_counter, timestamp)
        saved_files.append(fname)
        batch_counter += 1
    
    total_time = time.time() - t_start
    
    # 合并所有批次为一个文件
    print(f"\n{'='*60}")
    print(f"📦 合并所有批次 ({len(saved_files)} 文件)...")
    
    all_obs, all_policy, all_value = [], [], []
    for f in saved_files:
        d = np.load(f)
        all_obs.append(d['obs'])
        all_policy.append(d['policy'])
        all_value.append(d['value'])
    
    merged_obs = np.concatenate(all_obs, axis=0)
    merged_policy = np.concatenate(all_policy, axis=0)
    merged_value = np.concatenate(all_value, axis=0)
    
    merged_fname = os.path.join(DATA_DIR, f'cycle2_{n_games}games_aug8_merged_{timestamp}.npz')
    np.savez_compressed(merged_fname,
                        obs=merged_obs,
                        policy=merged_policy,
                        value=merged_value)
    
    # 删除批次文件（可选，保留也可）
    # for f in saved_files:
    #     os.remove(f)
    
    print(f"\n{'='*60}")
    print(f"✅ Cycle 2 数据生成完成!")
    print(f"{'='*60}")
    print(f"  原始样本:       {total_raw:,}")
    print(f"  8倍增强后:      {total_aug:,}")
    print(f"  批次文件:       {len(saved_files)} 个")
    print(f"  合并文件:       {merged_fname}")
    print(f"  耗时:           {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  速度:           {total_raw/total_time:.0f}步/s")
    
    return merged_fname


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=500, help='自对弈局数')
    parser.add_argument('--model', type=str, default='policy_value_v2', help='模型名称')
    parser.add_argument('--det', type=int, default=4, help='去迷雾化次数')
    parser.add_argument('--mcts', type=int, default=200, help='MCTS迭代次数')
    args = parser.parse_args()
    
    # 加载模型
    device = torch.device('cpu')
    net = PolicyValueNet()
    model_path = os.path.join(MODEL_DIR, f'{args.model}.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型 {model_path} 不存在！")
        sys.exit(1)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print(f"✓ 加载模型: {model_path}")
    print(f"  参数量: {sum(p.numel() for p in net.parameters()):,}")
    
    # 开始生成
    collect_nn_selfplay(net, n_games=args.games, det=args.det, mcts_iter=args.mcts)
