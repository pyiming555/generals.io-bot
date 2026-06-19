"""
generate_cycle3.py — Cycle 3 数据生成（ANALYSIS模式 + 探索噪声 + 对手池）

核心升级:
  1. Temperature 采样 — 前 15 步按 visit 分布采，非最大
  2. Dirichlet 噪声 — 根节点先验加噪，保证探索多样性
  3. 对手池 — v3, 纯MCTS, 脚本AI 混合对抗
  4. ANALYSIS 模式 — n_mcts=300，最高质量走法标签

用法: python generate_cycle3.py [--games 1000] [--model policy_value_v3]
"""
import ctypes
import time
import os
import sys
import numpy as np
import torch

# ================================================================
# C++ 引擎绑定 (libgenerals_nn.so — 带 BotMode C API)
# ================================================================
_lib_path = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'libgenerals_nn.so')
lib = ctypes.cdll.LoadLibrary(_lib_path)

lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.generals_create.restype = ctypes.c_void_p
lib.generals_destroy.argtypes = [ctypes.c_void_p]
lib.generals_destroy.restype = None
lib.generals_get_winner.argtypes = [ctypes.c_void_p]
lib.generals_get_winner.restype = ctypes.c_int
lib.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.generals_get_obs.restype = None
lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.generals_step_dual.restype = ctypes.c_int
lib.generals_skip_action.argtypes = [ctypes.c_void_p]
lib.generals_skip_action.restype = ctypes.c_int

lib.mcts_create.argtypes = [ctypes.c_uint]
lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]
lib.mcts_destroy.restype = None
lib.mcts_search_nn_with_policy.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
]
lib.mcts_search_nn_with_policy.restype = ctypes.c_int
lib.mcts_search_nn.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
]
lib.mcts_search_nn.restype = ctypes.c_int
lib.mcts_search_flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.mcts_search_flow.restype = ctypes.c_int

lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.belief_create.restype = ctypes.c_void_p
lib.belief_destroy.argtypes = [ctypes.c_void_p]
lib.belief_destroy.restype = None
lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.belief_observe.restype = None

# --- BotMode C API ---
lib.bot_get_default_mcts.argtypes = [ctypes.c_int]
lib.bot_get_default_mcts.restype = ctypes.c_int

# ================================================================
# 参数
# ================================================================
MAP_SIZE = 12
MAX_STEPS = 300
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1  # 1153
N_OBS_CH = 7
DATA_DIR = os.path.join(os.path.dirname(__file__), 'rl_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'rl_models')
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 50      # 每50局保存一批
TEMP_STEPS = 20      # 前20步用温度采样（渐进衰减）
TEMP_INIT = 1.0      # 初始温度
TEMP_FINAL = 0.1     # 最终温度（第TEMP_STEPS步）
DIRICHLET_ALPHA = 0.3   # Dirichlet α
DIRICHLET_EPS = 0.25    # 噪声占比 ϵ


# ================================================================
# Python 网络定义
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
    results = []
    rot_dir_map = [
        [3, 2, 1, 0],   # 旋转90°
        [1, 0, 3, 2],   # 旋转180°
        [2, 3, 0, 1],   # 旋转270°
    ]
    for rot_k in range(4):
        for flip_h in [False, True]:
            o = obs.copy()
            if flip_h:
                o = np.flip(o, axis=2).copy()
            o = np.rot90(o, rot_k, axes=(1, 2)).copy()

            p = np.zeros_like(policy)
            for aid in range(N_ACTIONS - 1):
                if policy[aid] <= 0:
                    continue
                tile = aid // 8
                r = tile // w
                c = tile % w
                d = (aid // 2) % 4
                half = aid % 2
                if flip_h:
                    c = w - 1 - c
                    if d == 2: d = 3
                    elif d == 3: d = 2
                if rot_k > 0:
                    for _ in range(rot_k):
                        r, c = c, h - 1 - r
                    d = rot_dir_map[rot_k - 1][d]
                new_tile = r * w + c
                new_aid = new_tile * 8 + d * 2 + half
                p[new_aid] = policy[aid]
            p[N_ACTIONS - 1] = policy[N_ACTIONS - 1]
            results.append((o, p))
    return results


# ================================================================
# NN 推理 + Dirichlet 噪声
# ================================================================
def nn_infer_noisy(net, obs_buf, map_size, n_obs_ch, add_noise=False):
    """NN推理，可选加 Dirichlet 噪声"""
    obs = torch.FloatTensor(np.frombuffer(obs_buf, dtype=np.float32)
                            .copy().reshape(1, n_obs_ch, map_size, map_size))
    with torch.no_grad():
        policy_pred, value_pred = net(obs)
    policy_arr = policy_pred.cpu().numpy().flatten().astype(np.float32)

    if add_noise:
        # Dirichlet(α) 噪声
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * N_ACTIONS).astype(np.float32)
        policy_arr = (1 - DIRICHLET_EPS) * policy_arr + DIRICHLET_EPS * noise
        # 重归一化
        policy_arr /= policy_arr.sum()

    value_float = float(value_pred.item())
    return policy_arr, value_float


# ================================================================
# Temperature 采样：从 visit 分布采样动作
# ================================================================
def get_temperature(step, temp_steps=TEMP_STEPS):
    """渐进衰减温度：τ从1.0线性降到0.1"""
    if step >= temp_steps:
        return 0.0
    ratio = step / max(1, temp_steps - 1)
    return TEMP_INIT - ratio * (TEMP_INIT - TEMP_FINAL)

def sample_with_temp(policy_out, temp):
    """按 visit 分布采样（温度 τ）"""
    probs = np.frombuffer(policy_out, dtype=np.float32).copy()
    if temp == 0:
        return int(np.argmax(probs))
    # 温度调整
    probs = np.power(np.maximum(probs, 1e-10), 1.0 / temp)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ================================================================
# 增量保存
# ================================================================
def save_batch(obs_list, policy_list, value_list, batch_idx, timestamp, prefix='cycle3'):
    if len(obs_list) == 0:
        return None
    obs_arr = np.array(obs_list, dtype=np.float32)
    policy_arr = np.array(policy_list, dtype=np.float32)
    value_arr = np.array(value_list, dtype=np.float32)
    fname = os.path.join(DATA_DIR, f'{prefix}_batch{batch_idx:03d}_aug8_{timestamp}.npz')
    np.savez_compressed(fname, obs=obs_arr, policy=policy_arr, value=value_arr)
    obs_list.clear()
    policy_list.clear()
    value_list.clear()
    return fname


# ================================================================
# 对手池
# ================================================================
OPPONENT_POOL = [
    # (name, type, n_mcts, weight)
    # type: 'nn' = NN+MCTS, 'flow' = 纯MCTS流场
    ('v3(n=300)', 'nn', 300, 800),    # 同等级对手
    ('v3(n=200)', 'nn', 200, 100),    # 弱一点
    ('纯MCTS流场', 'flow', 200, 100), # 完全不同打法
]

def pick_opponent(rng):
    """从对手池中随机选一个"""
    total_w = sum(w for _, _, _, w in OPPONENT_POOL)
    r = rng.randint(0, total_w)
    cum = 0
    for name, otype, n_mcts, w in OPPONENT_POOL:
        cum += w
        if r < cum:
            return name, otype, n_mcts
    return OPPONENT_POOL[-1][:3]


# ================================================================
# 数据收集：Cycle 3 自对弈
# ================================================================
def collect_cycle3(net, n_games=1000, det=4):
    """运行 Cycle 3 自对弈：温度 + Dirichlet + 对手池 + ANALYSIS"""
    print(f"\n{'='*65}")
    print(f"  🚀 Cycle 3: NN+MCTS 自对弈 (ANALYSIS n=300)")
    print(f"{'='*65}")
    print(f"  探索: 前{TEMP_STEPS}步温度采样, Dirichlet(α={DIRICHLET_ALPHA}, ϵ={DIRICHLET_EPS})")
    print(f"  对手池: {', '.join(f'{n}({w}局)' for n,_,_,w in OPPONENT_POOL)}")
    print(f"  增量保存: 每 {BATCH_SIZE} 局一批 + 8x增强\n")

    rng = np.random.RandomState(int(time.time()))

    obs_list, policy_list, value_list = [], [], []
    total_raw, total_aug = 0, 0
    saved_files, batch_counter = [], 0
    timestamp = int(time.time())
    t_start = time.time()

    # 对战统计
    opp_stats = {}

    for g in range(n_games):
        seed = int(rng.randint(0, 2**30))
        gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, seed)
        bs0 = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
        bs1 = lib.belief_create(MAP_SIZE, MAP_SIZE, 1)
        m0 = lib.mcts_create(seed + 12345)   # 与地图种子解耦
        m1 = lib.mcts_create(seed + 67890)

        # 选对手
        opp_name, opp_type, opp_n_mcts = pick_opponent(rng)
        opp_stats[opp_name] = opp_stats.get(opp_name, 0) + 1

        game_obs, game_policies = [], []
        step, winner = 0, -1

        while step < MAX_STEPS and winner == -1:
            # ======= 红方: NN+MCTS (v3, n=300, 有噪声) =======
            lib.belief_observe(bs0, gs, step)
            obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf, 0)

            # 前15步加 Dirichlet 噪声
            use_noise = (step < TEMP_STEPS)
            policy_arr, value_float = nn_infer_noisy(net, obs_buf, MAP_SIZE, N_OBS_CH, add_noise=use_noise)
            policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # MCTS 搜索 (n=300) + 输出 visit 分布
            policy_out = (ctypes.c_float * N_ACTIONS)()
            _ = lib.mcts_search_nn_with_policy(
                m0, bs0, 0, det, 300, policy_ptr, value_float, policy_out
            )

            # 温度采样：渐进衰减 τ=1.0→0.1（前20步）
            temp = get_temperature(step)
            if temp > 0:
                act0 = sample_with_temp(policy_out, temp)
            else:
                act0 = int(np.argmax(np.frombuffer(policy_out, dtype=np.float32)))

            # ======= 蓝方: 选对手 =======
            lib.belief_observe(bs1, gs, step)
            obs_buf1 = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf1, 1)

            if opp_type == 'nn':
                # NN+MCTS 对手
                policy_arr1, value_float1 = nn_infer_noisy(net, obs_buf1, MAP_SIZE, N_OBS_CH, add_noise=False)
                policy_ptr1 = policy_arr1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                act1 = lib.mcts_search_nn(m1, bs1, 1, det, opp_n_mcts, policy_ptr1, value_float1)
            elif opp_type == 'flow':
                # 纯MCTS流场对手
                act1 = lib.mcts_search_flow(m1, bs1, 1, det, opp_n_mcts)

            # ======= 保存红方观测 + MCTS visit 分布 =======
            obs_arr = np.frombuffer(obs_buf, dtype=np.float32).copy().reshape(N_OBS_CH, MAP_SIZE, MAP_SIZE)
            policy_arr_out = np.frombuffer(policy_out, dtype=np.float32).copy()
            game_obs.append(obs_arr)
            game_policies.append(policy_arr_out)

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
            if fname:
                saved_files.append(fname)
                batch_counter += 1
            elapsed = time.time() - t_start
            rate = (g + 1) / elapsed * 60
            # 回复给用户
            print(f"  [{g+1:4d}/{n_games}] ✅ 保存 | "
                  f"累计: {total_raw:,}原始→{total_aug:,}增强 | "
                  f"{rate:.1f}局/分 | 对手: {opp_name}", flush=True)

    # 最后一批
    if obs_list:
        fname = save_batch(obs_list, policy_list, value_list, batch_counter, timestamp)
        if fname:
            saved_files.append(fname)
            batch_counter += 1

    total_time = time.time() - t_start

    # 汇总
    print(f"\n{'='*65}")
    print(f"  ✅ Cycle 3 数据生成完成!")
    print(f"{'='*65}")
    print(f"  对战分布:")
    for name, cnt in sorted(opp_stats.items(), key=lambda x: -x[1]):
        print(f"    {name}: {cnt}局 ({cnt/n_games*100:.0f}%)")
    print(f"  原始样本:    {total_raw:,}")
    print(f"  8倍增强后:   {total_aug:,}")
    print(f"  批次文件:    {len(saved_files)} 个")
    print(f"  耗时:        {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  速度:        {total_raw/total_time:.0f}步/s")

    return saved_files


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cycle 3: 高质量自对弈数据生成')
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--model', type=str, default='policy_value_v3')
    parser.add_argument('--det', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cpu')
    net = PolicyValueNet()
    model_path = os.path.join(MODEL_DIR, f'{args.model}.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型 {model_path} 不存在！")
        sys.exit(1)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print(f"✓ 加载模型: {model_path} ({sum(p.numel() for p in net.parameters()):,} params)")

    collect_cycle3(net, n_games=args.games, det=args.det)
