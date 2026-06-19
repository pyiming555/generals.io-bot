"""
evaluate_v2_vs_v3.py — NN+MCTS(v2) vs NN+MCTS(v3) 锦标赛

用法: python evaluate_v2_vs_v3.py [--games 50] [--det 4] [--mcts 200]
"""
import ctypes
import time
import os
import sys
import numpy as np
import torch

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

lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.belief_create.restype = ctypes.c_void_p
lib.belief_destroy.argtypes = [ctypes.c_void_p]
lib.belief_destroy.restype = None
lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.belief_observe.restype = None

MAP_SIZE = 12
MAX_STEPS = 300
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1
N_OBS_CH = 7
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'rl_models')


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


def load_model(model_name):
    device = torch.device('cpu')
    net = PolicyValueNet()
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型 {model_path} 不存在！")
        sys.exit(1)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net


def nn_infer(net, obs_buf):
    obs = torch.FloatTensor(np.frombuffer(obs_buf, dtype=np.float32)
                            .copy().reshape(1, N_OBS_CH, MAP_SIZE, MAP_SIZE))
    with torch.no_grad():
        policy_pred, value_pred = net(obs)
    policy_arr = policy_pred.cpu().numpy().flatten().astype(np.float32)
    value_float = float(value_pred.item())
    return policy_arr, value_float


def run_tournament(net_red, net_blue, n_games=50, det=4, mcts_iter=200):
    """
    红方 = net_red (v3), 蓝方 = net_blue (v2)
    每局交换先后手
    """
    red_label = "v3" if net_red is not None else "v2"
    blue_label = "v2" if net_blue is not None else "v3"
    
    red_wins = 0  # v3 wins when red
    blue_wins = 0  # v2 wins when blue
    red_wins_swapped = 0  # v2 wins when red
    blue_wins_swapped = 0  # v3 wins when blue
    draws = 0
    
    nn_time_v3 = 0.0
    nn_time_v2 = 0.0
    nn_v3_steps = 0
    nn_v2_steps = 0

    print(f"\n{'='*60}")
    print(f"🏆 v3(红) vs v2(蓝) — {n_games} 局")
    print(f"{'='*60}")
    print(f"参数: n_det={det}, n_mcts={mcts_iter}")

    for g in range(n_games):
        seed = 42 + g * 13
        gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, seed)
        bs0 = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
        bs1 = lib.belief_create(MAP_SIZE, MAP_SIZE, 1)
        m0 = lib.mcts_create(seed)
        m1 = lib.mcts_create(seed + 999)

        step, winner = 0, -1
        while step < MAX_STEPS and winner == -1:
            # 红方 (v3)
            lib.belief_observe(bs0, gs, step)
            obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf, 0)
            t0 = time.time()
            policy_arr, value_float = nn_infer(net_red, obs_buf)
            t1 = time.time()
            nn_time_v3 += (t1 - t0)
            nn_v3_steps += 1
            policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            act0 = lib.mcts_search_nn(m0, bs0, 0, det, mcts_iter, policy_ptr, value_float)

            # 蓝方 (v2)
            lib.belief_observe(bs1, gs, step)
            obs_buf1 = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf1, 1)
            t2 = time.time()
            policy_arr1, value_float1 = nn_infer(net_blue, obs_buf1)
            t3 = time.time()
            nn_time_v2 += (t3 - t2)
            nn_v2_steps += 1
            policy_ptr1 = policy_arr1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            act1 = lib.mcts_search_nn(m1, bs1, 1, det, mcts_iter, policy_ptr1, value_float1)

            winner = lib.generals_step_dual(gs, act0, act1)
            step += 1

        if winner == 0: red_wins += 1
        elif winner == 1: blue_wins += 1
        else: draws += 1

        result = "v3胜" if winner == 0 else "v2胜" if winner == 1 else "平局"
        print(f"  [{g+1:2d}/{n_games}] v3(红) vs v2(蓝): {result}, {step}步")

        lib.belief_destroy(bs0); lib.belief_destroy(bs1)
        lib.mcts_destroy(m0); lib.mcts_destroy(m1)
        lib.generals_destroy(gs)

    # ====== 交换先后手 ======
    print(f"\n{'='*60}")
    print(f"🏆 v2(红) vs v3(蓝) — {n_games} 局（交换先后手）")
    print(f"{'='*60}")

    for g in range(n_games):
        seed = 42 + g * 13 + 777
        gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, seed)
        bs0 = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
        bs1 = lib.belief_create(MAP_SIZE, MAP_SIZE, 1)
        m0 = lib.mcts_create(seed)
        m1 = lib.mcts_create(seed + 999)

        step, winner = 0, -1
        while step < MAX_STEPS and winner == -1:
            # 红方 (v2)
            lib.belief_observe(bs0, gs, step)
            obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf, 0)
            t0 = time.time()
            policy_arr, value_float = nn_infer(net_blue, obs_buf)
            t1 = time.time()
            nn_time_v2 += (t1 - t0)
            nn_v2_steps += 1
            policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            act0 = lib.mcts_search_nn(m0, bs0, 0, det, mcts_iter, policy_ptr, value_float)

            # 蓝方 (v3)
            lib.belief_observe(bs1, gs, step)
            obs_buf1 = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
            lib.generals_get_obs(gs, obs_buf1, 1)
            t2 = time.time()
            policy_arr1, value_float1 = nn_infer(net_red, obs_buf1)
            t3 = time.time()
            nn_time_v3 += (t3 - t2)
            nn_v3_steps += 1
            policy_ptr1 = policy_arr1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            act1 = lib.mcts_search_nn(m1, bs1, 1, det, mcts_iter, policy_ptr1, value_float1)

            winner = lib.generals_step_dual(gs, act0, act1)
            step += 1

        if winner == 0: red_wins_swapped += 1  # v2 red wins
        elif winner == 1: blue_wins_swapped += 1  # v3 blue wins
        else: draws += 1

        result = "v2胜" if winner == 0 else "v3胜" if winner == 1 else "平局"
        print(f"  [{g+1:2d}/{n_games}] v2(红) vs v3(蓝): {result}, {step}步")

        lib.belief_destroy(bs0); lib.belief_destroy(bs1)
        lib.mcts_destroy(m0); lib.mcts_destroy(m1)
        lib.generals_destroy(gs)

    # ====== 汇总 ======
    v3_wins = red_wins + blue_wins_swapped  # v3 as red + v3 as blue
    v2_wins = blue_wins + red_wins_swapped  # v2 as red + v2 as blue
    total = n_games * 2

    print(f"\n{'='*60}")
    print(f"🏆 最终结果: v3 vs v2 直接对弈")
    print(f"{'='*60}")
    print(f"  {total} 总对局（含先后手交换）")
    print(f"")
    print(f"  🆕 v3: {v3_wins:3d} 胜 ({v3_wins/total*100:.1f}%)")
    print(f"  🧪 v2: {v2_wins:3d} 胜 ({v2_wins/total*100:.1f}%)")
    print(f"  🤝 平局: {draws}")
    print(f"")
    print(f"  📊 先后手分析:")
    print(f"     v3(红) vs v2(蓝): {red_wins}胜/{n_games}局 ({red_wins/n_games*100:.1f}%)")
    print(f"     v2(红) vs v3(蓝): {red_wins_swapped}胜/{n_games}局 ({red_wins_swapped/n_games*100:.1f}%)")
    print(f"")
    print(f"  ⏱ NN推理:")
    print(f"     v3: {nn_time_v3/max(1,nn_v3_steps)*1000:.2f}ms/步")
    print(f"     v2: {nn_time_v2/max(1,nn_v2_steps)*1000:.2f}ms/步")

    return v3_wins, v2_wins, draws


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=50, help='每颜色对局数')
    parser.add_argument('--det', type=int, default=4)
    parser.add_argument('--mcts', type=int, default=200)
    args = parser.parse_args()

    print("加载 v3 模型...")
    net_v3 = load_model('policy_value_v3')
    print(f"  参数量: {sum(p.numel() for p in net_v3.parameters()):,}")
    print("加载 v2 模型...")
    net_v2 = load_model('policy_value_v2')
    print(f"  参数量: {sum(p.numel() for p in net_v2.parameters()):,}")

    t0 = time.time()
    run_tournament(net_v3, net_v2, n_games=args.games, det=args.det, mcts_iter=args.mcts)
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
