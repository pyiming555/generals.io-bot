"""
evaluate_nn_vs_pure.py — NN+MCTS vs 纯MCTS(流场) 自对弈

用法: python evaluate_nn_vs_pure.py [--games 50] [--det 4] [--mcts 200] [--model policy_value_v1]
"""
import ctypes
import time
import os
import sys
import numpy as np
import torch

# --- 路径设置 ---
_lib_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'cpp', 'libgenerals.so'
)
lib = ctypes.cdll.LoadLibrary(_lib_path)

# --- C API 类型 ---
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
lib.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.generals_get_obs.restype = None
lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.generals_step_dual.restype = ctypes.c_int

lib.mcts_create.argtypes = [ctypes.c_uint]
lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]
lib.mcts_destroy.restype = None
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

# --- 参数 ---
MAP_SIZE = 12
MAX_STEPS = 300
N_ACTIONS = MAP_SIZE * MAP_SIZE * 8 + 1  # 1153
N_OBS_CH = 7
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl_models')


# --- 网络定义 (必须匹配训练时的架构) ---
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


def load_model(model_name='policy_value_v1'):
    """加载训练好的网络"""
    device = torch.device('cpu')
    net = PolicyValueNet()
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型 {model_path} 不存在！")
        sys.exit(1)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print(f"✓ 加载模型: {model_path}")
    return net


def run_nn_vs_pure(net, use_nn_as_red=True, n_games=50, det=4, mcts_iter=200):
    """
    运行 NN+MCTS vs Pure MCTS 锦标赛

    use_nn_as_red=True  → (红=NN, 蓝=Pure)
    交换先后手时 use_nn_as_red=False → (红=Pure, 蓝=NN)
    """
    if use_nn_as_red:
        label = "NN+MCTS(红) vs 纯MCTS(蓝)"
    else:
        label = "纯MCTS(红) vs NN+MCTS(蓝)"

    nn_wins = 0
    pure_wins = 0
    draws = 0
    nn_total_time = 0.0
    pure_total_time = 0.0
    nn_steps = 0
    pure_steps = 0

    print(f"\n{'='*60}")
    print(f"🏆 {label}")
    print(f"{'='*60}")
    print(f"参数: n_det={det}, n_mcts={mcts_iter}, {n_games} 局")

    for g in range(n_games):
        seed = 42 + g * 13
        gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, seed)
        bs0 = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
        bs1 = lib.belief_create(MAP_SIZE, MAP_SIZE, 1)
        m0 = lib.mcts_create(seed)
        m1 = lib.mcts_create(seed + 999)

        step = 0
        winner = -1

        while step < MAX_STEPS and winner == -1:
            if use_nn_as_red:
                # 红 = NN+MCTS
                lib.belief_observe(bs0, gs, step)
                obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
                lib.generals_get_obs(gs, obs_buf, 0)
                obs = torch.FloatTensor(np.frombuffer(obs_buf, dtype=np.float32)
                                        .copy().reshape(1, N_OBS_CH, MAP_SIZE, MAP_SIZE))
                t0 = time.time()
                with torch.no_grad():
                    policy_pred, value_pred = net(obs)
                t1 = time.time()
                nn_total_time += (t1 - t0)
                nn_steps += 1

                policy_arr = policy_pred.cpu().numpy().flatten().astype(np.float32)
                policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                value_float = float(value_pred.item())
                act0 = lib.mcts_search_nn(m0, bs0, 0, det, mcts_iter, policy_ptr, value_float)

                # 蓝 = 纯MCTS(流场)
                lib.belief_observe(bs1, gs, step)
                t2 = time.time()
                act1 = lib.mcts_search_flow(m1, bs1, 1, det, mcts_iter)
                t3 = time.time()
                pure_total_time += (t3 - t2)
                pure_steps += 1
            else:
                # 红 = 纯MCTS(流场)
                lib.belief_observe(bs0, gs, step)
                t0 = time.time()
                act0 = lib.mcts_search_flow(m0, bs0, 0, det, mcts_iter)
                t1 = time.time()
                pure_total_time += (t1 - t0)
                pure_steps += 1

                # 蓝 = NN+MCTS
                lib.belief_observe(bs1, gs, step)
                obs_buf = (ctypes.c_float * (N_OBS_CH * MAP_SIZE * MAP_SIZE))()
                lib.generals_get_obs(gs, obs_buf, 1)
                obs = torch.FloatTensor(np.frombuffer(obs_buf, dtype=np.float32)
                                        .copy().reshape(1, N_OBS_CH, MAP_SIZE, MAP_SIZE))
                t2 = time.time()
                with torch.no_grad():
                    policy_pred, value_pred = net(obs)
                t3 = time.time()
                nn_total_time += (t3 - t2)
                nn_steps += 1

                policy_arr = policy_pred.cpu().numpy().flatten().astype(np.float32)
                policy_ptr = policy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                value_float = float(value_pred.item())
                act1 = lib.mcts_search_nn(m1, bs1, 1, det, mcts_iter, policy_ptr, value_float)

            winner = lib.generals_step_dual(gs, act0, act1)
            step += 1

        # 计分
        if use_nn_as_red:
            if winner == 0: nn_wins += 1
            elif winner == 1: pure_wins += 1
            else: draws += 1
        else:
            if winner == 0: pure_wins += 1
            elif winner == 1: nn_wins += 1
            else: draws += 1

        result_str = "NN胜" if (winner == 0 and use_nn_as_red) or (winner == 1 and not use_nn_as_red) else \
                     "纯MCTS胜" if (winner == 1 and use_nn_as_red) or (winner == 0 and not use_nn_as_red) else \
                     "平局" if winner == 2 else "超时"
        print(f"  [{g+1:2d}/{n_games}] {label}: {result_str}, {step}步")

        lib.belief_destroy(bs0); lib.belief_destroy(bs1)
        lib.mcts_destroy(m0); lib.mcts_destroy(m1)
        lib.generals_destroy(gs)

    return nn_wins, pure_wins, draws, nn_total_time, pure_total_time, nn_steps, pure_steps


def main(n_games=50, det=4, mcts_iter=200, model_name='policy_value_v1'):
    net = load_model(model_name)

    total_nn_wins = 0
    total_pure_wins = 0
    total_draws = 0
    total_nn_time = 0.0
    total_pure_time = 0.0
    total_nn_steps = 0
    total_pure_steps = 0

    # 第1轮: NN(红) vs 纯MCTS(蓝)
    nn0, pure0, draw0, nn_t0, pu_t0, nn_s0, pu_s0 = \
        run_nn_vs_pure(net, use_nn_as_red=True, n_games=n_games, det=det, mcts_iter=mcts_iter)
    total_nn_wins += nn0
    total_pure_wins += pure0
    total_draws += draw0
    total_nn_time += nn_t0
    total_pure_time += pu_t0
    total_nn_steps += nn_s0
    total_pure_steps += pu_s0

    # 第2轮: 纯MCTS(红) vs NN(蓝) — 交换先后手
    nn1, pure1, draw1, nn_t1, pu_t1, nn_s1, pu_s1 = \
        run_nn_vs_pure(net, use_nn_as_red=False, n_games=n_games, det=det, mcts_iter=mcts_iter)
    total_nn_wins += nn1
    total_pure_wins += pure1
    total_draws += draw1
    total_nn_time += nn_t1
    total_pure_time += pu_t1
    total_nn_steps += nn_s1
    total_pure_steps += pu_s1

    total_games = n_games * 2

    # === 最终结果 ===
    print(f"\n{'='*60}")
    print(f"🏆 最终结果: NN+MCTS vs 纯MCTS(流场)")
    print(f"{'='*60}")
    print(f"  {total_games} 总对局（含先后手交换）")
    print(f"")
    print(f"  🤖 NN+MCTS:  {total_nn_wins:3d} 胜 ({total_nn_wins/total_games*100:.1f}%)")
    print(f"  🧮 纯MCTS:   {total_pure_wins:3d} 胜 ({total_pure_wins/total_games*100:.1f}%)")
    print(f"  🤝 平局:     {total_draws}")
    print(f"")
    print(f"  ⏱ 平均每步搜索:")
    print(f"     NN+MCTS:   {total_nn_time/max(1,total_nn_steps)*1000:.1f}ms")
    print(f"     纯MCTS:    {total_pure_time/max(1,total_pure_steps)*1000:.1f}ms")
    print(f"")
    print(f"  📊 先后手分析:")
    nn_red_win_rate = nn0 / max(n_games, 1) * 100
    pure_red_win_rate = pure1 / max(n_games, 1) * 100
    print(f"     NN(红) vs 纯MCTS(蓝): {nn0}胜 / {n_games}局 ({nn_red_win_rate:.1f}%)")
    print(f"     纯MCTS(红) vs NN(蓝): {pure1}胜 / {n_games}局 ({pure_red_win_rate:.1f}%)")

    # 保存结果
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl_data', 'nn_vs_pure_result.txt')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        f.write(f"NN+MCTS vs 纯MCTS(流场) 锦标赛结果\n")
        f.write(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"参数: n_det={det}, n_mcts={mcts_iter}\n")
        f.write(f"对局: {total_games}（含先后手交换）\n")
        f.write(f"\nNN+MCTS: {total_nn_wins} 胜 ({total_nn_wins/total_games*100:.1f}%)\n")
        f.write(f"纯MCTS:  {total_pure_wins} 胜 ({total_pure_wins/total_games*100:.1f}%)\n")
        f.write(f"平局:     {total_draws}\n")
        f.write(f"\nNN(红) vs 纯MCTS(蓝): {nn0}胜/{n_games}局\n")
        f.write(f"纯MCTS(红) vs NN(蓝): {pure1}胜/{n_games}局\n")
    print(f"\n  结果已保存: {result_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=50, help='每副颜色对局数（总对局 = games * 2）')
    parser.add_argument('--det', type=int, default=4, help='去迷雾化次数')
    parser.add_argument('--mcts', type=int, default=200, help='MCTS迭代次数')
    parser.add_argument('--model', type=str, default='policy_value_v1', help='模型名称')
    args = parser.parse_args()

    t0 = time.time()
    main(n_games=args.games, det=args.det, mcts_iter=args.mcts, model_name=args.model)

    # 修正：将模型名称改为 cycle1
    # 但用户可能用的是 cycle1 还是 policy_value_v1？
    # 从训练记录看，train_1000games.py 保存的是 policy_value_v1
    # 但 rl_pipeline.py 的默认模型名也是 policy_value_v1
    # 我们先运行 v2（刚训练好的），如果失败再试 v1
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
