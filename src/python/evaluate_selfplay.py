"""
evaluate_selfplay.py — MCTS随机Rollout vs MCTS流场Rollout 自对弈

用法: python evaluate_selfplay.py [--games 50] [--det 4] [--mcts 200]
"""
import ctypes
import time
import sys
import os

_lib_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'cpp', 'libgenerals.so'
)
lib = ctypes.cdll.LoadLibrary(_lib_path)

# --- 类型定义 ---
lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.generals_create.restype = ctypes.c_void_p
lib.generals_destroy.argtypes = [ctypes.c_void_p]
lib.generals_destroy.restype = None
lib.generals_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
lib.generals_reset.restype = None
lib.generals_get_winner.argtypes = [ctypes.c_void_p]
lib.generals_get_winner.restype = ctypes.c_int
lib.generals_get_step.argtypes = [ctypes.c_void_p]
lib.generals_get_step.restype = ctypes.c_int
lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.generals_step_dual.restype = ctypes.c_int
lib.generals_get_width.argtypes = [ctypes.c_void_p]
lib.generals_get_width.restype = ctypes.c_int
lib.generals_get_height.argtypes = [ctypes.c_void_p]
lib.generals_get_height.restype = ctypes.c_int

lib.mcts_create.argtypes = [ctypes.c_uint]
lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]
lib.mcts_destroy.restype = None
lib.mcts_search.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.mcts_search.restype = ctypes.c_int
lib.mcts_search_flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.mcts_search_flow.restype = ctypes.c_int

lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.belief_create.restype = ctypes.c_void_p
lib.belief_destroy.argtypes = [ctypes.c_void_p]
lib.belief_destroy.restype = None
lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.belief_observe.restype = None


def run_tournament(n_games=50, det=4, mcts_iter=200, max_steps=300, map_size=12):
    """
    每局跑两次（交换先后手），MCTS_random vs MCTS_flow。
    stats[0] = wins when random is player 0 (red)
    stats[1] = wins when flow is player 0 (red)
    """
    wins_random_p0 = 0  # random红时的 random胜场
    wins_flow_p0 = 0     # flow红时的 flow胜场
    draws = 0
    time_random = 0.0
    time_flow = 0.0
    total_steps = 0

    print(f"\nMCTS随机Rollout vs 流场Rollout 自对弈 ({n_games} 局×2 先后手)")
    print(f"参数: n_det={det}, n_mcts={mcts_iter}\n")

    for g in range(n_games):
        for swap in [0, 1]:
            # swap=0: 红=random, 蓝=flow
            # swap=1: 红=flow, 蓝=random
            seed = 42 + g * 13 + swap * 777
            use_flow_p0 = (swap == 1)
            use_flow_p1 = (swap == 0)

            gs = lib.generals_create(map_size, map_size, max_steps, seed)
            bs0 = lib.belief_create(map_size, map_size, 0)
            bs1 = lib.belief_create(map_size, map_size, 1)
            m0 = lib.mcts_create(seed)
            m1 = lib.mcts_create(seed + 999)

            step_count = 0
            winner = -1
            while step_count < max_steps:
                # 玩家 0 搜索
                lib.belief_observe(bs0, gs, step_count)
                t0 = time.time()
                if use_flow_p0:
                    act0 = lib.mcts_search_flow(m0, bs0, 0, det, mcts_iter)
                else:
                    act0 = lib.mcts_search(m0, bs0, 0, det, mcts_iter)
                t1 = time.time()
                if use_flow_p0: time_flow += (t1 - t0)
                else: time_random += (t1 - t0)

                # 玩家 1 搜索
                lib.belief_observe(bs1, gs, step_count)
                if use_flow_p1:
                    act1 = lib.mcts_search_flow(m1, bs1, 1, det, mcts_iter)
                else:
                    act1 = lib.mcts_search(m1, bs1, 1, det, mcts_iter)
                t2 = time.time()
                if use_flow_p1: time_flow += (t2 - t1)
                else: time_random += (t2 - t1)

                # 双动作步进
                winner = lib.generals_step_dual(gs, act0, act1)
                step_count += 1

                if winner != -1:
                    break

            total_steps += step_count

            # 计分
            if swap == 0:
                # random是红(0), flow是蓝(1)
                if winner == 0: wins_random_p0 += 1
                elif winner == 1: wins_flow_p0 += 1
                else: draws += 1
                label = "random(红) vs flow(蓝)"
            else:
                # flow是红(0), random是蓝(1)
                if winner == 0: wins_flow_p0 += 1
                elif winner == 1: wins_random_p0 += 1
                else: draws += 1
                label = "flow(红) vs random(蓝)"

            print(f"  [{g+1:2d}/{n_games}] {label}: 胜者={'random' if winner==0 and swap==0 or winner==1 and swap==1 else 'flow' if winner!=-1 and winner!=2 else '平局' if winner==2 else '超时'}, "
                  f"{step_count}步")

            lib.belief_destroy(bs0); lib.belief_destroy(bs1)
            lib.mcts_destroy(m0); lib.mcts_destroy(m1)
            lib.generals_destroy(gs)

    total = n_games * 2
    random_wins = wins_random_p0
    flow_wins = wins_flow_p0
    print(f"\n{'='*55}")
    print(f"🏆 最终结果")
    print(f"{'='*55}")
    print(f"  MCTS随机Rollout: {random_wins:3d} 胜 ({random_wins/total*100:.1f}%)")
    print(f"  MCTS流场 Rollout: {flow_wins:3d} 胜 ({flow_wins/total*100:.1f}%)")
    print(f"  平局: {draws}")
    print(f"  总对局: {total}")
    print(f"\n  每步搜索时间:")
    print(f"    随机Rollout: {time_random/max(1,total_steps)*1000:.1f}ms")
    print(f"    流场 Rollout: {time_flow/max(1,total_steps)*1000:.1f}ms")
    print(f"  平均局步数: {total_steps/total:.0f}")

    return random_wins, flow_wins, draws


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=30, help='对局数')
    parser.add_argument('--det', type=int, default=4, help='去迷雾化次数')
    parser.add_argument('--mcts', type=int, default=200, help='MCTS迭代次数')
    args = parser.parse_args()

    print(f"MCTS 随机Rollout vs 流场Rollout 自对弈锦标赛")
    t0 = time.time()
    r_wins, f_wins, draws = run_tournament(
        n_games=args.games, det=args.det, mcts_iter=args.mcts
    )
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
