"""
eval_mcts.py — 测试 IS-MCTS 对战随机 Bot

衡量：
  - 胜率
  - 平均步数
  - 斩首 vs 绝杀比例

用法:
  /usr/bin/python3 evaluation/eval_mcts.py [局数] [MCTS迭代数]
"""

import ctypes
import numpy as np
import time
import sys
import os

# 加载 C++ 引擎
_lib = ctypes.cdll.LoadLibrary(
    os.path.join(os.path.dirname(__file__), '..', 'src', 'cpp', 'libgenerals.so')
)

# ============================================================
# 类型定义
# ============================================================

_lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.generals_create.restype = ctypes.c_void_p
_lib.generals_destroy.argtypes = [ctypes.c_void_p]
_lib.generals_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.generals_step.restype = ctypes.c_int
_lib.generals_get_step.argtypes = [ctypes.c_void_p]
_lib.generals_get_step.restype = ctypes.c_int
_lib.generals_get_winner.argtypes = [ctypes.c_void_p]
_lib.generals_get_winner.restype = ctypes.c_int

_lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.belief_create.restype = ctypes.c_void_p
_lib.belief_destroy.argtypes = [ctypes.c_void_p]
_lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

_lib.mcts_create.argtypes = [ctypes.c_uint]
_lib.mcts_create.restype = ctypes.c_void_p
_lib.mcts_destroy.argtypes = [ctypes.c_void_p]
_lib.mcts_search.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.mcts_search.restype = ctypes.c_int


def run_one_game(seed, n_det=4, n_mcts=200, max_steps=300):
    """跑一局 IS-MCTS vs 随机 Bot，返回 (winner, steps, is_kill, search_time)"""
    game = _lib.generals_create(12, 12, max_steps, seed)
    belief = _lib.belief_create(12, 12, 0)
    mcts = _lib.mcts_create(seed + 999)

    skip = 12 * 12 * 8
    total_search = 0.0
    n_searches = 0

    done = False
    while not done:
        # IS-MCTS 思考
        t0 = time.perf_counter()
        action = _lib.mcts_search(mcts, belief, 0, n_det, n_mcts)
        total_search += time.perf_counter() - t0
        n_searches += 1

        # 执行
        winner = _lib.generals_step(game, action)

        # 更新信念
        step = _lib.generals_get_step(game)
        _lib.belief_observe(belief, game, step)

        done = winner != -1 or step >= max_steps

    w = _lib.generals_get_winner(game)
    steps = _lib.generals_get_step(game)

    _lib.mcts_destroy(mcts)
    _lib.belief_destroy(belief)
    _lib.generals_destroy(game)

    return w, steps, total_search, n_searches


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_det = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_mcts = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    MAX_STEPS = 300
    config = f"MCTS({n_det}宇宙×{n_mcts//n_det}迭代={n_mcts}总)"

    print("=" * 60)
    print("  IS-MCTS 强度测试")
    print(f"  配置: {config}")
    print(f"  对手: 随机 Bot")
    print(f"  地图: 12×12, 最多{MAX_STEPS}步")
    print(f"  局数: {n_games}")
    print("=" * 60)

    wins = 0
    kills = 0
    tiebreak_wins = 0
    draws = 0
    lengths = []
    search_times = []

    t_start = time.time()

    for g in range(n_games):
        w, steps, search_time, n_searches = run_one_game(g, n_det, n_mcts, MAX_STEPS)

        if w == 0:
            wins += 1
            # 粗略判断：如果步数 < max_steps 且 winner=0，大概率是斩首
            if steps < MAX_STEPS:
                kills += 1
            else:
                tiebreak_wins += 1
        elif w == -1:
            draws += 1

        lengths.append(steps)
        search_times.append(search_time)

        if (g + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{g+1:3d}/{n_games}] 胜率 {wins/(g+1)*100:.0f}%"
                  f"  搜索 {search_time/n_searches*1000:.1f}ms/步"
                  f"  总耗时 {elapsed:.0f}s")

    total_time = time.time() - t_start
    win_rate = wins / n_games * 100
    kill_rate = kills / n_games * 100
    avg_search = np.mean(search_times) / np.mean([1 for _ in range(n_games)])

    print()
    print("=" * 60)
    print("  结果")
    print("=" * 60)
    print(f"  🏆 胜率:       {win_rate:5.1f}% ({wins}/{n_games})")
    print(f"  ├─ ⚔️  斩首:   {kills:5d} 局 ({kill_rate:5.1f}%)")
    print(f"  ├─ 📊 绝杀:   {tiebreak_wins:5d} 局 ({tiebreak_wins/n_games*100:5.1f}%)")
    print(f"  └─ 🤝 平局:   {draws:5d} 局 ({draws/n_games*100:5.1f}%)")
    print(f"  平均步数:     {np.mean(lengths):5.0f}")
    print(f"  平均搜索:     {np.mean(search_times)/np.mean(lengths)*1000:.1f} ms/步")
    print(f"  总耗时:       {total_time:.1f}s ({total_time/n_games*1000:.0f}ms/局)")
    print("=" * 60)

    # 与 FCN 结果对比
    print()
    print("  对比 FCN 模型 (之前):")
    print(f"    FCN (V2):    24% 胜率 (3% 斩首)")
    print(f"    IS-MCTS:     {win_rate:.0f}% 胜率 ({kill_rate:.0f}% 斩首)")


if __name__ == '__main__':
    main()
