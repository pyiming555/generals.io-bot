"""
evaluate_vs_script.py — IS-MCTS vs 三种性格脚本 AI 锦标赛

用法: python evaluate_vs_script.py [--games 50] [--seed 42]
"""
import ctypes
import time
import sys
import os

# 加载共享库
_lib_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'cpp', 'libgenerals.so'
)
lib = ctypes.cdll.LoadLibrary(_lib_path)

# 类型
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
lib.generals_is_stalemate.argtypes = [ctypes.c_void_p]
lib.generals_is_stalemate.restype = ctypes.c_bool
lib.generals_get_width.argtypes = [ctypes.c_void_p]
lib.generals_get_width.restype = ctypes.c_int
lib.generals_get_height.argtypes = [ctypes.c_void_p]
lib.generals_get_height.restype = ctypes.c_int
lib.generals_skip_action.argtypes = [ctypes.c_void_p]
lib.generals_skip_action.restype = ctypes.c_int

# MCTS API
lib.mcts_create.argtypes = [ctypes.c_uint]
lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]
lib.mcts_destroy.restype = None
lib.mcts_search.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.mcts_search.restype = ctypes.c_int

lib.mcts_search_flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.mcts_search_flow.restype = ctypes.c_int

# Belief API
lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.belief_create.restype = ctypes.c_void_p
lib.belief_destroy.argtypes = [ctypes.c_void_p]
lib.belief_destroy.restype = None
lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.belief_observe.restype = None

# ScriptAgent API
lib.generals_script_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.generals_script_step.restype = ctypes.c_int


PERSONALITY_NAMES = {0: "A-扩张流", 1: "B-城市流", 2: "C-进攻流"}


def run_tournament(n_games=50, seed=42, n_det=4, n_mcts=200, max_steps=300, map_size=12):
    """对每种性格运行锦标赛"""
    results = {}
    
    for personality in [0, 1, 2]:
        name = PERSONALITY_NAMES[personality]
        wins, losses, draws = 0, 0, 0
        mcts_time = 0.0
        total_steps = 0
        
        print(f"\n{'='*50}")
        print(f"IS-MCTS vs {name}  ({n_games} 局)")
        print(f"{'='*50}")
        
        for g in range(n_games):
            game_seed = seed + g * 7
            state = lib.generals_create(map_size, map_size, max_steps, game_seed)
            bs = lib.belief_create(map_size, map_size, 0)
            mcts_engine = lib.mcts_create(game_seed + 999)
            
            done = False
            step_count = 0
            winner = -1
            
            while not done and step_count < max_steps:
                skip = map_size * map_size * 8
                
                # 观察（从真实状态到迷雾信念）
                lib.belief_observe(bs, state, step_count)
                
                # MCTS 搜索（时间计量）
                t0 = time.time()
                action = lib.mcts_search(mcts_engine, bs, 0, n_det, n_mcts)
                t1 = time.time()
                mcts_time += (t1 - t0)
                
                # 执行一步（MCTS vs 脚本）
                winner = lib.generals_script_step(state, action, personality)
                step_count += 1
                
                if winner != -1:
                    done = True
            
            total_steps += step_count
            
            if winner == 0:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
            
            lib.belief_destroy(bs)
            lib.mcts_destroy(mcts_engine)
            lib.generals_destroy(state)
            
            # 进度报告
            if (g + 1) % 10 == 0 or g == n_games - 1:
                pct = wins * 100.0 / (g + 1)
                avg_ms = mcts_time / max(1, total_steps) * 1000
                print(f"  [{g+1:3d}/{n_games}] 胜率 {pct:5.1f}%  ({wins}W/{losses}L/{draws}D)  "
                      f"MCTS {avg_ms:.1f}ms/步")
        
        results[personality] = {
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins * 100.0 / n_games,
            'mcts_ms_per_step': mcts_time / max(1, total_steps) * 1000,
            'avg_steps': total_steps / n_games,
        }
    
    return results


def print_summary(results, n_games, n_det, n_mcts):
    """打印最终汇总"""
    print(f"\n{'='*60}")
    print(f"🏆 IS-MCTS vs 脚本 AI 最终结果 ({n_games} 局/性格)")
    print(f"{'='*60}")
    print(f"  {'性格':<12} {'胜率':>7} {'胜':>4} {'负':>4} {'平':>4}  {'MCTS/步':>9} {'平均步数':>9}")
    print(f"  {'-'*55}")
    
    for personality in [0, 1, 2]:
        r = results[personality]
        name = PERSONALITY_NAMES[personality]
        print(f"  {name:<12} {r['win_rate']:>6.1f}% {r['wins']:>4d} {r['losses']:>4d} {r['draws']:>4d}  "
              f"{r['mcts_ms_per_step']:>7.1f}ms {r['avg_steps']:>7.0f}")
    
    print(f"\n  搜索参数: n_det={n_det}, n_mcts={n_mcts}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=50, help='每性格对局数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--det', type=int, default=4, help='去迷雾化次数')
    parser.add_argument('--mcts', type=int, default=200, help='MCTS迭代次数')
    args = parser.parse_args()
    
    print(f"IS-MCTS vs 脚本 AI 锦标赛")
    print(f"地图: 12x12 | 每局最多 300 步")
    print(f"参数: n_det={args.det}, n_mcts={args.mcts}")
    
    t0 = time.time()
    results = run_tournament(
        n_games=args.games, seed=args.seed,
        n_det=args.det, n_mcts=args.mcts
    )
    elapsed = time.time() - t0
    
    print_summary(results, args.games, args.det, args.mcts)
    print(f"\n总耗时: {elapsed:.1f}s")
