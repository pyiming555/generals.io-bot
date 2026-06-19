"""
evaluate_nn_cpp.py — 用 C++ LibTorch NN 推理评估 vs 纯MCTS

用法: python evaluate_nn_cpp.py [--games 25] [--det 4] [--mcts 200] [--model resnet_v4.ptl]
"""
import ctypes
import time
import os
import sys
import numpy as np

# ================================================================
# 加载 LibTorch 版共享库
# ================================================================
_lib_path = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'libgenerals_nn.so')
lib = ctypes.cdll.LoadLibrary(_lib_path)

# --- 基本引擎 ---
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
lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.generals_step_dual.restype = ctypes.c_int

# --- MCTS 搜索 ---
lib.mcts_create.argtypes = [ctypes.c_uint]
lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]
lib.mcts_destroy.restype = None
lib.mcts_search_flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.mcts_search_flow.restype = ctypes.c_int
lib.mcts_search_nn_auto.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.mcts_search_nn_auto.restype = ctypes.c_int

# --- NNPredictor ---
lib.nn_create.argtypes = [ctypes.c_char_p]
lib.nn_create.restype = ctypes.c_void_p
lib.nn_destroy.argtypes = [ctypes.c_void_p]
lib.nn_destroy.restype = None

# --- Belief ---
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
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'rl_models')


def run_tournament(model_name, n_games=25, det=4, mcts_iter=200):
    """
    NN+MCTS (C++ LibTorch 推理) vs 纯MCTS 锦标赛
    双边各 n_games 局然后交换先后手
    """
    # 创建 NNPredictor
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"❌ 模型 {model_path} 不存在！")
        # 尝试 .pt 格式
        model_path = model_path.replace('.ptl', '.pt')
        if not os.path.exists(model_path):
            print(f"❌ 也找不到 {model_path}")
            sys.exit(1)
    
    nn = lib.nn_create(model_path.encode())
    if not nn:
        print(f"❌ 无法加载模型 {model_path}")
        sys.exit(1)
    print(f"✓ 加载模型 (C++ LibTorch): {model_path}")
    
    total_nn_wins = 0
    total_pure_wins = 0
    total_draws = 0
    total_time = 0.0
    nn_calls = 0  # 统计NN+MCTS调用次数
    
    for swap in [0, 1]:
        label = "NN(红) vs 纯MCTS(蓝)" if swap == 0 else "纯MCTS(红) vs NN(蓝)"
        print(f"\n{'='*55}")
        print(f"🏆 {label}")
        print(f"{'='*55}")
        
        nn_wins = 0
        pure_wins = 0
        draws = 0
        
        for g in range(n_games):
            seed = 42 + g * 13 + swap * 777
            gs = lib.generals_create(MAP_SIZE, MAP_SIZE, MAX_STEPS, seed)
            bs0 = lib.belief_create(MAP_SIZE, MAP_SIZE, 0)
            bs1 = lib.belief_create(MAP_SIZE, MAP_SIZE, 1)
            m0 = lib.mcts_create(seed)
            m1 = lib.mcts_create(seed + 999)
            
            step, winner = 0, -1
            while step < MAX_STEPS and winner == -1:
                if swap == 0:
                    # 红 = NN, 蓝 = 纯MCTS
                    lib.belief_observe(bs0, gs, step)
                    t0 = time.time()
                    act0 = lib.mcts_search_nn_auto(m0, bs0, 0, det, mcts_iter, nn)
                    t1 = time.time()
                    total_time += (t1 - t0)
                    nn_calls += 1
                    
                    lib.belief_observe(bs1, gs, step)
                    act1 = lib.mcts_search_flow(m1, bs1, 1, det, mcts_iter)
                else:
                    # 红 = 纯MCTS, 蓝 = NN
                    lib.belief_observe(bs0, gs, step)
                    act0 = lib.mcts_search_flow(m0, bs0, 0, det, mcts_iter)
                    
                    lib.belief_observe(bs1, gs, step)
                    t0 = time.time()
                    act1 = lib.mcts_search_nn_auto(m1, bs1, 1, det, mcts_iter, nn)
                    t1 = time.time()
                    total_time += (t1 - t0)
                    nn_calls += 1
                
                winner = lib.generals_step_dual(gs, act0, act1)
                step += 1
            
            if swap == 0:
                if winner == 0: nn_wins += 1
                elif winner == 1: pure_wins += 1
                else: draws += 1
            else:
                if winner == 0: pure_wins += 1
                elif winner == 1: nn_wins += 1
                else: draws += 1
            
            result = "NN胜" if (winner == 0 and swap == 0) or (winner == 1 and swap == 1) else \
                     "纯MCTS胜" if (winner == 1 and swap == 0) or (winner == 0 and swap == 1) else \
                     "平局" if winner == 2 else "超时"
            print(f"  [{g+1:2d}/{n_games}] {result}, {step}步")
            
            lib.belief_destroy(bs0); lib.belief_destroy(bs1)
            lib.mcts_destroy(m0); lib.mcts_destroy(m1)
            lib.generals_destroy(gs)
        
        total_nn_wins += nn_wins
        total_pure_wins += pure_wins
        total_draws += draws
    
    lib.nn_destroy(nn)
    
    total = n_games * 2
    avg_ms = total_time / max(1, nn_calls) * 1000 if nn_calls > 0 else 0
    print(f"\n{'='*55}")
    print(f"🏆 最终结果 (C++ LibTorch NN)")
    print(f"{'='*55}")
    print(f"  🤖 NN+MCTS:  {total_nn_wins:3d} 胜 ({total_nn_wins/total*100:.1f}%)")
    print(f"  🧮 纯MCTS:   {total_pure_wins:3d} 胜 ({total_pure_wins/total*100:.1f}%)")
    print(f"  🤝 平局:     {total_draws}")
    print(f"  ⏱ NN+MCTS:  {avg_ms:.1f}ms/步 (共{nn_calls}次调用)")
    print(f"     整局换算: {avg_ms * 150:.0f}ms/局")
    
    # 保存结果
    result_path = os.path.join(os.path.dirname(__file__), 'rl_data', 'nn_cpp_vs_pure_result.txt')
    with open(result_path, 'w') as f:
        f.write(f"C++ LibTorch NN vs 纯MCTS\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"对局: {total}\n")
        f.write(f"NN+MCTS: {total_nn_wins}胜 ({total_nn_wins/total*100:.1f}%)\n")
        f.write(f"纯MCTS:  {total_pure_wins}胜 ({total_pure_wins/total*100:.1f}%)\n")
    print(f"\n  结果保存: {result_path}")
    
    return total_nn_wins, total_pure_wins, total_draws


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=25, help='每颜色对局数')
    parser.add_argument('--det', type=int, default=4)
    parser.add_argument('--mcts', type=int, default=None,
                        help='MCTS迭代数 (默认: LADDER=260)')
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1, 2],
                        help='BotMode: 0=FAST_TRAIN(200), 1=LADDER(260), 2=ANALYSIS(300)')
    parser.add_argument('--model', type=str, default='resnet_v4.ptl', help='TorchScript模型文件')
    args = parser.parse_args()
    
    # BotMode 解析：如果未指定 --mcts，则根据 --mode 自动选择
    lib.bot_get_default_mcts.argtypes = [ctypes.c_int]
    lib.bot_get_default_mcts.restype = ctypes.c_int
    if args.mcts is None:
        mcts_iters = lib.bot_get_default_mcts(args.mode)
        mode_names = ['FAST_TRAIN(200)', 'LADDER(260)', 'ANALYSIS(300)']
        print(f"🤖 BotMode: {mode_names[args.mode]} → n_mcts={mcts_iters}")
    else:
        mcts_iters = args.mcts
        print(f"🎯 手动指定: n_mcts={mcts_iters}")
    
    t0 = time.time()
    run_tournament(args.model, args.games, args.det, mcts_iters)
    print(f"\n总耗时: {time.time()-t0:.1f}s")
