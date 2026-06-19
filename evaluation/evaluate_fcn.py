"""
evaluate_fcn.py — FCN 模型评估

让 FCN 模型 vs 随机 Bot 打 N 局，报告胜率和详细统计。
"""
import sys
import os
import numpy as np
import random
import time
from collections import Counter

# 固定随机种子保证可复现
SEED = 42

# 加载 FCN 模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
# 模型路径
_model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
from fcn_model import LightweightFCN
from inference import FCNInference
from generals_gym_v4_tiebreaker import GeneralsEnvV4TieBreaker

# 结果类型
KILL = '斩首'        # 直接杀死将军
TIEBREAK_WIN = '绝杀'  # 600步到时间，兵力+领地评分胜出
LOSS = '落败'
TRUE_DRAW = '绝对平局'

def evaluate(n_games=100, model_path=None, verbose=False):
    if model_path is None:
        model_path = os.path.join(_model_dir, 'fcn_script_model.pt')
    agent = FCNInference(model_path)
    
    results = []
    episode_lengths = []
    win_types = []
    max_troops_per_game = []
    
    t0 = time.time()
    
    for game in range(n_games):
        env = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=600)
        obs, _ = env.reset()
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            # FCN 模型决策
            action = agent.decide(env)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
        
        # 结果分析
        max_troops = np.max(obs[0]) if obs[0].sum() > 0 else 0
        
        if env.winner == 0:
            if env.stalemate:
                result = TIEBREAK_WIN
            else:
                result = KILL
            win_types.append(result)
        elif env.winner == 1:
            result = LOSS
        else:
            result = TRUE_DRAW
        
        results.append(result)
        episode_lengths.append(step_count)
        max_troops_per_game.append(max_troops)
        
        if verbose:
            print(f"Game {game+1:3d}/{n_games}: {result:6s} ({step_count:3d} steps)")
    
    elapsed = time.time() - t0
    
    # ====== 统计 ======
    total = len(results)
    wins = sum(1 for r in results if r != LOSS)
    losses = sum(1 for r in results if r == LOSS)
    kills = sum(1 for r in results if r == KILL)
    tiebreak_wins = sum(1 for r in results if r == TIEBREAK_WIN)
    draws = sum(1 for r in results if r == TRUE_DRAW)
    win_rate = wins / total * 100
    
    print(f"\n{'='*50}")
    print(f"  FCN 模型评估结果")
    print(f"  模型: {model_path}")
    print(f"  对手: 随机 Bot")
    print(f"  局数: {n_games}")
    print(f"{'='*50}")
    print(f"  🏆 总胜率:    {win_rate:5.1f}% ({wins}/{total})")
    print(f"  ├─ ⚔️  斩首胜:  {kills:5d} 局 ({kills/total*100:5.1f}%)")
    print(f"  ├─ 📊 绝杀胜:  {tiebreak_wins:5d} 局 ({tiebreak_wins/total*100:5.1f}%)")
    print(f"  ├─ 💀 落败:    {losses:5d} 局 ({losses/total*100:5.1f}%)")
    print(f"  └─ 🤝 绝对平局: {draws:5d} 局 ({draws/total*100:5.1f}%)")
    print(f"")
    print(f"  平均步数:     {np.mean(episode_lengths):5.1f}")
    print(f"  中位数步数:   {np.median(episode_lengths):5.1f}")
    print(f"  平均最大兵力: {np.mean(max_troops_per_game):5.0f}")
    print(f"  总耗时:       {elapsed:.1f}s ({elapsed/n_games*1000:.0f}ms/局)")
    print(f"{'='*50}")
    
    return results, episode_lengths

if __name__ == '__main__':
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    evaluate(n_games, model_path, verbose=True)
