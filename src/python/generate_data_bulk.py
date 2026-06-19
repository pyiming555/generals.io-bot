"""
generate_data_bulk.py — 批量生成训练数据

运行脚本对战，只保存胜者轨迹
"""
import os, sys, time
import numpy as np
from collections import deque

# 确保能找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generals_gym_v4_tiebreaker import GeneralsEnvV4TieBreaker
from script_agent import ScriptAgent
from generate_training_data import FeatureEngine, action_to_label


def play_and_record(personality_a, personality_b, max_steps=300):
    """打一局并记录胜者（player 0）轨迹"""
    # 第一遍：看谁赢
    env = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=max_steps)
    obs, info = env.reset()
    agent_a = ScriptAgent(env, personality=personality_a, player_id=0)
    
    for _ in range(max_steps):
        act = agent_a.decide()
        if act is None:
            aid = env.SKIP_ACTION
        else:
            r,c,d,hf = act
            aid = (r*env.width+c)*8+d*2+hf
        obs, r, t, tr, info = env.step(aid)
        if t or tr:
            break
    
    if env.winner != 0:
        return None  # 输了不记录
    
    # 第二遍：重新打并记录
    env2 = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=max_steps)
    obs, info = env2.reset()
    agent_a2 = ScriptAgent(env2, personality=personality_a, player_id=0)
    
    states, policies, actions = [], [], []
    h, w = env2.height, env2.width
    
    for _ in range(max_steps):
        feat = FeatureEngine.compute(env2, player_id=0)
        states.append(feat)
        
        act = agent_a2.decide()
        focus, atype = action_to_label(agent_a2, act)
        
        pol = np.zeros(h * w, dtype=np.float32)
        if focus is not None:
            pol[focus[0] * w + focus[1]] = 1.0
        policies.append(pol)
        actions.append(atype)
        
        if act is None:
            aid = env2.SKIP_ACTION
        else:
            r,c,d,hf = act
            aid = (r*w+c)*8+d*2+hf
        obs, r, t, tr, info = env2.step(aid)
        if t or tr:
            break
    
    return {
        'states': np.array(states, dtype=np.float32),
        'policies': np.array(policies, dtype=np.float32),
        'actions': np.array(actions, dtype=np.int64),
    }


def main():
    os.makedirs('training_data', exist_ok=True)
    
    # 对战组合（只保留 player 0 能赢的）
    matchups = [
        ('A', 'B'), ('A', 'C'), ('B', 'C'),
        ('B', 'A'), ('C', 'A'), ('C', 'B'),
        ('A', 'A'), ('B', 'B'), ('C', 'C'),
    ]
    
    target_samples = 30000
    all_data = []
    total = 0
    t0 = time.time()
    
    print(f'目标: {target_samples:,} 样本')
    matchup_str = ', '.join([f'{a}vs{b}' for a,b in matchups])
    print(f'组合: [{matchup_str}]')
    print()
    
    # 按组合轮流采集，直到达到目标
    round_num = 0
    while total < target_samples:
        round_num += 1
        for p1, p2 in matchups:
            data = play_and_record(p1, p2, max_steps=300)
            if data is not None:
                all_data.append(data)
                total += len(data['states'])
            
            if total >= target_samples:
                break
        
        if round_num % 5 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f'  Round {round_num}: {total:,} samples ({rate:.0f}/s) — {elapsed:.0f}s')
    
    # 合并数据
    print(f'\n合并数据...')
    merged = {
        'states': np.concatenate([d['states'] for d in all_data], axis=0),
        'policies': np.concatenate([d['policies'] for d in all_data], axis=0),
        'actions': np.concatenate([d['actions'] for d in all_data], axis=0),
    }
    
    filepath = 'training_data/script_dataset.npz'
    np.savez_compressed(filepath, **merged)
    
    elapsed = time.time() - t0
    unique, counts = np.unique(merged['actions'], return_counts=True)
    labels = ['扩张', '集结', '进攻', '跳过']
    
    mn = len(merged['states'])
    print(f'\n✅ 数据集保存: {filepath}')
    print(f'   总样本: {mn:,}')
    print(f'   状态维度: {merged["states"].shape[1:]}')
    print(f'   耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')
    speed = mn / elapsed if elapsed > 0 else 0
    print(f'   速度: {speed:.0f} samples/s')
    print(f'\n   动作分布:')
    total_actions = len(merged['actions'])
    for u, c in zip(unique, counts):
        pct = c / total_actions * 100
        print(f'     {labels[u]}: {c} ({pct:.1f}%)')


if __name__ == '__main__':
    main()
