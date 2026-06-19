"""
generate_training_data.py — 脚本对战数据生成器

工作流程：
  1. 用 script_agent.py 让不同性格脚本对战
  2. 每步记录：特征图 + 胜者动作
  3. 只保存"获胜方"的轨迹（仅高效策略）
  4. 输出为 .npz 文件

特征通道 (C=7):
  0: log(己方兵力+1)
  1: log(敌方兵力+1)
  2: log(中立兵力+1)
  3: 地形 [0=空地, 1=山脉, 2=城市, 3=将军]
  4: 到敌方将军的距离图（归一化）
  5: 到己方将军的距离图（归一化）
  6: 边界标记（邻接非己方格子的格子=1）

输出标签:
  policy_target (N*N): 胜者动作的"关注格" one-hot
  action_target (): 0=扩张, 1=集结, 2=进攻, 3=跳过
"""
import os
import numpy as np
import time
from collections import deque
from generals_gym_v4_tiebreaker import GeneralsEnvV4TieBreaker
from script_agent import ScriptAgent, BFSPlanner


class DataCollector:
    """游戏过程中采集状态-动作对"""
    
    def __init__(self, env, winning_player):
        self.env = env
        self.winning_player = winning_player
        self.states = []  # 特征图列表
        self.policies = []  # 关注格 (N*N one-hot)
        self.actions = []  # 动作类型
        self.game_length = 0

    def step_record(self, state_features, focus_cell, action_type):
        """记录一步训练数据（仅当当前玩家是胜者）"""
        if self.env.winner != self.winning_player:
            return
        self.states.append(state_features)
        
        # policy target: focus_cell 位置设为 1
        h, w = self.env.height, self.env.width
        policy = np.zeros(h * w, dtype=np.float32)
        if focus_cell is not None:
            r, c = focus_cell
            policy[r * w + c] = 1.0
        self.policies.append(policy)
        self.actions.append(action_type)

    def get_trajectory(self):
        """返回本局数据（仅保留胜者轨迹）"""
        if len(self.states) == 0:
            return None
        return {
            'states': np.array(self.states, dtype=np.float32),      # (T, C, H, W)
            'policies': np.array(self.policies, dtype=np.float32),  # (T, H*W)
            'actions': np.array(self.actions, dtype=np.int64),      # (T,)
            'winner': self.winning_player,
        }


class FeatureEngine:
    """特征工程：将原始环境状态转为训练特征图"""
    
    @staticmethod
    def compute(env, player_id):
        h, w = env.height, env.width
        features = np.zeros((7, h, w), dtype=np.float32)
        bfs = BFSPlanner()
        
        # Ch0: 己方兵力 (log)
        own = np.where(env.grid_owner == player_id, env.grid_troops, 0)
        features[0] = np.log1p(own)
        
        # Ch1: 敌方兵力 (log)
        enemy = np.where(env.grid_owner == 1 - player_id, env.grid_troops, 0)
        features[1] = np.log1p(enemy)
        
        # Ch2: 中立兵力 (log)
        neutral = np.where(env.grid_owner == -1, env.grid_troops, 0)
        features[2] = np.log1p(neutral)
        
        # Ch3: 地形编码 [0=空地, 1=山脉, 2=城市, 3=将军]
        terrain = np.zeros((h, w), dtype=np.float32)
        terrain[env.grid_type == 1] = 1.0     # 山脉
        terrain[env.grid_type == 3] = 2.0     # 城市
        terrain[env.grid_type == 2] = 3.0     # 将军
        features[3] = terrain
        
        # Ch4: 到敌方将军的距离（归一化）
        enemy_gen = [(r,c) for r in range(h) for c in range(w) 
                    if env.grid_type[r,c]==2 and env.grid_owner[r,c]==1-player_id]
        if enemy_gen:
            dmap = bfs.distance_from_general(env.grid_owner, env.grid_type, 1-player_id)
            dmap = np.where(dmap > 900, 20.0, dmap)
            features[4] = dmap / np.maximum(dmap[dmap < 900].max(), 1)
        
        # Ch5: 到己方将军的距离（归一化）
        own_gen = [(r,c) for r in range(h) for c in range(w) 
                  if env.grid_type[r,c]==2 and env.grid_owner[r,c]==player_id]
        if own_gen:
            dmap = bfs.distance_from_general(env.grid_owner, env.grid_type, player_id)
            dmap = np.where(dmap > 900, 20.0, dmap)
            features[5] = dmap / np.maximum(dmap[dmap < 900].max(), 1)
        
        # Ch6: 边界标记
        frontier = np.zeros((h, w), dtype=np.float32)
        for r in range(h):
            for c in range(w):
                if env.grid_owner[r, c] == player_id:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and env.grid_owner[nr, nc] != player_id:
                            frontier[r, c] = 1.0
                            break
        features[6] = frontier
        
        return features


def action_to_label(agent, action_tuple):
    """将动作元组转为标签 (focus_cell, action_type)
    
    注意: focus_cell 是动作的目标准 (nr, nc)，不是源格 (r,c)！
    这样 FCN 会学到"把兵往哪送"，而不是"兵从哪出来"。
    """
    if action_tuple is None:
        return None, 3  # skip
    
    r, c, direction_idx, is_half = action_tuple
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]  # [上,下,左,右]
    dr, dc = dirs[direction_idx]
    env = agent.env
    nr, nc = r+dr, c+dc
    
    if not (0 <= nr < env.height and 0 <= nc < env.width):
        return None, 3  # 出界→skip
    
    target_owner = env.grid_owner[nr, nc]
    if target_owner == 1 - agent.player_id:
        action_type = 2  # 进攻: 目标是敌方格
    elif target_owner == -1:
        action_type = 0  # 扩张: 目标是中立格
    else:
        action_type = 1  # 集结: 目标是己方格
    
    return (nr, nc), action_type  # ← 改为重点：返回目标格


def run_script_battle(personality_a, personality_b, max_steps=300):
    """运行一场脚本对战，返回胜者数据"""
    env = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=max_steps)
    obs, info = env.reset()
    
    agent_a = ScriptAgent(env, personality=personality_a, player_id=0)
    agent_b = ScriptAgent(env, personality=personality_b, player_id=1)
    
    collector = None
    trajectories = []
    game_data = []
    
    for step in range(max_steps):
        # Player 0 行动
        action_a = agent_a.decide()
        if action_a is None:
            action_id = env.SKIP_ACTION
        else:
            r, c, d, hf = action_a
            action_id = (r * env.width + c) * 8 + d * 2 + hf
        
        obs, reward, terminated, truncated, info = env.step(action_id)
        
        # Player 0 的记录（如果最终是胜者）
        focus, atype = action_to_label(agent_a, action_a)
        
        if step == 0:
            # 第一局不知道胜者，先收集所有数据
            pass
        
        if terminated or truncated:
            winner = env.winner
            break
        
        # 更新记忆
        agent_a.last_enemy_seen.clear()
        for r in range(env.height):
            for c in range(env.width):
                if env.grid_owner[r, c] == 1:
                    agent_a.last_enemy_seen[(r, c)] = env.grid_troops[r, c]

    if winner == 0:
        # 重放并记录胜者（player 0）的轨迹
        env2 = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=max_steps)
        obs, info = env2.reset()
        agent_a2 = ScriptAgent(env2, personality=personality_a, player_id=0)
        
        states, policies, actions = [], [], []
        for step in range(max_steps):
            feat = FeatureEngine.compute(env2, player_id=0)
            act = agent_a2.decide()
            focus, atype = action_to_label(agent_a2, act)
            
            states.append(feat)
            if focus is not None:
                pol = np.zeros(env2.height * env2.width, dtype=np.float32)
                pol[focus[0] * env2.width + focus[1]] = 1.0
                policies.append(pol)
            else:
                policies.append(np.zeros(env2.height * env2.width, dtype=np.float32))
            actions.append(atype)
            
            if act is None:
                aid = env2.SKIP_ACTION
            else:
                r, c, d, hf = act
                aid = (r * env2.width + c) * 8 + d * 2 + hf
            
            obs, r, term, trunc, info = env2.step(aid)
            if term or trunc:
                break
        
        return {
            'states': np.array(states, dtype=np.float32),
            'policies': np.array(policies, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'winner': 0,
            'personality_a': personality_a,
            'personality_b': personality_b,
        }
    
    return None


def generate_dataset(num_games=1000, output_dir='training_data', personalities=None):
    """批量生成训练数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    if personalities is None:
        personalities = ['A', 'B', 'C']
    
    # 所有组合的对战
    matchups = [(p1, p2) for p1 in personalities for p2 in personalities if p1 != p2]
    # 也加入同性格内战
    for p in personalities:
        matchups.append((p, p))
    
    all_data = []
    games_per_matchup = max(1, num_games // len(matchups))
    
    total = 0
    t0 = time.time()
    
    for p1, p2 in matchups:
        for i in range(games_per_matchup):
            data = run_script_battle(p1, p2, max_steps=300)
            if data is not None:
                all_data.append(data)
                total += len(data['states'])
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f'  [{p1} vs {p2}] {i+1}/{games_per_matchup} games | '
                      f'{total} samples | {elapsed:.1f}s')
    
    # 合并所有数据
    if all_data:
        merged = {
            'states': np.concatenate([d['states'] for d in all_data], axis=0),
            'policies': np.concatenate([d['policies'] for d in all_data], axis=0),
            'actions': np.concatenate([d['actions'] for d in all_data], axis=0),
        }
        
        filepath = os.path.join(output_dir, 'script_dataset.npz')
        np.savez_compressed(filepath, **merged)
        
        elapsed = time.time() - t0
        print(f'\n✅ 数据集保存: {filepath}')
        print(f'   总样本: {len(merged["states"]):,}')
        print(f'   状态维度: {merged["states"].shape[1:]}')
        print(f'   耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')
        print(f'   速度: {len(merged["states"])/elapsed:.0f} samples/s')
        
        # 动作分布
        unique, counts = np.unique(merged['actions'], return_counts=True)
        labels = ['扩张', '集结', '进攻', '跳过']
        print(f'\n   动作分布:')
        for u, c in zip(unique, counts):
            print(f'     {labels[u]}: {c} ({c/len(merged["actions"])*100:.1f}%)')
        
        return merged
    else:
        print('⚠️ 没有采集到任何数据')
        return None


if __name__ == '__main__':
    # 快速测试：先跑 100 局验证
    print('=== 数据生成器测试 (100局) ===')
    data = generate_dataset(num_games=100)
