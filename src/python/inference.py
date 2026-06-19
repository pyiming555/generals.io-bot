"""
inference.py — FCN 推理管线

工作流程：
  1. FeatureEngine 计算环境特征
  2. FCN 预测关注格 + 动作类型
  3. A* 规划路径，将兵力运往关注格
  4. 执行移动

线上使用：
  agent = FCNInference('fcn_script_model.pt')
  action_id = agent.decide(env)
"""
import os
import math
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

from fcn_model import LightweightFCN
from generate_training_data import FeatureEngine


class AStarPlanner:
    """A* 寻路引擎 — 从出兵格到目标的微观移动"""
    
    @staticmethod
    def find_path(env, start, goal):
        """
        A* 寻路，绕开山脉
        返回 [(r,c), ...] 从 start 到 goal 的路径
        """
        h, w = env.height, env.width
        if env.grid_type[start[0], start[1]] == 1 or env.grid_type[goal[0], goal[1]] == 1:
            return None
        
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan
        
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda p: f_score.get(p, float('inf')))
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            open_set.remove(current)
            
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = current[0]+dr, current[1]+dc
                neighbor = (nr, nc)
                
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if env.grid_type[nr, nc] == 1:
                    continue  # 山脉不可通行
                
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    open_set.add(neighbor)
        
        return None  # 无路径

    @staticmethod
    def find_nearest_owned_with_troops(env, target_r, target_c, player_id=0):
        """BFS 从目标格找最近的己方有兵格子"""
        h, w = env.height, env.width
        q = deque()
        q.append((target_r, target_c))
        visited = {(target_r, target_c)}
        parent = {(target_r, target_c): None}
        
        found = None
        while q:
            r, c = q.popleft()
            if env.grid_owner[r, c] == player_id and env.grid_troops[r, c] > 1:
                found = (r, c)
                break
            
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and env.grid_type[nr, nc] != 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        
        if found is None:
            return None
        
        # 重构路径
        path = [found]
        cur = found
        while parent.get(cur):
            cur = parent[cur]
            if cur is None: break
            path.append(cur)
        return path  # [出兵格, ..., 目标格]


class FCNInference:
    """FCN 推理包装器"""
    
    def __init__(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', '..', 'models', 'fcn_script_model.pt'
            )
        self.device = torch.device(device)
        self.model = LightweightFCN(in_channels=7, base_filters=64)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.feature_engine = FeatureEngine()
        self.planner = AStarPlanner()
        self.last_focus = None  # 记住上一帧的关注目标，保持连续性
    
    @torch.no_grad()
    def predict(self, env):
        """
        FCN 推理，返回 (focus_r, focus_c, action_type)
        """
        # 计算特征
        features = self.feature_engine.compute(env, player_id=0)
        x = torch.from_numpy(features[None]).float().to(self.device)
        
        # FCN 推理
        policy_logits, action_logits = self.model(x)
        
        # Policy: 关注格概率图
        h, w = features.shape[1:]
        policy_flat = policy_logits.view(1, -1)
        policy_prob = F.softmax(policy_flat, dim=1).cpu().numpy()[0]
        
        # 🌟 屏蔽己方已占格（只选未占/敌方格作为目标）
        owned = (env.grid_owner == 0)
        owned_mask = owned.flatten()
        # 但保留将军格 (type=2) 在所有情况下可以选（防御需要）
        # 以及保留部分己方格（集结需要）
        # 简单策略：把己方已占格的概率降低到 1/100
        policy_prob_masked = policy_prob.copy()
        for idx in range(len(policy_prob_masked)):
            r, c = idx // w, idx % w
            if env.grid_owner[r, c] == 0 and env.grid_troops[r, c] > 1:
                policy_prob_masked[idx] *= 0.01  # 大降己方有兵格
        
        # 归一化
        policy_prob_masked = policy_prob_masked / policy_prob_masked.sum()
        
        # 从屏蔽后的分布选择
        best_idx = np.argmax(policy_prob_masked)
        focus_r, focus_c = best_idx // w, best_idx % w
        
        # 动作类型
        action_logits_np = action_logits[0, :, focus_r, focus_c].cpu().numpy()
        action_type = int(np.argmax(action_logits_np))
        
        self.last_focus = (focus_r, focus_c)
        return focus_r, focus_c, action_type
    
    def decide(self, env):
        """
        根据 FCN 推理决定下一步动作
        返回动作ID (可直接传入 env.step)
        """
        focus_r, focus_c, action_type = self.predict(env)
        
        # 找从最近出兵格到关注格的路线
        path = self.planner.find_nearest_owned_with_troops(env, focus_r, focus_c, player_id=0)
        
        if path is None or len(path) < 2:
            return env.SKIP_ACTION
        
        # 从 path[0] (出兵格) 向 path[1] 移动
        sr, sc = path[0]
        tr, tc = path[1]
        
        # 方向: 0=上(-1,0), 1=下(1,0), 2=左(0,-1), 3=右(0,1)
        direction = -1
        if tr < sr: direction = 0  # 上
        elif tr > sr: direction = 1  # 下
        elif tc < sc: direction = 2  # 左
        elif tc > sc: direction = 3  # 右
        
        if direction == -1:
            return env.SKIP_ACTION
        
        troops = env.grid_troops[sr, sc]
        is_half = 1 if troops >= 5 else 0  # 兵多则分兵
        
        return (sr * env.width + sc) * 8 + direction * 2 + is_half


if __name__ == '__main__':
    from generals_gym_v4_tiebreaker import GeneralsEnvV4TieBreaker
    import time
    
    # 快速测试
    env = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=100)
    obs, info = env.reset()
    
    agent = FCNInference(None)  # 自动找到 models/fcn_script_model.pt
    
    t0 = time.time()
    for i in range(100):
        act = agent.decide(env)
        obs, r, term, trunc, info = env.step(act)
        if term or trunc: break
    
    elapsed = time.time() - t0
    print(f'Game: {i+1} steps, {elapsed:.1f}s ({elapsed/(i+1)*1000:.1f}ms/step)')
    print(f'Winner: {env.winner}')
    print('✅ FCN inference works!')
