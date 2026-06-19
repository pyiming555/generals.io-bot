"""
script_agent.py — 多性格规则脚本智能体（老师）

核心组件：
  1. BFS 寻路引擎：计算可达格、距离图
  2. 评分系统：对每个格子计算多维度分数
  3. 三种性格：扩张流、城市流、进攻流

使用方法：
  agent = ScriptAgent(env, personality='A')
  action = agent.decide()  # 返回 (r, c, direction, is_half) 或 None（跳过）
"""
import numpy as np
from collections import deque


class BFSPlanner:
    """BFS 寻路与状态评估引擎"""
    
    @staticmethod
    def reachable_tiles(grid_type, grid_owner, start_positions, max_steps=600):
        """从 start_positions 出发，可达的格子集合"""
        h, w = grid_type.shape
        visited = set(start_positions)
        q = deque(start_positions)
        dist = {p: 0 for p in start_positions}
        
        while q:
            r, c = q.popleft()
            if dist[(r, c)] >= max_steps:
                continue
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and grid_type[nr, nc] != 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
        return visited, dist

    @staticmethod
    def distance_to_enemy(grid_owner, player_id):
        """每格到最近敌方格子的棋盘距离（对山脉取大值）"""
        h, w = grid_owner.shape
        enemy_positions = [(r,c) for r in range(h) for c in range(w) 
                          if grid_owner[r,c] == 1-player_id]
        if not enemy_positions:
            return np.full((h, w), 999, dtype=np.float32)
        
        dist = np.full((h, w), 999, dtype=np.float32)
        q = deque(enemy_positions)
        for r, c in enemy_positions:
            dist[r, c] = 0
        while q:
            r, c = q.popleft()
            nd = dist[r, c] + 1
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > nd:
                    dist[nr, nc] = nd
                    q.append((nr, nc))
        return dist

    @staticmethod
    def distance_from_general(grid_owner, grid_type, player_id):
        """每格到己方将军的步数（绕山脉）"""
        h, w = grid_owner.shape
        general_positions = [(r,c) for r in range(h) for c in range(w) 
                            if grid_type[r,c]==2 and grid_owner[r,c]==player_id]
        if not general_positions:
            return np.full((h, w), 999, dtype=np.float32)
        
        dist = np.full((h, w), 999, dtype=np.float32)
        q = deque(general_positions)
        for r, c in general_positions:
            dist[r, c] = 0
        while q:
            r, c = q.popleft()
            nd = dist[r, c] + 1
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and grid_type[nr, nc] != 1 and dist[nr, nc] > nd:
                    dist[nr, nc] = nd
                    q.append((nr, nc))
        return dist

    @staticmethod
    def frontier_cells(grid_owner, player_id):
        """边界格：己方格子邻接敌方或中立"""
        h, w = grid_owner.shape
        frontier = []
        for r in range(h):
            for c in range(w):
                if grid_owner[r, c] != player_id:
                    continue
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and grid_owner[nr, nc] != player_id:
                        frontier.append((r, c))
                        break
        return frontier


class ScriptAgent:
    """多性格规则脚本智能体"""
    
    PERSONALITIES = {
        'A': {  # 疯狂扩张流
            'w_expand': 1.0,
            'w_city': 0.3,
            'w_attack': 0.1,
            'w_defense': 0.1,
            'w_frontier': 0.8,
            'min_troops_to_move': 2,
            'half_move_threshold': 5,
            'aggression_radius': 15,
        },
        'B': {  # 憋兵城市流
            'w_expand': 0.4,
            'w_city': 1.0,
            'w_attack': 0.3,
            'w_defense': 0.6,
            'w_frontier': 0.3,
            'min_troops_to_move': 3,
            'half_move_threshold': 8,
            'aggression_radius': 10,
        },
        'C': {  # 暴力进攻流
            'w_expand': 0.3,
            'w_city': 0.5,
            'w_attack': 1.0,
            'w_defense': 0.2,
            'w_frontier': 0.6,
            'min_troops_to_move': 2,
            'half_move_threshold': 4,
            'aggression_radius': 25,
        },
    }

    def __init__(self, env, personality='A', player_id=0):
        self.env = env
        self.player_id = player_id
        self.personality = personality
        self.config = self.PERSONALITIES[personality].copy()
        self.bfs = BFSPlanner()
        self.last_enemy_seen = {}  # 记忆：最后一次看见敌方兵力的位置

    def set_personality(self, personality):
        self.personality = personality
        self.config = self.PERSONALITIES[personality].copy()

    def _score_cell(self, r, c, dist_to_enemy):
        """对单个格子计算多维度得分（传距离图避免重复BFS）"""
        env = self.env
        grid = {
            'type': env.grid_type,
            'owner': env.grid_owner,
            'troops': env.grid_troops,
        }
        cfg = self.config
        score = 0.0
        h, w = grid['type'].shape

        # 1) 扩张分：占领中立格
        if grid['owner'][r, c] != self.player_id:
            # 中立格子（非山脉、非将军）
            if grid['type'][r, c] != 1 and not (grid['type'][r, c] == 2 and grid['owner'][r, c] != -1):
                score += cfg['w_expand'] * 0.5
            # 城市
            if grid['type'][r, c] == 3:
                score += cfg['w_city'] * 2.0 * (1.0 - grid['troops'][r, c] / 50.0)
            # 敌方格子
            if grid['owner'][r, c] == 1 - self.player_id:
                score += cfg['w_attack'] * 1.0
                # 如果是敌方将军，猛加分
                if grid['type'][r, c] == 2:
                    score += cfg['w_attack'] * 5.0
        else:
            # 2) 防守分：本方格子
            score += cfg['w_defense'] * 0.1

        # 3) 边界分：靠近敌人的前线
        if dist_to_enemy[r, c] < cfg['aggression_radius']:
            score += cfg['w_frontier'] * (1.0 - dist_to_enemy[r, c] / cfg['aggression_radius'])

        # 4) 记忆分：最后一次在附近看见敌人
        if (r,c) in self.last_enemy_seen:
            score += 0.3

        return score

    def _select_target(self):
        """选择当前最优目标格子（返回 (r, c)）"""
        env = self.env
        h, w = env.grid_type.shape
        cfg = self.config
        
        # 距离图一次性算好，避免每个格子重复 BFS
        dist_to_enemy = self.bfs.distance_to_enemy(env.grid_owner, self.player_id)

        best_score = -999
        best_cell = None
        
        for r in range(h):
            for c in range(w):
                if env.grid_type[r, c] == 1:
                    continue  # 跳过山脉
                # 只考虑己方可达范围内的非己方格子
                if env.grid_owner[r, c] == self.player_id:
                    # 本方格子作为中转集结
                    if env.grid_troops[r, c] >= cfg['min_troops_to_move']:
                        score = self._score_cell(r, c, dist_to_enemy)
                        if score > best_score:
                            best_score = score
                            best_cell = (r, c)
                else:
                    score = self._score_cell(r, c, dist_to_enemy)
                    if score > best_score:
                        best_score = score
                        best_cell = (r, c)

        return best_cell

    def _path_to_target(self, target_r, target_c):
        """BFS 找从目标格到最近己方出兵格的路径（返回 [出兵格, ..., 目标格]）"""
        env = self.env
        h, w = env.grid_type.shape
        
        q = deque()
        q.append((target_r, target_c))
        visited = {(target_r, target_c)}
        parent = {(target_r, target_c): None}
        
        found = None
        while q:
            r, c = q.popleft()
            if env.grid_owner[r, c] == self.player_id and env.grid_troops[r, c] > 1:
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
        
        # 重构路径: found → ... → target
        path = [found]
        cur = found
        while parent.get(cur):
            cur = parent[cur]
            if cur is None: break
            path.append(cur)
        # path = [found, step1, ..., target]
        return path

    def _move_troops_toward(self, src_r, src_c, tgt_r, tgt_c):
        """从源格向目标格方向移动兵力"""
        # 方向: 0=上(-1,0), 1=下(1,0), 2=左(0,-1), 3=右(0,1)
        direction_map = {
            (-1, 0): 0,  # 上
            (1, 0): 1,   # 下
            (0, -1): 2,  # 左
            (0, 1): 3,   # 右
        }
        
        moves = []
        if tgt_r < src_r: moves.append((-1, 0))  # 向上
        elif tgt_r > src_r: moves.append((1, 0)) # 向下
        if tgt_c < src_c: moves.append((0, -1))  # 向左
        elif tgt_c > src_c: moves.append((0, 1)) # 向右
        
        for dr, dc in moves:
            nr, nc = src_r+dr, src_c+dc
            if 0 <= nr < self.env.height and 0 <= nc < self.env.width:
                if self.env.grid_type[nr, nc] != 1:
                    direction = direction_map[(dr, dc)]
                    troops = self.env.grid_troops[src_r, src_c]
                    is_half = 1 if troops >= self.config['half_move_threshold'] else 0
                    return (src_r, src_c, direction, is_half)
        
        return None

    def decide(self):
        """返回动作元组 (r, c, direction, is_half) 或 None（跳过）"""
        env = self.env
        
        # 更新记忆（记录看到的敌方兵力）
        for r in range(env.height):
            for c in range(env.width):
                if env.grid_owner[r, c] == 1 - self.player_id:
                    self.last_enemy_seen[(r, c)] = env.grid_troops[r, c]
        
        # 选择目标格
        target = self._select_target()
        if target is None:
            return None  # 无目标，跳过
        
        tgt_r, tgt_c = target
        
        # 如果目标已经是本方领土且有兵，从它往敌方方向走
        if env.grid_owner[tgt_r, tgt_c] == self.player_id and env.grid_troops[tgt_r, tgt_c] > 1:
            # 向最近的非本方格子推进
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = tgt_r+dr, tgt_c+dc
                if 0 <= nr < env.height and 0 <= nc < env.width:
                    if env.grid_owner[nr, nc] != self.player_id and env.grid_type[nr, nc] != 1:
                        direction = [(1,0), (-1,0), (0,1), (0,-1)].index((dr, dc))
                        troops = env.grid_troops[tgt_r, tgt_c]
                        is_half = 1 if troops >= self.config['half_move_threshold'] else 0
                        return (tgt_r, tgt_c, direction, is_half)
            return None
        
        # 找路径并从最近的出兵格运兵
        path = self._path_to_target(tgt_r, tgt_c)
        if path is None or len(path) < 2:
            return None
        
        # 路径第一步：从 path[0] 向 path[1] 移动
        sr, sc = path[0]
        tr, tc = path[1]
        return self._move_troops_toward(sr, sc, tr, tc)

    def random_valid_move(self):
        """随机合法动作（后退选项）"""
        env = self.env
        mask = env.valid_action_mask(self.player_id)
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return None
        # 尽量不要跳过
        non_skip = [a for a in valid if a != env.SKIP_ACTION]
        if non_skip:
            choice = np.random.choice(non_skip)
        else:
            choice = np.random.choice(valid)
        return env._decode_action(choice)
