import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
from sb3_contrib import MaskablePPO

class GeneralsSelfPlayEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, width=8, height=8, max_steps=400):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        
        self.action_space = spaces.Discrete(self.height * self.width * 4 + 1)
        self.SKIP_ACTION = self.height * self.width * 4
        self.observation_space = spaces.Box(low=0, high=10000, shape=(6, self.height, self.width), dtype=np.float32)

        self.grid_type = np.zeros((self.height, self.width), dtype=int)
        self.grid_owner = np.full((self.height, self.width), -1, dtype=int)
        self.grid_troops = np.zeros((self.height, self.width), dtype=int)
        self.winner = -1
        
        # 🌟 新增：内置的对手模型
        self.opponent_model = None 

    def load_opponent(self, model_path):
        """让外部训练脚本能动态替换蓝方的 AI 模型"""
        if model_path is None:
            self.opponent_model = None
        else:
            self.opponent_model = MaskablePPO.load(model_path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.winner = -1
        self.grid_type.fill(0)
        self.grid_owner.fill(-1)
        self.grid_troops.fill(0)

        for _ in range(int(self.width * self.height * 0.15)):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            self.grid_type[r, c] = 1

        p0_r, p0_c = random.randint(0, 2), random.randint(0, 2)
        self.grid_type[p0_r, p0_c] = 2
        self.grid_owner[p0_r, p0_c] = 0
        self.grid_troops[p0_r, p0_c] = 1

        p1_r, p1_c = random.randint(self.height-3, self.height-1), random.randint(self.width-3, self.width-1)
        while p1_r == p0_r and p1_c == p0_c:
            p1_r, p1_c = random.randint(self.height-3, self.height-1), random.randint(self.width-3, self.width-1)
        self.grid_type[p1_r, p1_c] = 2
        self.grid_owner[p1_r, p1_c] = 1
        self.grid_troops[p1_r, p1_c] = 1

        return self._get_obs(player_id=0), {}

    def _get_obs(self, player_id):
        obs = np.zeros((6, self.height, self.width), dtype=np.float32)
        enemy_id = 1 - player_id
        # 因为我们用了对称设计，不管 player_id 是谁，通道 0 永远是"自己"，通道 1 永远是"敌人"
        obs[0] = np.where(self.grid_owner == player_id, self.grid_troops, 0)
        obs[1] = np.where(self.grid_owner == enemy_id, self.grid_troops, 0)
        obs[2] = np.where(self.grid_owner == -1, self.grid_troops, 0)
        obs[3] = np.where(self.grid_type == 1, 1, 0)
        obs[4] = np.where((self.grid_type == 2) & (self.grid_owner == player_id), 1, 0)
        obs[5] = np.where((self.grid_type == 2) & (self.grid_owner == enemy_id), 1, 0)
        return obs

    def valid_action_mask(self, player_id=0):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_owner[r, c] == player_id and self.grid_troops[r, c] > 1:
                    base_idx = (r * self.width + c) * 4
                    if r > 0 and self.grid_type[r-1, c] != 1: mask[base_idx + 0] = True
                    if r < self.height - 1 and self.grid_type[r+1, c] != 1: mask[base_idx + 1] = True
                    if c > 0 and self.grid_type[r, c-1] != 1: mask[base_idx + 2] = True
                    if c < self.width - 1 and self.grid_type[r, c+1] != 1: mask[base_idx + 3] = True
        mask[self.SKIP_ACTION] = True
        return mask

    def _decode_action(self, action_id):
        if action_id == self.SKIP_ACTION: return None
        direction = action_id % 4
        c = (action_id // 4) % self.width
        r = action_id // (self.width * 4)
        return (r, c, direction)

    def _apply_move(self, player, action_tuple):
        if action_tuple is None: return
        r, c, direction = action_tuple
        if self.grid_owner[r, c] == player and self.grid_troops[r, c] > 1:
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid_type[nr, nc] != 1:
                moving = self.grid_troops[r, c] - 1
                self.grid_troops[r, c] = 1
                target_owner = self.grid_owner[nr, nc]
                target_troops = self.grid_troops[nr, nc]

                if target_owner == player:
                    self.grid_troops[nr, nc] += moving
                else:
                    if moving > target_troops:
                        self.grid_owner[nr, nc] = player
                        self.grid_troops[nr, nc] = moving - target_troops
                        if self.grid_type[nr, nc] == 2 and target_owner != -1:
                            self.winner = player
                    else:
                        self.grid_troops[nr, nc] = target_troops - moving

    def _opponent_turn(self):
        if self.winner != -1: return
        
        # 🌟 核心修改：如果池子里有对手模型，蓝方就用神经网络决策！
        if self.opponent_model is not None:
            obs = self._get_obs(player_id=1) # 获取蓝方视角
            mask = self.valid_action_mask(player_id=1)
            # 预测动作 (加入一定的随机性防止一直走死胡同)
            action, _ = self.opponent_model.predict(obs, action_masks=mask, deterministic=False)
            self._apply_move(player=1, action_tuple=self._decode_action(action))
        else:
            # 兜底：如果模型为空，依然用随机 Bot
            opp_mask = self.valid_action_mask(player_id=1)
            valid_actions = np.where(opp_mask)[0]
            if len(valid_actions) > 1:
                valid_actions = [a for a in valid_actions if a != self.SKIP_ACTION]
                chosen_action = random.choice(valid_actions)
            else:
                chosen_action = self.SKIP_ACTION
            self._apply_move(player=1, action_tuple=self._decode_action(chosen_action))

    def step(self, action):
        self.current_step += 1
        
        # 记录行动前的领地数量（用于混合奖励计算）
        old_tiles = np.sum(self.grid_owner == 0)
        
        self._apply_move(player=0, action_tuple=self._decode_action(action))
        self._opponent_turn()
        
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_type[r, c] == 2 and self.grid_owner[r, c] != -1:
                    self.grid_troops[r, c] += 1
        if self.current_step % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r, c] != -1:
                        self.grid_troops[r, c] += 1

        terminated = (self.winner != -1)
        truncated = (self.current_step >= self.max_steps)
        
        # ==========================================
        # 🌟 混合奖励 (Hybrid Reward) 🌟
        # ==========================================
        new_tiles = np.sum(self.grid_owner == 0)
        
        reward = -0.01 # 基础时间惩罚
        
        # 1. 微弱的占地奖励：只有原来的 1/5，不足以让它沉迷当贪吃蛇，但能鼓励出门
        reward += (new_tiles - old_tiles) * 0.01
        
        # 2. 核心斩首奖励保持不变
        if self.winner == 0:
            reward += 20.0
        elif self.winner == 1:
            reward -= 20.0

        obs = self._get_obs(player_id=0)
        info = {"action_mask": self.valid_action_mask(player_id=0)}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """保留终端打印功能"""
        print(f"\n--- Step: {self.current_step} ---")
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                troops = self.grid_troops[r, c]
                owner = self.grid_owner[r, c]
                g_type = self.grid_type[r, c]
                if g_type == 1: cell = "  M  "
                elif g_type == 2: cell = f"[{troops:2d}]"
                else: cell = f" {troops:2d} "

                if owner == 0: row_str += f"\033[91m{cell}\033[0m"
                elif owner == 1: row_str += f"\033[94m{cell}\033[0m"
                elif g_type == 1: row_str += f"\033[90m{cell}\033[0m"
                else: row_str += f"\033[37m{cell}\033[0m"
            print(row_str)
        print("-" * 20)
