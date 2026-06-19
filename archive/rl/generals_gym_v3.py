import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GeneralsEnvV3(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=12, height=12, max_steps=600):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        
        # 🌟 核心升级 1：动作空间翻倍
        # H * W * 4(方向) * 2(1=一半, 0=全部) + 1(跳过)
        self.action_space = spaces.Discrete(self.height * self.width * 8 + 1)
        self.SKIP_ACTION = self.height * self.width * 8
        
        # 🌟 核心升级 2：7通道观测矩阵
        # [己方兵, 敌方兵, 中立兵, 山脉, 城市, 己方将, 敌方将]
        self.observation_space = spaces.Box(low=0, high=10000, shape=(7, self.height, self.width), dtype=np.float32)

        self.grid_type = np.zeros((self.height, self.width), dtype=int) # 0=空, 1=山, 2=将, 3=城
        self.grid_owner = np.full((self.height, self.width), -1, dtype=int)
        self.grid_troops = np.zeros((self.height, self.width), dtype=int)
        self.winner = -1
        self.opponent_model = None 

    def load_opponent(self, model_path):
        from sb3_contrib import MaskablePPO
        if model_path is None: self.opponent_model = None
        else: self.opponent_model = MaskablePPO.load(model_path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.winner = -1
        self.grid_type.fill(0)
        self.grid_owner.fill(-1)
        self.grid_troops.fill(0)

        # 1. 生成山脉 (15%)
        for _ in range(int(self.width * self.height * 0.15)):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            self.grid_type[r, c] = 1

        # 2. 生成中立城市 (随机 4-6 个)
        num_cities = random.randint(4, 6)
        for _ in range(num_cities):
            r, c = random.randint(1, self.height-2), random.randint(1, self.width-2)
            if self.grid_type[r, c] != 1:
                self.grid_type[r, c] = 3
                # 🌟 课程学习 Phase 1: 弱化城市兵力 (10-15)，让 AI 能轻松破城
                self.grid_troops[r, c] = random.randint(10, 15)

        # 3. 双方将军
        p0_r, p0_c = random.randint(0, 3), random.randint(0, 3)
        self.grid_type[p0_r, p0_c] = 2
        self.grid_owner[p0_r, p0_c] = 0
        self.grid_troops[p0_r, p0_c] = 1

        p1_r, p1_c = random.randint(self.height-4, self.height-1), random.randint(self.width-4, self.width-1)
        while p1_r == p0_r and p1_c == p0_c:
            p1_r, p1_c = random.randint(self.height-4, self.height-1), random.randint(self.width-4, self.width-1)
        self.grid_type[p1_r, p1_c] = 2
        self.grid_owner[p1_r, p1_c] = 1
        self.grid_troops[p1_r, p1_c] = 1

        return self._get_obs(player_id=0), {}

    def _get_obs(self, player_id):
        obs = np.zeros((7, self.height, self.width), dtype=np.float32)
        enemy_id = 1 - player_id
        obs[0] = np.where(self.grid_owner == player_id, self.grid_troops, 0)
        obs[1] = np.where(self.grid_owner == enemy_id, self.grid_troops, 0)
        obs[2] = np.where(self.grid_owner == -1, self.grid_troops, 0)
        obs[3] = np.where(self.grid_type == 1, 1, 0)
        obs[4] = np.where(self.grid_type == 3, 1, 0) # 城市掩码
        obs[5] = np.where((self.grid_type == 2) & (self.grid_owner == player_id), 1, 0)
        obs[6] = np.where((self.grid_type == 2) & (self.grid_owner == enemy_id), 1, 0)
        return obs

    def valid_action_mask(self, player_id=0):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_owner[r, c] == player_id and self.grid_troops[r, c] > 1:
                    base_idx = (r * self.width + c) * 8
                    # 方向合法性
                    can_up = (r > 0 and self.grid_type[r-1, c] != 1)
                    can_down = (r < self.height - 1 and self.grid_type[r+1, c] != 1)
                    can_left = (c > 0 and self.grid_type[r, c-1] != 1)
                    can_right = (c < self.width - 1 and self.grid_type[r, c+1] != 1)
                    
                    if can_up:    mask[base_idx + 0], mask[base_idx + 1] = True, True # 0=全, 1=半
                    if can_down:  mask[base_idx + 2], mask[base_idx + 3] = True, True
                    if can_left:  mask[base_idx + 4], mask[base_idx + 5] = True, True
                    if can_right: mask[base_idx + 6], mask[base_idx + 7] = True, True
        mask[self.SKIP_ACTION] = True
        return mask

    def _decode_action(self, action_id):
        if action_id == self.SKIP_ACTION: return None
        is_half = action_id % 2
        direction = (action_id // 2) % 4
        c = (action_id // 8) % self.width
        r = action_id // (self.width * 8)
        return (r, c, direction, is_half)

    def _apply_move(self, player, action_tuple):
        if action_tuple is None: return
        r, c, direction, is_half = action_tuple
        
        if self.grid_owner[r, c] == player and self.grid_troops[r, c] > 1:
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction]
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid_type[nr, nc] != 1:
                total_avail = self.grid_troops[r, c]
                # 🌟 核心升级 3：计算移动兵力
                if is_half == 1:
                    moving = total_avail // 2
                else:
                    moving = total_avail - 1
                
                if moving == 0: return # 防止除以2后变成0
                
                self.grid_troops[r, c] -= moving
                
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
        if self.opponent_model is not None:
            obs = self._get_obs(player_id=1)
            mask = self.valid_action_mask(player_id=1)
            action, _ = self.opponent_model.predict(obs, action_masks=mask, deterministic=False)
            self._apply_move(player=1, action_tuple=self._decode_action(action))
        else:
            opp_mask = self.valid_action_mask(player_id=1)
            valid_actions = np.where(opp_mask)[0]
            if len(valid_actions) > 1:
                valid_actions = [a for a in valid_actions if a != self.SKIP_ACTION]
                self._apply_move(player=1, action_tuple=self._decode_action(random.choice(valid_actions)))

    def step(self, action):
        old_tiles = np.sum(self.grid_owner == 0)
        # 🌟 记录行动前的己方城市数量（用于破城奖励）
        old_cities = np.sum((self.grid_type == 3) & (self.grid_owner == 0))
        self.current_step += 1
        
        self._apply_move(player=0, action_tuple=self._decode_action(action))
        self._opponent_turn()
        
        # 🌟 核心升级 4：将军和城市每回合+1
        for r in range(self.height):
            for c in range(self.width):
                if (self.grid_type[r, c] == 2 or self.grid_type[r, c] == 3) and self.grid_owner[r, c] != -1:
                    self.grid_troops[r, c] += 1
                    
        # 领地每25回合+1
        if self.current_step % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r, c] != -1:
                        self.grid_troops[r, c] += 1

        terminated = (self.winner != -1)
        truncated = (self.current_step >= self.max_steps)
        
        new_tiles = np.sum(self.grid_owner == 0)
        # 🌟 记录行动后的己方城市数量
        new_cities = np.sum((self.grid_type == 3) & (self.grid_owner == 0))
        
        reward = -0.01 + (new_tiles - old_tiles) * 0.05
        
        # 🌟 破城重赏：+5.0 每座城，鼓励 AI 争夺战略资源
        if new_cities > old_cities:
            reward += 5.0 * (new_cities - old_cities)
            
        if self.winner == 0: reward += 20.0
        elif self.winner == 1: reward -= 20.0

        return self._get_obs(player_id=0), reward, terminated, truncated, {"action_mask": self.valid_action_mask(player_id=0)}

    def render(self):
        print(f"\n--- Step: {self.current_step} ---")
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                troops = self.grid_troops[r, c]
                owner = self.grid_owner[r, c]
                g_type = self.grid_type[r, c]
                if g_type == 1: cell = "  M  "
                elif g_type == 2: cell = f"[{troops:2d}]"
                elif g_type == 3: cell = f" C{str(troops).rjust(2)} "  # 城市
                else: cell = f" {troops:2d} "

                if owner == 0: row_str += f"\033[91m{cell}\033[0m"
                elif owner == 1: row_str += f"\033[94m{cell}\033[0m"
                elif g_type == 1: row_str += f"\033[90m{cell}\033[0m"
                elif g_type == 3: row_str += f"\033[93m{cell}\033[0m"  # 黄色=中立城市
                else: row_str += f"\033[37m{cell}\033[0m"
            print(row_str)
        print("-" * 20)
