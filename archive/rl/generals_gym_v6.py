"""
V6 绝杀 + 斩首重赏环境 (Assassination & Tiebreaker)

核心改动 vs V5:
  1. 🎯 斩首重赏：斩首胜利 = +100 / 绝杀胜利 = +20（奖励断层制造绝对落差）
  2. ⏱️ 步数惩罚：每步 -0.01（鼓励速战速决）
  3. 📏 可配置 max_steps (Phase A=450, Phase B=300)
  4. 🏆 保留绝杀平局 + 微密引导 (advantage_diff × 0.001)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class GeneralsEnvV6(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=12, height=12, max_steps=450,
                 decap_reward=50.0, tiebreaker_reward=20.0,
                 step_penalty=0.01, adv_scale=0.001):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.decap_reward = decap_reward          # 斩首重赏 (Phase A=50, Phase B=100)
        self.tiebreaker_reward = tiebreaker_reward  # 绝杀保底
        self.step_penalty = step_penalty            # 每步惩罚
        self.adv_scale = adv_scale                  # 微密引导系数
        self.current_step = 0
        self.stalemate = False

        # 动作空间
        self.action_space = spaces.Discrete(self.height * self.width * 8 + 1)
        self.SKIP_ACTION = self.height * self.width * 8

        # 观测空间
        self.observation_space = spaces.Box(
            low=0, high=10000, shape=(7, self.height, self.width), dtype=np.float32
        )

        self.grid_type = np.zeros((self.height, self.width), dtype=int)
        self.grid_owner = np.full((self.height, self.width), -1, dtype=int)
        self.grid_troops = np.zeros((self.height, self.width), dtype=int)
        self.winner = -1

        # Self-Play 对手模型
        self.opponent_model = None

    def load_opponent(self, model_path):
        if model_path is None or model_path == "random":
            self.opponent_model = None
        else:
            from sb3_contrib import MaskablePPO
            self.opponent_model = MaskablePPO.load(model_path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.winner = -1
        self.stalemate = False
        self.grid_type.fill(0)
        self.grid_owner.fill(-1)
        self.grid_troops.fill(0)

        # 山脉 (15%)
        for _ in range(int(self.width * self.height * 0.15)):
            r, c = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            self.grid_type[r, c] = 1

        # 城市 (4-6, 15-25兵)
        num_cities = random.randint(4, 6)
        for _ in range(num_cities):
            r, c = random.randint(1, self.height - 2), random.randint(1, self.width - 2)
            if self.grid_type[r, c] != 1:
                self.grid_type[r, c] = 3
                self.grid_troops[r, c] = random.randint(15, 25)

        # 红方 (左上)
        p0_r, p0_c = random.randint(0, 3), random.randint(0, 3)
        self.grid_type[p0_r, p0_c] = 2
        self.grid_owner[p0_r, p0_c] = 0
        self.grid_troops[p0_r, p0_c] = 1

        # 蓝方 (右下)
        p1_r, p1_c = (
            random.randint(self.height - 4, self.height - 1),
            random.randint(self.width - 4, self.width - 1),
        )
        while p1_r == p0_r and p1_c == p0_c:
            p1_r, p1_c = (
                random.randint(self.height - 4, self.height - 1),
                random.randint(self.width - 4, self.width - 1),
            )
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
        obs[4] = np.where(self.grid_type == 3, 1, 0)
        obs[5] = np.where((self.grid_type == 2) & (self.grid_owner == player_id), 1, 0)
        obs[6] = np.where((self.grid_type == 2) & (self.grid_owner == enemy_id), 1, 0)
        return obs

    def valid_action_mask(self, player_id=0):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_owner[r, c] == player_id and self.grid_troops[r, c] > 1:
                    base_idx = (r * self.width + c) * 8
                    can_up = r > 0 and self.grid_type[r - 1, c] != 1
                    can_down = r < self.height - 1 and self.grid_type[r + 1, c] != 1
                    can_left = c > 0 and self.grid_type[r, c - 1] != 1
                    can_right = c < self.width - 1 and self.grid_type[r, c + 1] != 1
                    if can_up:
                        mask[base_idx + 0] = mask[base_idx + 1] = True
                    if can_down:
                        mask[base_idx + 2] = mask[base_idx + 3] = True
                    if can_left:
                        mask[base_idx + 4] = mask[base_idx + 5] = True
                    if can_right:
                        mask[base_idx + 6] = mask[base_idx + 7] = True
        mask[self.SKIP_ACTION] = True
        return mask

    def _decode_action(self, action_id):
        if action_id == self.SKIP_ACTION:
            return None
        is_half = action_id % 2
        direction = (action_id // 2) % 4
        c = (action_id // 8) % self.width
        r = action_id // (self.width * 8)
        return (r, c, direction, is_half)

    def _apply_move(self, player, action_tuple):
        if action_tuple is None:
            return
        r, c, direction, is_half = action_tuple
        if self.grid_owner[r, c] != player or self.grid_troops[r, c] <= 1:
            return
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return
        if self.grid_type[nr, nc] == 1:
            return
        total_avail = self.grid_troops[r, c]
        moving = total_avail // 2 if is_half == 1 else total_avail - 1
        if moving == 0:
            return
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
        if self.winner != -1:
            return
        if self.opponent_model is not None:
            obs = self._get_obs(player_id=1)
            mask = self.valid_action_mask(player_id=1)
            action, _ = self.opponent_model.predict(obs, action_masks=mask, deterministic=False)
            self._apply_move(player=1, action_tuple=self._decode_action(action))
        else:
            opp_mask = self.valid_action_mask(player_id=1)
            valid_actions = np.where(opp_mask)[0]
            if len(valid_actions) > 1:
                non_skip = [a for a in valid_actions if a != self.SKIP_ACTION]
                if non_skip:
                    self._apply_move(player=1, action_tuple=self._decode_action(random.choice(non_skip)))

    def _tiebreaker_score(self, player_id):
        troops = np.sum(self.grid_troops[self.grid_owner == player_id])
        tiles = np.sum(self.grid_owner == player_id)
        return troops + tiles * 10

    def step(self, action):
        # === 1. 计算行动前优势 ===
        old_my = np.sum(self.grid_troops[self.grid_owner == 0]) + np.sum(self.grid_owner == 0) * 10
        old_enemy = np.sum(self.grid_troops[self.grid_owner == 1]) + np.sum(self.grid_owner == 1) * 10
        old_adv = old_my - old_enemy

        self.current_step += 1
        is_decap = False

        # === 2. 行动 ===
        self._apply_move(player=0, action_tuple=self._decode_action(action))
        self._opponent_turn()

        # === 3. 产兵 ===
        for r in range(self.height):
            for c in range(self.width):
                if (self.grid_type[r, c] == 2 or self.grid_type[r, c] == 3) and self.grid_owner[r, c] != -1:
                    self.grid_troops[r, c] += 1
        if self.current_step % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r, c] != -1:
                        self.grid_troops[r, c] += 1

        # === 4. 检查和结算 ===
        terminated = self.winner != -1
        truncated = self.current_step >= self.max_steps

        if terminated:
            is_decap = True  # 斩首成功
        elif truncated:
            self.stalemate = True
            my_score = np.sum(self.grid_troops[self.grid_owner == 0]) + np.sum(self.grid_owner == 0) * 10
            enemy_score = np.sum(self.grid_troops[self.grid_owner == 1]) + np.sum(self.grid_owner == 1) * 10
            if my_score > enemy_score:
                self.winner = 0
            elif enemy_score > my_score:
                self.winner = 1

        # === 5. 计算优势变化 ===
        new_my = np.sum(self.grid_troops[self.grid_owner == 0]) + np.sum(self.grid_owner == 0) * 10
        new_enemy = np.sum(self.grid_troops[self.grid_owner == 1]) + np.sum(self.grid_owner == 1) * 10
        new_adv = new_my - new_enemy

        # === 6. 奖励 ===
        reward = 0.0
        reward += (new_adv - old_adv) * self.adv_scale   # 微密引导
        reward -= self.step_penalty                       # 步数惩罚

        if self.winner == 0:
            if is_decap:
                reward += self.decap_reward               # 🎯 斩首重赏
            else:
                reward += self.tiebreaker_reward           # 🏆 绝杀保底
        elif self.winner == 1:
            if is_decap:
                reward -= self.decap_reward
            else:
                reward -= self.tiebreaker_reward

        return (
            self._get_obs(player_id=0),
            reward,
            terminated,
            truncated,
            {"action_mask": self.valid_action_mask(player_id=0)},
        )

    def render(self):
        print(f"\n--- Step: {self.current_step} ---")
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                troops = self.grid_troops[r, c]
                owner = self.grid_owner[r, c]
                g_type = self.grid_type[r, c]
                if g_type == 1:
                    cell = "  M  "
                elif g_type == 2:
                    cell = f"[{troops:2d}]"
                elif g_type == 3:
                    cell = f" C{troops:2d} "
                else:
                    cell = f" {troops:2d} "
                if owner == 0:
                    row_str += f"\033[91m{cell}\033[0m"
                elif owner == 1:
                    row_str += f"\033[94m{cell}\033[0m"
                elif g_type == 1:
                    row_str += f"\033[90m{cell}\033[0m"
                elif g_type == 3:
                    row_str += f"\033[93m{cell}\033[0m"
                else:
                    row_str += f"\033[37m{cell}\033[0m"
            print(row_str)
        print("-" * 20)
