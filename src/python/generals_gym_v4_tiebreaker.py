"""
V4 绝杀平局环境 (Tie-Breaker Edition)

核心改动 vs V3 Self-Play:
  1. 🏆 绝杀平局：600步到点时，按兵力+领地*10算总分，高者直接判胜
  2. 🧹 纯净奖励：完全抛弃密集奖励（占地、破城），只有 winner +20 / -20
  3. 🎲 对手为随机 Bot（无模型加载）

原理：
  AI 无法通过"种田刷分"获得奖励，因为奖励只在游戏结束时发放。
  但 AI 又必须保证兵力+领地在 600步时比对手多，否则就是输。
  被迫在中期开始主动进攻以拉开差距——打破和平。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class GeneralsEnvV4TieBreaker(gym.Env):
    """V4 绝杀平局 Self-Play 环境（纯净奖励）"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=12, height=12, max_steps=600):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        self.stalemate = False  # 是否由绝杀机制判定

        # V3 动作空间
        self.action_space = spaces.Discrete(self.height * self.width * 8 + 1)
        self.SKIP_ACTION = self.height * self.width * 8

        # V3 观测空间
        self.observation_space = spaces.Box(
            low=0, high=10000, shape=(7, self.height, self.width), dtype=np.float32
        )

        self.grid_type = np.zeros((self.height, self.width), dtype=int)
        self.grid_owner = np.full((self.height, self.width), -1, dtype=int)
        self.grid_troops = np.zeros((self.height, self.width), dtype=int)
        self.winner = -1
        self.stalemate = False  # 是否由绝杀机制判定

        # Self-Play 对手模型（由 LeagueCallback 注入）
        self.opponent_model = None

    def load_opponent(self, model_path):
        from sb3_contrib import MaskablePPO
        if model_path is None:
            self.opponent_model = None
        else:
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

        # 中立城市 (4-6 个, 15-25 兵力)
        num_cities = random.randint(4, 6)
        for _ in range(num_cities):
            r, c = random.randint(1, self.height - 2), random.randint(1, self.width - 2)
            if self.grid_type[r, c] != 1:
                self.grid_type[r, c] = 3
                self.grid_troops[r, c] = random.randint(15, 25)

        # 红方将军 (左上)
        p0_r, p0_c = random.randint(0, 3), random.randint(0, 3)
        self.grid_type[p0_r, p0_c] = 2
        self.grid_owner[p0_r, p0_c] = 0
        self.grid_troops[p0_r, p0_c] = 1

        # 蓝方将军 (右下)
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
        """蓝方走一步（模型对手或随机 Bot）"""
        if self.winner != -1:
            return
        if self.opponent_model is not None:
            obs = self._get_obs(player_id=1)
            mask = self.valid_action_mask(player_id=1)
            action, _ = self.opponent_model.predict(obs, action_masks=mask, deterministic=False)
            self._apply_move(player=1, action_tuple=self._decode_action(action))
        else:
            # 随机 Bot
            opp_mask = self.valid_action_mask(player_id=1)
            valid_actions = np.where(opp_mask)[0]
            if len(valid_actions) > 1:
                non_skip = [a for a in valid_actions if a != self.SKIP_ACTION]
                if non_skip:
                    self._apply_move(player=1, action_tuple=self._decode_action(random.choice(non_skip)))

    def _tiebreaker_score(self, player_id):
        """结算时刻的总分 = 兵力总值 + 领地数*10"""
        troops = np.sum(self.grid_troops[self.grid_owner == player_id])
        tiles = np.sum(self.grid_owner == player_id)
        return troops + tiles * 10

    def step(self, action):
        self.current_step += 1

        # 红方行动
        self._apply_move(player=0, action_tuple=self._decode_action(action))
        # 蓝方行动
        self._opponent_turn()

        # 将军和城市产兵
        for r in range(self.height):
            for c in range(self.width):
                if (self.grid_type[r, c] == 2 or self.grid_type[r, c] == 3) and self.grid_owner[r, c] != -1:
                    self.grid_troops[r, c] += 1

        # 领地产兵
        if self.current_step % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r, c] != -1:
                        self.grid_troops[r, c] += 1

        terminated = self.winner != -1
        truncated = self.current_step >= self.max_steps

        # ============================================================
        # 🌟 绝杀平局机制 (Tie-Breaker) 🌟
        # ============================================================
        if truncated and not terminated:
            self.stalemate = True
            p0_score = self._tiebreaker_score(0)
            p1_score = self._tiebreaker_score(1)
            if p0_score > p1_score:
                self.winner = 0
            elif p1_score > p0_score:
                self.winner = 1
            # 绝对平局（极小概率）— winner remains -1

        # ============================================================
        # 🌟 返璞归真的纯净奖励 (Pure Sparse Reward) 🌟
        # ============================================================
        reward = 0.0
        if self.winner == 0:
            reward = 20.0
        elif self.winner == 1:
            reward = -20.0
        # 绝对平局或游戏中 = 0 分

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
