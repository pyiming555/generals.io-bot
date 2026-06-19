import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class GeneralsEnvV3SelfPlay(gym.Env):
    """
    V3 完整自对弈环境 (Self-Play Edition)
    
    继承 V3 物理引擎（12x12, 分兵, 城市, 将军产兵），
    但针对 Phase 2 自对弈做了两项调整：
      1. 占地奖励从 +0.05 降回 +0.01 —— 种田发不了大财
      2. 城市兵力从 10~15 上调到 15~25 —— 课程学习逐步逼近真实 40
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=12, height=12, max_steps=600):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0

        # V3 动作空间：H*W*4(方向)*2(全/半) + 1(跳过)
        self.action_space = spaces.Discrete(self.height * self.width * 8 + 1)
        self.SKIP_ACTION = self.height * self.width * 8

        # V3 观测空间：7 通道 [己方兵, 敌方兵, 中立兵, 山脉, 城市, 己方将, 敌方将]
        self.observation_space = spaces.Box(
            low=0, high=10000, shape=(7, self.height, self.width), dtype=np.float32
        )

        self.grid_type = np.zeros((self.height, self.width), dtype=int)   # 0=空地, 1=山脉, 2=将军, 3=城市
        self.grid_owner = np.full((self.height, self.width), -1, dtype=int)
        self.grid_troops = np.zeros((self.height, self.width), dtype=int)
        self.winner = -1

        # Self-Play 对手模型（由外部训练脚本注入）
        self.opponent_model = None

    def load_opponent(self, model_path):
        """由 LeagueTrainingCallback 调用来动态切换蓝方的 AI"""
        from sb3_contrib import MaskablePPO
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

        # 1. 山脉 (15%)
        for _ in range(int(self.width * self.height * 0.15)):
            r, c = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            self.grid_type[r, c] = 1

        # 2. 中立城市 (4-6 个)  — Phase 2: 15-25 兵力
        num_cities = random.randint(4, 6)
        for _ in range(num_cities):
            r, c = random.randint(1, self.height - 2), random.randint(1, self.width - 2)
            if self.grid_type[r, c] != 1:
                self.grid_type[r, c] = 3
                self.grid_troops[r, c] = random.randint(15, 25)

        # 3. 红方将军 (左上)
        p0_r, p0_c = random.randint(0, 3), random.randint(0, 3)
        self.grid_type[p0_r, p0_c] = 2
        self.grid_owner[p0_r, p0_c] = 0
        self.grid_troops[p0_r, p0_c] = 1

        # 4. 蓝方将军 (右下)
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
        """生成 7 通道观测（红蓝对称）"""
        obs = np.zeros((7, self.height, self.width), dtype=np.float32)
        enemy_id = 1 - player_id
        obs[0] = np.where(self.grid_owner == player_id, self.grid_troops, 0)  # 己方兵力
        obs[1] = np.where(self.grid_owner == enemy_id, self.grid_troops, 0)   # 敌方兵力
        obs[2] = np.where(self.grid_owner == -1, self.grid_troops, 0)         # 中立兵力
        obs[3] = np.where(self.grid_type == 1, 1, 0)                          # 山脉掩码
        obs[4] = np.where(self.grid_type == 3, 1, 0)                          # 城市掩码
        obs[5] = np.where((self.grid_type == 2) & (self.grid_owner == player_id), 1, 0)  # 己方将军
        obs[6] = np.where((self.grid_type == 2) & (self.grid_owner == enemy_id), 1, 0)   # 敌方将军
        return obs

    def valid_action_mask(self, player_id=0):
        """生成动作掩码，禁止无效/自杀动作"""
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
        """将一维动作 ID 解码为 (r, c, direction, is_half) 或 None（跳过）"""
        if action_id == self.SKIP_ACTION:
            return None
        is_half = action_id % 2
        direction = (action_id // 2) % 4
        c = (action_id // 8) % self.width
        r = action_id // (self.width * 8)
        return (r, c, direction, is_half)

    def _apply_move(self, player, action_tuple):
        """执行一步移动：分兵、战斗、斩首判定"""
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
            # 移动到自己地盘
            self.grid_troops[nr, nc] += moving
        else:
            if moving > target_troops:
                # 攻占成功
                self.grid_owner[nr, nc] = player
                self.grid_troops[nr, nc] = moving - target_troops
                if self.grid_type[nr, nc] == 2 and target_owner != -1:
                    self.winner = player
            else:
                # 进攻失败
                self.grid_troops[nr, nc] = target_troops - moving

    def _opponent_turn(self):
        """让蓝方 AI 走一步（从加载的模型或随机 bot）"""
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
                valid_actions = [a for a in valid_actions if a != self.SKIP_ACTION]
                if valid_actions:
                    self._apply_move(player=1, action_tuple=self._decode_action(random.choice(valid_actions)))

    def step(self, action):
        old_tiles = np.sum(self.grid_owner == 0)
        old_cities = np.sum((self.grid_type == 3) & (self.grid_owner == 0))

        self.current_step += 1

        # 红方行动
        self._apply_move(player=0, action_tuple=self._decode_action(action))
        # 蓝方行动
        self._opponent_turn()

        # 将军和城市每回合 +1 兵力
        for r in range(self.height):
            for c in range(self.width):
                if (self.grid_type[r, c] == 2 or self.grid_type[r, c] == 3) and self.grid_owner[r, c] != -1:
                    self.grid_troops[r, c] += 1

        # 领地兵每 25 回合 +1
        if self.current_step % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r, c] != -1:
                        self.grid_troops[r, c] += 1

        terminated = self.winner != -1
        truncated = self.current_step >= self.max_steps

        new_tiles = np.sum(self.grid_owner == 0)
        new_cities = np.sum((self.grid_type == 3) & (self.grid_owner == 0))

        # === Phase 2 奖励设计 ===
        reward = -0.01  # 基础时间惩罚
        reward += (new_tiles - old_tiles) * 0.01  # 种田奖励大幅削减！

        if new_cities > old_cities:
            reward += 5.0 * (new_cities - old_cities)  # 破城重赏不变

        if self.winner == 0:
            reward += 20.0
        elif self.winner == 1:
            reward -= 20.0

        return (
            self._get_obs(player_id=0),
            reward,
            terminated,
            truncated,
            {"action_mask": self.valid_action_mask(player_id=0)},
        )

    def render(self):
        """终端彩色渲染"""
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
