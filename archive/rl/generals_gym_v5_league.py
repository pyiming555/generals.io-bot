"""
V5 混合联赛环境 (League Edition)

核心改动:
  1. 🏆 绝杀平局：600步到点按总分判胜负
  2. 🎯 微密引导：优势差变化量 × 0.001（不喧宾夺主，但让因果链可见）
  3. 🏅 斩首重赏：win=+20 / lose=-20
  4. 🎲 load_opponent("random") 切换随机 Bot

对手池:
  20% Random Bot（虐菜防遗忘）
  20% V3 种田大师（防守反击）
  30% 最新自己（上限提升）
  30% 历史版本（防遗忘）
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class GeneralsEnvV5League(gym.Env):
    """V5 混合联赛环境（绝杀平局 + 微密引导 + 随机 Bot）"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=12, height=12, max_steps=600):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        self.stalemate = False

        self.action_space = spaces.Discrete(self.height * self.width * 8 + 1)
        self.SKIP_ACTION = self.height * self.width * 8

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
        """注入对手：None/"random"=随机Bot，其他=模型加载"""
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

        # 中立城市 (4-6, 15-25)
        for _ in range(random.randint(4, 6)):
            r, c = random.randint(1, self.height - 2), random.randint(1, self.width - 2)
            if self.grid_type[r, c] != 1:
                self.grid_type[r, c] = 3
                self.grid_troops[r, c] = random.randint(15, 25)

        # 红方将军
        p0r, p0c = random.randint(0, 3), random.randint(0, 3)
        self.grid_type[p0r, p0c] = 2
        self.grid_owner[p0r, p0c] = 0
        self.grid_troops[p0r, p0c] = 1

        # 蓝方将军
        p1r = random.randint(self.height - 4, self.height - 1)
        p1c = random.randint(self.width - 4, self.width - 1)
        while p1r == p0r and p1c == p0c:
            p1r = random.randint(self.height - 4, self.height - 1)
            p1c = random.randint(self.width - 4, self.width - 1)
        self.grid_type[p1r, p1c] = 2
        self.grid_owner[p1r, p1c] = 1
        self.grid_troops[p1r, p1c] = 1

        return self._get_obs(player_id=0), {}

    def _get_obs(self, player_id):
        obs = np.zeros((7, self.height, self.width), dtype=np.float32)
        eid = 1 - player_id
        obs[0] = np.where(self.grid_owner == player_id, self.grid_troops, 0)
        obs[1] = np.where(self.grid_owner == eid, self.grid_troops, 0)
        obs[2] = np.where(self.grid_owner == -1, self.grid_troops, 0)
        obs[3] = np.where(self.grid_type == 1, 1, 0)
        obs[4] = np.where(self.grid_type == 3, 1, 0)
        obs[5] = np.where((self.grid_type == 2) & (self.grid_owner == player_id), 1, 0)
        obs[6] = np.where((self.grid_type == 2) & (self.grid_owner == eid), 1, 0)
        return obs

    def valid_action_mask(self, player_id=0):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_owner[r, c] == player_id and self.grid_troops[r, c] > 1:
                    base = (r * self.width + c) * 8
                    u = r > 0 and self.grid_type[r-1, c] != 1
                    d = r < self.height-1 and self.grid_type[r+1, c] != 1
                    l = c > 0 and self.grid_type[r, c-1] != 1
                    r_ok = c < self.width-1 and self.grid_type[r, c+1] != 1
                    if u: mask[base] = mask[base+1] = True
                    if d: mask[base+2] = mask[base+3] = True
                    if l: mask[base+4] = mask[base+5] = True
                    if r_ok: mask[base+6] = mask[base+7] = True
        mask[self.SKIP_ACTION] = True
        return mask

    def _decode_action(self, action_id):
        if action_id == self.SKIP_ACTION: return None
        return (action_id // (self.width * 8), (action_id // 8) % self.width,
                (action_id // 2) % 4, action_id % 2)

    def _apply_move(self, player, at):
        if at is None: return
        r, c, direction, half = at
        if self.grid_owner[r, c] != player or self.grid_troops[r, c] <= 1: return
        dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][direction]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < self.height and 0 <= nc < self.width): return
        if self.grid_type[nr, nc] == 1: return
        moving = self.grid_troops[r, c] // 2 if half else self.grid_troops[r, c] - 1
        if moving == 0: return
        self.grid_troops[r, c] -= moving
        to, tt = self.grid_owner[nr, nc], self.grid_troops[nr, nc]
        if to == player:
            self.grid_troops[nr, nc] += moving
        elif moving > tt:
            self.grid_owner[nr, nc] = player
            self.grid_troops[nr, nc] = moving - tt
            if self.grid_type[nr, nc] == 2 and to != -1:
                self.winner = player
        else:
            self.grid_troops[nr, nc] = tt - moving

    def _opponent_turn(self):
        if self.winner != -1: return
        if self.opponent_model is not None:
            obs = self._get_obs(player_id=1)
            mask = self.valid_action_mask(player_id=1)
            action, _ = self.opponent_model.predict(obs, action_masks=mask, deterministic=False)
            self._apply_move(player=1, at=self._decode_action(action))
        else:
            # 随机 Bot
            opp_mask = self.valid_action_mask(player_id=1)
            valid = np.where(opp_mask)[0]
            if len(valid) > 1:
                ns = [a for a in valid if a != self.SKIP_ACTION]
                if ns: self._apply_move(player=1, at=self._decode_action(random.choice(ns)))

    def _tiebreaker_score(self, player_id):
        return int(np.sum(self.grid_troops[self.grid_owner == player_id])
                   + np.sum(self.grid_owner == player_id) * 10)

    def step(self, action):
        # 记分前优势
        old_my = self._tiebreaker_score(0)
        old_en = self._tiebreaker_score(1)
        old_adv = old_my - old_en

        self.current_step += 1

        self._apply_move(player=0, at=self._decode_action(action))
        self._opponent_turn()

        # 产兵
        for r in range(self.height):
            for c in range(self.width):
                if (self.grid_type[r,c] in (2,3)) and self.grid_owner[r,c] != -1:
                    self.grid_troops[r,c] += 1
        if self.current_step % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r,c] != -1:
                        self.grid_troops[r,c] += 1

        terminated = self.winner != -1
        truncated = self.current_step >= self.max_steps

        # 记分后优势
        new_my = self._tiebreaker_score(0)
        new_en = self._tiebreaker_score(1)
        new_adv = new_my - new_en

        # 🌟 绝杀平局
        if truncated and not terminated:
            self.stalemate = True
            if new_my > new_en: self.winner = 0
            elif new_en > new_my: self.winner = 1
            terminated = True

        # ==========================================
        # 🌟 V5 混合奖励
        # ==========================================
        reward = 0.0
        reward += (new_adv - old_adv) * 0.001  # 微密引导

        if self.winner == 0:
            reward += 20.0
        elif self.winner == 1:
            reward -= 20.0

        return (self._get_obs(player_id=0), reward, terminated, truncated,
                {"action_mask": self.valid_action_mask(player_id=0)})

    def render(self):
        print(f"\n--- Step {self.current_step} ---")
        for r in range(self.height):
            row = ""
            for c in range(self.width):
                t, o, g = self.grid_troops[r,c], self.grid_owner[r,c], self.grid_type[r,c]
                cell = f"[{t:2d}]" if g==2 else f" C{t:2d} " if g==3 else f" {t:2d} " if g!=1 else "  M  "
                ccode = {0:'\033[91m',1:'\033[94m'}.get(o,'\033[90m' if g==1 else '\033[93m' if g==3 else '\033[37m')
                row += f"{ccode}{cell}\033[0m"
            print(row)
        print("-"*20)
