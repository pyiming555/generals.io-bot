import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

class GeneralsRLGymEnv(gym.Env):
    """
    符合 Gymnasium 标准的 generals.io 单智能体 RL 环境
    玩家 0 是 RL Agent，玩家 1 是内置的固定策略对手（当前为随机 Bot）
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, width=8, height=8, max_steps=500):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        
        # --- 1. 定义动作空间 (Action Space) ---
        # 每个格子有 4 个方向 (0=上, 1=下, 2=左, 3=右)
        # 动作总数 = H * W * 4 + 1 (最后一个动作代表 "什么都不做/跳过")
        self.action_space = spaces.Discrete(self.height * self.width * 4 + 1)
        self.SKIP_ACTION = self.height * self.width * 4

        # --- 2. 定义观测空间 (Observation Space) ---
        # 我们使用 6 个通道 (Channels) 的 2D 矩阵，形状为 (6, H, W)
        # C0: 己方兵力 (非己方为0)
        # C1: 敌方兵力 (非敌方为0)
        # C2: 中立兵力 (非中立为0)
        # C3: 山脉掩码 (1=山脉, 0=空地)
        # C4: 己方将军塔 (1=是, 0=否)
        # C5: 敌方将军塔 (1=是, 0=否)
        self.observation_space = spaces.Box(
            low=0, high=10000, # 兵力上限假设为 10000
            shape=(6, self.height, self.width),
            dtype=np.float32
        )

        # 内部状态（与之前类似）
        self.grid_type = np.zeros((self.height, self.width), dtype=int)
        self.grid_owner = np.full((self.height, self.width), -1, dtype=int)
        self.grid_troops = np.zeros((self.height, self.width), dtype=int)
        self.winner = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.winner = -1
        
        # 初始化地图（与之前一样）
        self.grid_type.fill(0)
        self.grid_owner.fill(-1)
        self.grid_troops.fill(0)

        for _ in range(int(self.width * self.height * 0.15)):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            self.grid_type[r, c] = 1 # 山脉

        # 玩家 0 (RL Agent)
        p0_r, p0_c = random.randint(0, 2), random.randint(0, 2)
        self.grid_type[p0_r, p0_c] = 2
        self.grid_owner[p0_r, p0_c] = 0
        self.grid_troops[p0_r, p0_c] = 1

        # 玩家 1 (内置对手)
        p1_r, p1_c = random.randint(self.height-3, self.height-1), random.randint(self.width-3, self.width-1)
        while p1_r == p0_r and p1_c == p0_c:
            p1_r, p1_c = random.randint(self.height-3, self.height-1), random.randint(self.width-3, self.width-1)
        self.grid_type[p1_r, p1_c] = 2
        self.grid_owner[p1_r, p1_c] = 1
        self.grid_troops[p1_r, p1_c] = 1

        return self._get_obs(player_id=0), {}

    def _get_obs(self, player_id):
        """将内部状态转化为 (6, H, W) 的多通道特征图"""
        obs = np.zeros((6, self.height, self.width), dtype=np.float32)
        enemy_id = 1 - player_id
        
        # C0: 己方兵力
        obs[0] = np.where(self.grid_owner == player_id, self.grid_troops, 0)
        # C1: 敌方兵力
        obs[1] = np.where(self.grid_owner == enemy_id, self.grid_troops, 0)
        # C2: 中立兵力
        obs[2] = np.where(self.grid_owner == -1, self.grid_troops, 0)
        # C3: 山脉
        obs[3] = np.where(self.grid_type == 1, 1, 0)
        # C4: 己方将军塔
        obs[4] = np.where((self.grid_type == 2) & (self.grid_owner == player_id), 1, 0)
        # C5: 敌方将军塔
        obs[5] = np.where((self.grid_type == 2) & (self.grid_owner == enemy_id), 1, 0)
        
        return obs

    def valid_action_mask(self, player_id=0):
        """
        ***极其重要***: 返回一个布尔数组，长度等于 action_space.n
        True 表示动作合法，False 表示动作非法。RL 算法依赖这个避免撞墙。
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        for r in range(self.height):
            for c in range(self.width):
                # 只有属于自己且兵力 > 1 的格子才能发兵
                if self.grid_owner[r, c] == player_id and self.grid_troops[r, c] > 1:
                    base_idx = (r * self.width + c) * 4
                    
                    # 检查 4 个方向 (0=上, 1=下, 2=左, 3=右)
                    if r > 0 and self.grid_type[r-1, c] != 1:              # 上
                        mask[base_idx + 0] = True
                    if r < self.height - 1 and self.grid_type[r+1, c] != 1: # 下
                        mask[base_idx + 1] = True
                    if c > 0 and self.grid_type[r, c-1] != 1:              # 左
                        mask[base_idx + 2] = True
                    if c < self.width - 1 and self.grid_type[r, c+1] != 1:  # 右
                        mask[base_idx + 3] = True
                        
        mask[self.SKIP_ACTION] = True # "跳过"永远合法
        return mask

    def _decode_action(self, action_id):
        """将一维动作 ID 还原为 (r, c, direction)"""
        if action_id == self.SKIP_ACTION:
            return None
        direction = action_id % 4
        c = (action_id // 4) % self.width
        r = action_id // (self.width * 4)
        return (r, c, direction)

    def _apply_move(self, player, action_tuple):
        """执行移动和战斗（核心物理引擎逻辑）"""
        if action_tuple is None: return
        r, c, direction = action_tuple
        
        if self.grid_owner[r, c] == player and self.grid_troops[r, c] > 1:
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction]
            nr, nc = r + dr, c + dc
            
            # 再次校验合法性（防止意外）
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid_type[nr, nc] != 1:
                moving = self.grid_troops[r, c] - 1
                self.grid_troops[r, c] = 1
                
                target_owner = self.grid_owner[nr, nc]
                target_troops = self.grid_troops[nr, nc]

                if target_owner == player:
                    self.grid_troops[nr, nc] += moving
                else:
                    if moving > target_troops: # 攻占
                        self.grid_owner[nr, nc] = player
                        self.grid_troops[nr, nc] = moving - target_troops
                        if self.grid_type[nr, nc] == 2 and target_owner != -1:
                            self.winner = player
                    else: # 消耗
                        self.grid_troops[nr, nc] = target_troops - moving

    def _opponent_turn(self):
        """玩家 1 (内置对手) 的回合: 当前使用随机策略"""
        if self.winner != -1: return
        # 获取对手的 action mask
        opp_mask = self.valid_action_mask(player_id=1)
        valid_actions = np.where(opp_mask)[0]
        if len(valid_actions) > 1:
            # 排除纯 Skip，尽量让对手移动
            valid_actions = [a for a in valid_actions if a != self.SKIP_ACTION]
            chosen_action = random.choice(valid_actions)
        else:
            chosen_action = self.SKIP_ACTION
        
        self._apply_move(player=1, action_tuple=self._decode_action(chosen_action))

    def step(self, action):
        """RL Agent 走一步，然后对手走一步，并计算密集奖励"""
        # 记录行动前的领地数量
        old_tiles = np.sum(self.grid_owner == 0)
        
        self.current_step += 1
        
        # 1. 玩家 0 (RL Agent) 行动
        self._apply_move(player=0, action_tuple=self._decode_action(action))
        
        # 2. 玩家 1 (Opponent) 行动
        self._opponent_turn()
        
        # 3. 经济/产兵逻辑 (Tick)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_type[r, c] == 2 and self.grid_owner[r, c] != -1:
                    self.grid_troops[r, c] += 1
        if self.current_step % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r, c] != -1:
                        self.grid_troops[r, c] += 1

        # 记录行动后的领地数量
        new_tiles = np.sum(self.grid_owner == 0)

        # 4. 判断游戏结束状态
        terminated = (self.winner != -1)
        truncated = (self.current_step >= self.max_steps)
        
        # ==========================================
        # 🌟 核心：密集奖励设计 (Reward Shaping) 🌟
        # ==========================================
        reward = 0.0
        
        # a. 占地奖励：每多占领一块地 +0.05，丢掉一块地 -0.05
        # 这会驱使 AI 在前期疯狂往外扩张（贪吃蛇行为）
        reward += (new_tiles - old_tiles) * 0.05
        
        # b. 时间惩罚：每回合扣一点分，逼迫 AI 尽快结束战斗，不要挂机
        reward -= 0.01 
        
        # c. 终极奖励：赢了拿大头，输了重罚
        if self.winner == 0:
            reward += 10.0
        elif self.winner == 1:
            reward -= 10.0
        
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


if __name__ == "__main__":
    # 直接运行此文件时仅做环境测试（不再有游戏循环，测试脚本请用 train.py）
    env = GeneralsRLGymEnv(width=8, height=8)
    obs, info = env.reset()
    env.render()
    print(f"Obs shape: {obs.shape}, Action space: {env.action_space.n}")
    print("generals_gym.py 已成功加载。运行 python train.py 开始训练。")
