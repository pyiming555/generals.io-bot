import numpy as np
import random
import time

class SimpleGeneralsEnv:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        # 地形: 0=空地, 1=山脉, 2=将军塔(主基地)
        self.grid_type = np.zeros((height, width), dtype=int)
        # 归属权: -1=中立, 0=玩家0(红), 1=玩家1(蓝)
        self.grid_owner = np.full((height, width), -1, dtype=int)
        # 兵力值
        self.grid_troops = np.zeros((height, width), dtype=int)
        
        self.turn_count = 0
        self.winner = -1

    def reset(self):
        """初始化地图"""
        self.grid_type.fill(0)
        self.grid_owner.fill(-1)
        self.grid_troops.fill(0)
        self.turn_count = 0
        self.winner = -1

        # 随机生成一些山脉 (大概占地图15%)
        for _ in range(int(self.width * self.height * 0.15)):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            self.grid_type[r, c] = 1

        # 设置玩家0的将军塔 (左上角区域)
        p0_r, p0_c = random.randint(0, 2), random.randint(0, 2)
        self.grid_type[p0_r, p0_c] = 2
        self.grid_owner[p0_r, p0_c] = 0
        self.grid_troops[p0_r, p0_c] = 1

        # 设置玩家1的将军塔 (右下角区域)
        p1_r, p1_c = random.randint(self.height-3, self.height-1), random.randint(self.width-3, self.width-1)
        # 确保不重叠
        while p1_r == p0_r and p1_c == p0_c:
            p1_r, p1_c = random.randint(self.height-3, self.height-1), random.randint(self.width-3, self.width-1)
        
        self.grid_type[p1_r, p1_c] = 2
        self.grid_owner[p1_r, p1_c] = 1
        self.grid_troops[p1_r, p1_c] = 1

        return self._get_obs()

    def _get_obs(self):
        """获取当前状态（暂时返回所有矩阵）"""
        return {
            "type": self.grid_type.copy(),
            "owner": self.grid_owner.copy(),
            "troops": self.grid_troops.copy()
        }

    def step(self, player, action):
        """
        执行玩家的动作并更新游戏状态。
        action 格式: (from_r, from_c, direction)
        direction: 0=上, 1=下, 2=左, 3=右
        如果是 None，表示该玩家本回合不操作。
        """
        if self.winner != -1:
            return self._get_obs(), 0, True, {"msg": "Game Over"}

        # 1. 处理移动与战斗逻辑
        if action is not None:
            r, c, direction = action
            
            # 检查出发地是否合法 (属于自己，且兵力>1)
            if self.grid_owner[r, c] == player and self.grid_troops[r, c] > 1:
                # 计算目标位置
                dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction]
                nr, nc = r + dr, c + dc
                
                # 检查目标是否在边界内，且不是山脉
                if 0 <= nr < self.height and 0 <= nc < self.width and self.grid_type[nr, nc] != 1:
                    moving_troops = self.grid_troops[r, c] - 1
                    self.grid_troops[r, c] = 1 # 原地留下1个兵
                    
                    target_owner = self.grid_owner[nr, nc]
                    target_troops = self.grid_troops[nr, nc]

                    if target_owner == player:
                        # 增援自己的地盘
                        self.grid_troops[nr, nc] += moving_troops
                    else:
                        # 攻击敌方或中立地盘
                        if moving_troops > target_troops:
                            # 攻占成功
                            self.grid_owner[nr, nc] = player
                            self.grid_troops[nr, nc] = moving_troops - target_troops
                            
                            # 检查是否推平了对面的主基地
                            if self.grid_type[nr, nc] == 2 and target_owner != -1:
                                self.winner = player
                        else:
                            # 攻占失败，消耗对方兵力
                            self.grid_troops[nr, nc] = target_troops - moving_troops

        # 2. 经济/产兵逻辑 (Tick)
        self.turn_count += 1
        # 每1回合，将军塔兵力+1
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_type[r, c] == 2 and self.grid_owner[r, c] != -1:
                    self.grid_troops[r, c] += 1
        
        # 每25回合，所有属于玩家的领地兵力+1
        if self.turn_count % 25 == 0:
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid_owner[r, c] != -1:
                        self.grid_troops[r, c] += 1

        done = (self.winner != -1)
        reward = 1 if self.winner == player else (-1 if done else 0)
        
        return self._get_obs(), reward, done, {}

    def render(self):
        """在终端打印彩色地图"""
        print(f"\n--- Turn: {self.turn_count} ---")
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                troops = self.grid_troops[r, c]
                owner = self.grid_owner[r, c]
                g_type = self.grid_type[r, c]

                # 格式化格子字符串
                if g_type == 1:
                    cell = "  M  " # 山脉
                elif g_type == 2:
                    cell = f"[{troops:2d}]" # 将军塔
                else:
                    cell = f" {troops:2d} " # 空地

                # 添加终端颜色
                if owner == 0:
                    row_str += f"\033[91m{cell}\033[0m" # 红色 (玩家0)
                elif owner == 1:
                    row_str += f"\033[94m{cell}\033[0m" # 蓝色 (玩家1)
                elif g_type == 1:
                    row_str += f"\033[90m{cell}\033[0m" # 灰色 (山脉)
                else:
                    row_str += f"\033[37m{cell}\033[0m" # 白色 (中立空地)
            print(row_str)
        print("-" * 20)


def random_bot(env, player_id):
    """
    一个简单的随机机器人：
    找到自己所有兵力 > 1 的格子，随机挑一个，随机选一个方向移动。
    """
    valid_moves = []
    # 遍历地图，找到可以移动的格子
    for r in range(env.height):
        for c in range(env.width):
            if env.grid_owner[r, c] == player_id and env.grid_troops[r, c] > 1:
                # 生成4个方向的可能动作: 0=上, 1=下, 2=左, 3=右
                for direction in range(4):
                    valid_moves.append((r, c, direction))
    
    # 如果有可以移动的选项，随机返回一个；否则返回 None
    if valid_moves:
        return random.choice(valid_moves)
    return None

if __name__ == "__main__":
    # 创建 8x8 的微型地图
    env = SimpleGeneralsEnv(width=8, height=8)
    obs = env.reset()
    env.render()
    
    done = False
    
    # 开始游戏循环
    while not done:
        # 玩家 0 (红色) 动作
        action_p0 = random_bot(env, 0)
        obs, reward, done, info = env.step(player=0, action=action_p0)
        
        if done: break
        
        # 玩家 1 (蓝色) 动作
        action_p1 = random_bot(env, 1)
        obs, reward, done, info = env.step(player=1, action=action_p1)
        
        # 每 5 个回合刷新一次画面，太快了看不清
        if env.turn_count % 5 == 0:
            env.render()
            time.sleep(0.2) # 暂停0.2秒以便观察

    # 游戏结束
    env.render()
    print(f"\nGame Over! Winner is Player {env.winner} " 
          f"({'Red' if env.winner == 0 else 'Blue'})")
