"""
script_bot.py — 启发式脚本 AI (扩张-集结-突击三段式)

设计哲学:
  1. 扩张优先: 疯狂占领无主空地，增加造兵基数
  2. 前线引力: 后方兵力自动流向前线，严惩龟缩
  3. 精准击杀: 兵力碾压时才进攻，不打亏本仗
  4. 迷雾探索: 主动向未知区域渗透

用法:
  from script_bot import ScriptBot
  bot = ScriptBot(player_id=1, grid_size=12)
  action = bot.get_action(state_numpy)
  # action: (sy, sx, ny, nx, is_half) 或 None (跳过)
"""

import numpy as np


class ScriptBot:
    """启发式贪心脚本 AI"""

    def __init__(self, player_id, grid_size=12):
        self.player_id = player_id
        self.enemy_id = 1 - player_id
        self.grid_size = grid_size
        # 方向: 上、下、左、右 (dr, dc)
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_action(self, owner, army, terrain, fog):
        """
        输入: (owner, army, terrain, fog) 均为 (H, W) numpy数组
          owner: -1=中立, 0=红, 1=蓝
          army: 兵力数量
          terrain: 0=空地, 1=山脉, 2=将军, 3=城市
          fog: True=不可见

        返回: (sy, sx, ny, nx, is_half) 或 None (跳过)
        """
        best_action = None
        best_score = -999999

        for sy in range(self.grid_size):
            for sx in range(self.grid_size):
                if owner[sy, sx] != self.player_id:
                    continue
                if army[sy, sx] <= 1:
                    continue

                arm = army[sy, sx]

                for dr, dc in self.dirs:
                    ny, nx = sy + dr, sx + dc

                    # 越界
                    if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                        continue

                    # 不能撞山
                    if terrain[ny, nx] == 1:
                        continue

                    target_army = army[ny, nx]
                    target_owner = owner[ny, nx]

                    # 中立城市: 必须兵力 > 城市兵力+1 才能占领
                    if terrain[ny, nx] == 3 and target_owner != self.player_id:
                        if arm <= target_army + 1:
                            continue

                    # 敌方地盘: 必须兵力 > 敌方兵力+1 (不打亏本仗)
                    if target_owner == self.enemy_id and arm <= target_army + 1:
                        continue

                    # === 评分 ===
                    score = self._evaluate(sy, sx, ny, nx, arm,
                                           owner, army, terrain, fog)

                    if score > best_score:
                        best_score = score
                        is_half = False
                        best_action = (sy, sx, ny, nx, is_half)

        return best_action

    def _evaluate(self, sy, sx, ny, nx, arm, owner, army, terrain, fog):
        """战术评分: 越高越好"""
        score = 0
        target_owner = owner[ny, nx]

        # --- 1. 攻击敌方 (最高优先级) ---
        if target_owner == self.enemy_id:
            if terrain[ny, nx] == 2:  # 敌方将军
                score += 1000000  # 直接赢!
            else:
                score += 50000 + arm * 10

        # --- 2. 占领无主空地 (扩张) ---
        elif target_owner == -1 and terrain[ny, nx] != 3:
            score += 10000 + arm * 2

        # --- 3. 探索迷雾 ---
        if fog[ny, nx]:
            score += 5000

        # --- 4. 前线引力 (解决龟缩) ---
        if target_owner == self.player_id:
            front_s = self._count_front(sy, sx, owner, fog)
            front_n = self._count_front(ny, nx, owner, fog)

            if front_n > front_s:
                # 从后方流向前线: 大奖励 (兵力越大越应该去前线)
                score += 2000 + arm * 5
            elif front_n == 0 and front_s == 0:
                # 都在大后方: 随意流动 (防死锁)
                score += arm * 0.1 + np.random.rand() * 10
            else:
                # 严惩从前线调回后方
                score -= 10000

        return score

    def _count_front(self, y, x, owner, fog):
        """计算周围有多少个非己方格子 (越多越像前线)"""
        cnt = 0
        for dr, dc in self.dirs:
            ny, nx = y + dr, x + dc
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if owner[ny, nx] != self.player_id or fog[ny, nx]:
                    cnt += 1
        return cnt


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import ctypes
    import os

    # 加载引擎获取测试状态
    CPP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp')
    lib = ctypes.CDLL(os.path.join(CPP_DIR, "libgenerals.so"))

    lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    lib.generals_create.restype = ctypes.c_void_p
    lib.generals_get_grid_data.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.generals_get_grid_data.restype = None

    state = lib.generals_create(12, 12, 600, 42)

    owner_buf = (ctypes.c_int8 * 144)()
    army_buf = (ctypes.c_int16 * 144)()
    terrain_buf = (ctypes.c_uint8 * 144)()

    lib.generals_get_grid_data(state, owner_buf, army_buf, terrain_buf)

    owner = np.frombuffer(owner_buf, dtype=np.int8).reshape(12, 12).copy()
    army = np.frombuffer(army_buf, dtype=np.int16).reshape(12, 12).copy()
    terrain = np.frombuffer(terrain_buf, dtype=np.uint8).reshape(12, 12).copy()
    fog = np.zeros((12, 12), dtype=bool)

    # 创建脚本 AI (蓝方 player_id=1)
    bot = ScriptBot(player_id=1, grid_size=12)

    print("=== Script Bot Test ===")
    print(f"Owner map (0=red, 1=blue):")
    for y in range(12):
        row = ""
        for x in range(12):
            if terrain[y, x] == 2:
                row += "G" if owner[y, x] == 0 else "g"
            elif terrain[y, x] == 1:
                row += "M"
            elif terrain[y, x] == 3:
                row += "C"
            elif owner[y, x] == 0:
                row += "R"
            elif owner[y, x] == 1:
                row += "B"
            else:
                row += "."
        print(f"  {row}")

    # 跑 100 步看脚本 AI 行为
    print(f"\nRunning 100 steps with ScriptBot (blue) vs skip (red)...")
    for step in range(100):
        # 蓝方用脚本 AI
        action = bot.get_action(owner, army, terrain, fog)
        if action:
            sy, sx, ny, nx, is_half = action
            print(f"  Step {step}: Blue ({sx},{sy}) -> ({nx},{ny}), army={army[sy,sx]}")
            # 执行 (简化: 直接用 step_dual)
            lib.generals_step(state, -1)  # red skip
        else:
            print(f"  Step {step}: Blue has no valid move (skip)")
            lib.generals_step(state, -1)

        # 重新获取状态
        lib.generals_get_grid_data(state, owner_buf, army_buf, terrain_buf)
        owner = np.frombuffer(owner_buf, dtype=np.int8).reshape(12, 12).copy()
        army = np.frombuffer(army_buf, dtype=np.int16).reshape(12, 12).copy()
        terrain = np.frombuffer(terrain_buf, dtype=np.uint8).reshape(12, 12).copy()

    # 统计
    blue_tiles = np.sum(owner == 1)
    red_tiles = np.sum(owner == 0)
    blue_army = int(np.sum(army[owner == 1]))
    red_army = int(np.sum(army[owner == 0]))
    print(f"\nAfter 100 steps:")
    print(f"  Blue: {blue_tiles} tiles, {blue_army} army")
    print(f"  Red:  {red_tiles} tiles, {red_army} army")

    lib.generals_destroy(state)
    print("\n=== Test Complete ===")
