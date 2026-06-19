"""
script_bot.py — 启发式脚本 AI V5 (严格流场 + 防守嗅觉)

核心哲学:
  1. 绝对铁律: 水往低处流, 禁止平移/倒退 (dist_n < dist_s)
  2. 蓄水效应: 前线打不过就等后方兵力汇聚, 直到溃坝
  3. 大军主导: 同等条件下大军团优先行动
  4. 危机雷达: 发现偷家敌军时切换引力源, 全军回防

用法:
  from script_bot import ScriptBot
  bot = ScriptBot(player_id=1, grid_size=12)
  action = bot.get_action(owner, army, terrain, fog)
"""

import numpy as np
from collections import deque


class ScriptBot:
    """V5 严格流场 + 防守嗅觉脚本 AI"""

    def __init__(self, player_id, grid_size=12):
        self.player_id = player_id
        self.enemy_id = 1 - player_id
        self.grid_size = grid_size
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.enemy_general_pos = None

    def get_action(self, owner, army, terrain, fog):
        """
        输入: (owner, army, terrain, fog) 均为 (H, W) numpy数组
        返回: (sy, sx, ny, nx, is_half) 或 None (跳过)
        """
        GS = self.grid_size

        # 1. 记忆敌方将军位置
        for y in range(GS):
            for x in range(GS):
                if owner[y, x] == self.enemy_id and terrain[y, x] == 2:
                    self.enemy_general_pos = (y, x)
                    break
            if self.enemy_general_pos:
                break

        # 2. 找到我方将军位置
        my_general_pos = None
        for y in range(GS):
            for x in range(GS):
                if owner[y, x] == self.player_id and terrain[y, x] == 2:
                    my_general_pos = (y, x)
                    break
            if my_general_pos:
                break

        # ============================================
        # 3. 危机雷达: 扫描偷家敌军
        # ============================================
        threat_level = 0
        threat_pos = None
        defense_mode = False

        if my_general_pos:
            gy, gx = my_general_pos
            for y in range(GS):
                for x in range(GS):
                    # 发现敌军且兵力 > 10 (才算真正的威胁)
                    if owner[y, x] == self.enemy_id and army[y, x] > 10:
                        dist = abs(y - gy) + abs(x - gx)
                        # 敌军离我家 < 6 格, 且兵力接近或超过我家将军
                        if dist < 6 and army[y, x] > army[gy, gx] - 5:
                            current_threat = army[y, x] / max(1, dist)
                            if current_threat > threat_level:
                                threat_level = current_threat
                                threat_pos = (y, x)

        # ============================================
        # 4. 构建目标矩阵 (根据模式切换)
        # ============================================
        target_grid = np.zeros((GS, GS), dtype=bool)

        if threat_pos is not None:
            # 🚨 红色警报! 全军回防, 目标锁定偷家敌军
            target_grid[threat_pos[0], threat_pos[1]] = True
            defense_mode = True
        else:
            # 正常进攻/扩张模式
            defense_mode = False
            if self.enemy_general_pos is not None:
                target_grid[self.enemy_general_pos[0], self.enemy_general_pos[1]] = True
            else:
                for y in range(GS):
                    for x in range(GS):
                        if owner[y, x] == self.enemy_id or fog[y, x]:
                            target_grid[y, x] = True
                        elif owner[y, x] == -1 and terrain[y, x] != 1:
                            target_grid[y, x] = True

        # 5. BFS 距离场
        dist_map = self._build_distance_map(target_grid, terrain)

        # 6. 寻找最优动作
        best_action = None
        best_score = -999999

        for sy in range(GS):
            for sx in range(GS):
                if owner[sy, sx] != self.player_id:
                    continue
                if army[sy, sx] <= 1:
                    continue

                arm = army[sy, sx]
                dist_s = dist_map[sy, sx]

                for dr, dc in self.dirs:
                    ny, nx = sy + dr, sx + dc

                    if not (0 <= ny < GS and 0 <= nx < GS):
                        continue
                    if terrain[ny, nx] == 1:
                        continue

                    target_army = army[ny, nx]
                    target_owner = owner[ny, nx]
                    dist_n = dist_map[ny, nx]

                    # 绝对铁律: 水往低处流
                    if dist_n >= dist_s:
                        continue

                    # 打不过就等
                    if target_owner != self.player_id and arm <= target_army + 1:
                        continue

                    # === 评分系统 ===
                    score = arm  # 基础分: 大部队先动

                    if target_owner != self.player_id:
                        score += 100000
                        if target_owner == self.enemy_id:
                            score += 50000
                            # 防守模式: 撞向偷家敌军给最高优先级
                            if defense_mode and (ny, nx) == threat_pos:
                                score += 1000000

                    if score > best_score:
                        best_score = score
                        best_action = (sy, sx, ny, nx, False)

        return best_action

    def _build_distance_map(self, target_grid, terrain, owner=None):
        """BFS 距离场: 从所有目标点出发, 计算每个格子到最近目标点的距离"""
        GS = self.grid_size
        dist = np.full((GS, GS), 999, dtype=np.int32)
        q = deque()

        for y in range(GS):
            for x in range(GS):
                if target_grid[y, x]:
                    dist[y, x] = 0
                    q.append((y, x))

        while q:
            cy, cx = q.popleft()
            for dr, dc in self.dirs:
                ny, nx = cy + dr, cx + dc
                if 0 <= ny < GS and 0 <= nx < GS:
                    if terrain[ny, nx] != 1 and dist[ny, nx] == 999:
                        dist[ny, nx] = dist[cy, cx] + 1
                        q.append((ny, nx))

        return dist


# 保持向后兼容
ScriptBotV3 = ScriptBot
ScriptBotV4 = ScriptBot


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import os
    import ctypes

    CPP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp')
    lib = ctypes.CDLL(os.path.join(CPP_DIR, "libgenerals.so"))

    lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    lib.generals_create.restype = ctypes.c_void_p
    lib.generals_destroy.argtypes = [ctypes.c_void_p]
    lib.generals_destroy.restype = None
    lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.generals_step_dual.restype = ctypes.c_int
    lib.generals_get_winner.argtypes = [ctypes.c_void_p]
    lib.generals_get_winner.restype = ctypes.c_int
    lib.generals_get_grid_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint8)]
    lib.generals_get_grid_data.restype = None

    state = lib.generals_create(12, 12, 600, 42)

    o = (ctypes.c_int8 * 144)()
    a = (ctypes.c_int16 * 144)()
    t = (ctypes.c_uint8 * 144)()

    lib.generals_get_grid_data(state, o, a, t)
    owner = np.frombuffer(o, dtype=np.int8).reshape(12, 12).copy()
    army = np.frombuffer(a, dtype=np.int16).reshape(12, 12).copy()
    terrain = np.frombuffer(t, dtype=np.uint8).reshape(12, 12).copy()
    fog = np.zeros((12, 12), dtype=bool)

    bot = ScriptBot(player_id=1, grid_size=12)

    print("=== ScriptBot V5 (Defense Radar) ===\n")

    blue_gen = None
    for y in range(12):
        for x in range(12):
            if terrain[y,x] == 2 and owner[y,x] == 1:
                blue_gen = (x, y)
    print(f"Blue general: {blue_gen}")

    action_log = []
    defense_triggers = 0

    for step in range(300):
        if lib.generals_get_winner(state) != -1:
            print(f"Game over at step {step}, winner: {lib.generals_get_winner(state)}")
            break

        action = bot.get_action(owner, army, terrain, fog)

        if action:
            sy, sx, ny, nx, h = action
            enc = (sy * 12 + sx) * 8 + [(-1,0),(1,0),(0,-1),(0,1)].index((ny-sy, nx-sx)) * 2
            action_log.append((sx, sy, nx, ny, int(army[sy, sx])))
        else:
            enc = 1152
            action_log.append(None)

        lib.generals_step_dual(state, 1152, enc)
        lib.generals_get_grid_data(state, o, a, t)
        owner = np.frombuffer(o, dtype=np.int8).reshape(12, 12).copy()
        army = np.frombuffer(a, dtype=np.int16).reshape(12, 12).copy()
        terrain = np.frombuffer(t, dtype=np.uint8).reshape(12, 12).copy()

    blue_tiles = np.sum(owner == 1)
    blue_army = int(np.sum(army[owner == 1]))
    red_tiles = np.sum(owner == 0)
    red_army = int(np.sum(army[owner == 0]))

    print(f"\nFinal: Blue {blue_tiles} tiles / {blue_army} army, Red {red_tiles} tiles / {red_army} army")

    max_a = 0
    max_pos = None
    for y in range(12):
        for x in range(12):
            if army[y,x] > max_a:
                max_a = army[y,x]
                max_pos = (x, y)
    print(f"Max tile: {max_pos} = {max_a}")

    # Last 10
    print(f"\nLast 10 actions:")
    for entry in action_log[-10:]:
        if entry:
            sx, sy, nx, ny, arm = entry
            print(f"  ({sx},{sy})->({nx},{ny}) arm={arm}")
        else:
            print(f"  SKIP")

    lib.generals_destroy(state)
    print("\n=== V5 Test Complete ===")
