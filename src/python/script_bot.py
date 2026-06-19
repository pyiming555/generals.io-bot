"""
script_bot.py — 启发式脚本 AI V6 (专家系统版)

核心升级:
  1. MST 有向树: parent_map 替代自由流场, 兵力单向汇聚不分流
  2. 咽喉点防守: 几何探测咽喉, 隘口列阵
  3. 迷雾残影: ghost_grid 追踪隐残影

用法:
  from script_bot import ScriptBot
  bot = ScriptBot(player_id=1, grid_size=12)
  action = bot.get_action(owner, army, terrain, fog)
"""

import numpy as np
from collections import deque


class ScriptBot:
    """V6 专家系统脚本 AI"""

    def __init__(self, player_id, grid_size=12):
        self.player_id = player_id
        self.enemy_id = 1 - player_id
        self.grid_size = grid_size
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        self.enemy_general_pos = None
        self.my_general_pos = None

        # 迷雾残影: 记录最后看到的敌方兵力, 随时间衰减
        self.enemy_ghost = np.zeros((grid_size, grid_size), dtype=np.float32)

        # 咽喉点地图 (静态, 游戏开始时计算)
        self.chokepoints = np.zeros((grid_size, grid_size), dtype=bool)

    def get_action(self, owner, army, terrain, fog):
        """返回: (sy, sx, ny, nx, is_half) 或 None"""
        GS = self.grid_size

        # 1. 更新敌方将军位置
        for y in range(GS):
            for x in range(GS):
                if owner[y, x] == self.enemy_id and terrain[y, x] == 2:
                    self.enemy_general_pos = (y, x)
                    break
            if self.enemy_general_pos:
                break

        # 2. 找到我方将军位置
        if self.my_general_pos is None:
            for y in range(GS):
                for x in range(GS):
                    if owner[y, x] == self.player_id and terrain[y, x] == 2:
                        self.my_general_pos = (y, x)
                        break
                if self.my_general_pos:
                    break

        # 3. 计算咽喉点 (只在第一次)
        if not np.any(self.chokepoints):
            self._compute_chokepoints(terrain)

        # 4. 更新迷雾残影
        self._update_ghost(owner, army, fog)

        # 5. 危机雷达: 扫描真实敌人 + 残影
        threat_level = 0
        threat_pos = None
        defense_mode = False

        if self.my_general_pos:
            gy, gx = self.my_general_pos
            # 扫描真实敌人
            for y in range(GS):
                for x in range(GS):
                    if owner[y, x] == self.enemy_id and army[y, x] > 10:
                        dist = abs(y - gy) + abs(x - gx)
                        if dist < 8 and army[y, x] > army[gy, gx] - 5:
                            current_threat = army[y, x] / max(1, dist)
                            if current_threat > threat_level:
                                threat_level = current_threat
                                threat_pos = (y, x)

            # 扫描残影 (迷雾中隐形的敌军)
            if threat_pos is None:
                for y in range(GS):
                    for x in range(GS):
                        ghost_army = self.enemy_ghost[y, x]
                        if ghost_army > 15:  # 残影阈值
                            dist = abs(y - gy) + abs(x - gx)
                            if dist < 8:
                                current_threat = ghost_army / max(1, dist)
                                if current_threat > threat_level:
                                    threat_level = current_threat
                                    threat_pos = (y, x)

            if threat_pos is not None:
                defense_mode = True

        # 6. 构建目标矩阵
        target_grid = np.zeros((GS, GS), dtype=bool)

        if defense_mode and threat_pos:
            # 防守模式: 目标 = 咽喉点 (如果敌军路径上有咽喉)
            # 否则目标 = 威胁位置
            choke_target = self._find_nearest_chokepoint(threat_pos)
            if choke_target:
                target_grid[choke_target[0], choke_target[1]] = True
            else:
                target_grid[threat_pos[0], threat_pos[1]] = True
        else:
            # 进攻模式
            if self.enemy_general_pos is not None:
                target_grid[self.enemy_general_pos[0], self.enemy_general_pos[1]] = True
            else:
                # 没有将军信息: 所有迷雾 + 空地都是目标
                for y in range(GS):
                    for x in range(GS):
                        if owner[y, x] == self.enemy_id or fog[y, x]:
                            target_grid[y, x] = True
                        elif owner[y, x] == -1 and terrain[y, x] != 1:
                            target_grid[y, x] = True

        # 7. BFS 生成距离场 + Parent 有向树
        dist_map, parent_map = self._build_distance_and_parent(target_grid, terrain)

        # 8. 寻找最优动作 (沿有向树移动)
        best_action = None
        best_score = -999999

        for sy in range(GS):
            for sx in range(GS):
                if owner[sy, sx] != self.player_id:
                    continue
                if army[sy, sx] <= 1:
                    continue

                arm = army[sy, sx]

                # V6 有向树: 只能移动到 parent 指向的格子
                parent = parent_map[sy, sx]
                if parent[0] == -1:
                    continue  # 没有 parent (已在目标或不可达)

                ny, nx = parent[0], parent[1]

                # 验证合法性
                if terrain[ny, nx] == 1:
                    continue

                target_army = army[ny, nx]
                target_owner = owner[ny, nx]

                # 打不过就等
                if target_owner != self.player_id and arm <= target_army + 1:
                    continue

                # === 评分 ===
                score = arm  # 基础分: 大部队先动

                if target_owner != self.player_id:
                    score += 100000
                    if target_owner == self.enemy_id:
                        score += 50000
                        if defense_mode and (ny, nx) == threat_pos:
                            score += 1000000

                if score > best_score:
                    best_score = score
                    best_action = (sy, sx, ny, nx, False)

        return best_action

    def _update_ghost(self, owner, army, fog):
        """更新迷雾残影: 看到真实敌人时更新, 否则衰减"""
        GS = self.grid_size
        for y in range(GS):
            for x in range(GS):
                if owner[y, x] == self.enemy_id and not fog[y, x]:
                    # 看到真实敌人: 更新残影
                    self.enemy_ghost[y, x] = army[y, x]
                elif fog[y, x]:
                    # 迷雾中: 残影衰减
                    self.enemy_ghost[y, x] *= 0.95
                    if self.enemy_ghost[y, x] < 0.5:
                        self.enemy_ghost[y, x] = 0
                else:
                    # 没有敌人: 清除残影
                    self.enemy_ghost[y, x] = 0

    def _compute_chokepoints(self, terrain):
        """
        几何探测咽喉点:
        如果某个非山脉格子的上下(或左右)都被山脉或地图边缘夹住,
        则它是咽喉点 (1格宽的通道).
        """
        GS = self.grid_size
        for y in range(GS):
            for x in range(GS):
                if terrain[y, x] == 1:
                    continue

                # 检查垂直方向: 上下都是山或边缘
                up_blocked = (y == 0) or (terrain[y-1, x] == 1)
                down_blocked = (y == GS-1) or (terrain[y+1, x] == 1)
                if up_blocked and down_blocked:
                    self.chokepoints[y, x] = True
                    continue

                # 检查水平方向: 左右都是山或边缘
                left_blocked = (x == 0) or (terrain[y, x-1] == 1)
                right_blocked = (x == GS-1) or (terrain[y, x+1] == 1)
                if left_blocked and right_blocked:
                    self.chokepoints[y, x] = True

    def _find_nearest_chokepoint(self, threat_pos):
        """找到威胁位置到我方将军路径上的最近咽喉点"""
        if self.my_general_pos is None:
            return None

        my_y, my_x = self.my_general_pos
        threat_y, threat_x = threat_pos

        # 在威胁和我方将军之间的连线上找最近的咽喉点
        best_choke = None
        best_dist = 999

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if not self.chokepoints[y, x]:
                    continue
                # 咽喉点应该在威胁和我方将军之间
                dist_to_threat = abs(y - threat_y) + abs(x - threat_x)
                dist_to_me = abs(y - my_y) + abs(x - my_x)
                total = dist_to_threat + dist_to_me
                if total < best_dist:
                    best_dist = total
                    best_choke = (y, x)

        return best_choke

    def _build_distance_and_parent(self, target_grid, terrain):
        """
        BFS 同时生成距离场和 Parent 有向树.
        parent[y,x] = (py, px) 表示这个格子的最佳下一步走向.
        """
        GS = self.grid_size
        dist = np.full((GS, GS), 999, dtype=np.int32)
        parent = np.full((GS, GS, 2), -1, dtype=np.int32)
        q = deque()

        # 初始化: 所有目标点 dist=0, parent=自己
        for y in range(GS):
            for x in range(GS):
                if target_grid[y, x]:
                    dist[y, x] = 0
                    parent[y, x] = (y, x)  # 目标是自己的根
                    q.append((y, x))

        # BFS
        while q:
            cy, cx = q.popleft()
            for dr, dc in self.dirs:
                ny, nx = cy + dr, cx + dc
                if 0 <= ny < GS and 0 <= nx < GS:
                    if terrain[ny, nx] != 1 and dist[ny, nx] == 999:
                        dist[ny, nx] = dist[cy, cx] + 1
                        # Parent 指向来源 (即从 (ny,nx) 走向 (cy,cx) 更近)
                        parent[ny, nx] = (cy, cx)
                        q.append((ny, nx))

        return dist, parent


# 保持向后兼容
ScriptBotV3 = ScriptBot
ScriptBotV4 = ScriptBot
ScriptBotV5 = ScriptBot


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

    print("=== ScriptBot V6 (MST + Chokepoint + Ghost) ===\n")

    blue_gen = None
    for y in range(12):
        for x in range(12):
            if terrain[y, x] == 2 and owner[y, x] == 1:
                blue_gen = (x, y)
    print(f"Blue general: {blue_gen}")

    action_log = []
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
            if army[y, x] > max_a:
                max_a = army[y, x]
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
    print("\n=== V6 Test Complete ===")
