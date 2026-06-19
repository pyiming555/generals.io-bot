"""
script_bot.py — 启发式脚本 AI V4 (严格流场: 水往低处流, 蓄水破城)

核心哲学:
  1. 绝对铁律: 水往低处流, 禁止平移/倒退 (dist_n < dist_s)
  2. 蓄水效应: 前线打不过就等后方兵力汇聚, 直到溃坝
  3. 大军主导: 同等条件下大军团优先行动
  4. BFS 距离场: 以敌方将军/迷雾/空地为目标, 生成全局梯度

用法:
  from script_bot import ScriptBot
  bot = ScriptBot(player_id=1, grid_size=12)
  action = bot.get_action(owner, army, terrain, fog)
"""

import numpy as np
from collections import deque


class ScriptBot:
    """V4 严格流场脚本 AI"""

    def __init__(self, player_id, grid_size=12):
        self.player_id = player_id
        self.enemy_id = 1 - player_id
        self.grid_size = grid_size
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        self.enemy_general_pos = None

    def get_action(self, owner, army, terrain, fog):
        """
        输入: (owner, army, terrain, fog) 均为 (H, W) numpy数组
        返回: (sy, sx, ny, nx, is_half) 或 None (跳过)
        """
        GS = self.grid_size

        # 1. 寻找敌方将军位置 (如果已知)
        if self.enemy_general_pos is None:
            for y in range(GS):
                for x in range(GS):
                    if owner[y, x] == self.enemy_id and terrain[y, x] == 2:
                        self.enemy_general_pos = (y, x)
                        break
                if self.enemy_general_pos:
                    break

        # 2. 构建目标矩阵 (BFS 的起点集合)
        target_grid = np.zeros((GS, GS), dtype=bool)
        if self.enemy_general_pos is not None:
            # 优先目标: 敌方将军
            target_grid[self.enemy_general_pos[0], self.enemy_general_pos[1]] = True
        else:
            # 如果没有发现将军, 所有迷雾和空地都是目标
            for y in range(GS):
                for x in range(GS):
                    if owner[y, x] == self.enemy_id or fog[y, x]:
                        target_grid[y, x] = True
                    elif owner[y, x] == -1 and terrain[y, x] != 1:
                        target_grid[y, x] = True

        # 3. BFS 生成距离场
        dist_map = self._build_distance_map(target_grid, terrain, owner)

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

                    # 越界
                    if not (0 <= ny < GS and 0 <= nx < GS):
                        continue
                    # 不能撞山
                    if terrain[ny, nx] == 1:
                        continue

                    target_army = army[ny, nx]
                    target_owner = owner[ny, nx]
                    dist_n = dist_map[ny, nx]

                    # === 绝对铁律: 水往低处流！禁止平移和倒退！ ===
                    if dist_n >= dist_s:
                        continue

                    # === 物理定律: 打不过就别送死 (等后方兵力汇聚) ===
                    if target_owner != self.player_id and arm <= target_army + 1:
                        continue

                    # === 极简评分系统 ===
                    score = 0

                    if target_owner != self.player_id:
                        # 攻击/占领: 最高优先级, 前线部队先动
                        score += 100000
                        if target_owner == self.enemy_id:
                            score += 50000  # 杀敌优先
                    # else: 己方合并, 只靠 arm 决定优先级

                    # 💡 打破僵局: 同等条件下, 优先移动兵力最大的部队
                    # 大军团像推土机一样往前碾压, 不会被后方运 2 兵的操作抢占
                    score += arm

                    if score > best_score:
                        best_score = score
                        best_action = (sy, sx, ny, nx, False)

        return best_action

    def _build_distance_map(self, target_grid, terrain, owner):
        """
        BFS 距离场: 从所有目标点出发, 计算每个格子到最近目标点的距离.
        不可达格子标记为 999.
        """
        GS = self.grid_size
        dist = np.full((GS, GS), 999, dtype=np.int32)
        q = deque()

        # 初始化: 所有目标点 dist=0
        for y in range(GS):
            for x in range(GS):
                if target_grid[y, x]:
                    dist[y, x] = 0
                    q.append((y, x))

        # BFS
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

    print("=== ScriptBot V4 (Strict Flow Field) ===\n")

    # 找将军
    bg = None
    for y in range(12):
        for x in range(12):
            if terrain[y,x] == 2 and owner[y,x] == 1:
                bg = (x, y)
    print(f"Blue general: {bg}")

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

    # Max tile
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
    print("\n=== V4 Test Complete ===")
