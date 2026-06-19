"""
script_bot.py — 启发式脚本 AI V3 (角色分离 + 严格梯度)

V3 修复内容:
  1. 角色分离: 小股部队(<=10)铺地毯, 大部队(>20)打敌人
  2. 严格梯度: 兵力只能流向更前线, 严惩平移/倒退
  3. 目标感: 大军团优先攻击/探索迷雾, 不再抓苍蝇
  4. 防喂食: 禁止往已占大兵力格子运兵

用法:
  from script_bot import ScriptBot
  bot = ScriptBot(player_id=1, grid_size=12)
  action = bot.get_action(owner, army, terrain, fog)
"""

import os
import numpy as np


class ScriptBotV3:
    """启发式贪心脚本 AI — V3 角色分离版"""

    def __init__(self, player_id, grid_size=12):
        self.player_id = player_id
        self.enemy_id = 1 - player_id
        self.grid_size = grid_size
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

    def get_action(self, owner, army, terrain, fog):
        """
        输入: (owner, army, terrain, fog) 均为 (H, W) numpy数组
        返回: (sy, sx, ny, nx, is_half) 或 None (跳过)
        """
        best_action = None
        best_score = -999999

        # 预计算: 每个格子的"前线度" (周围有多少非己方/迷雾)
        front_map = self._compute_frontmap(owner, fog)

        for sy in range(self.grid_size):
            for sx in range(self.grid_size):
                if owner[sy, sx] != self.player_id:
                    continue
                if army[sy, sx] <= 1:
                    continue

                arm = army[sy, sx]

                for dr, dc in self.dirs:
                    ny, nx = sy + dr, sx + dc

                    if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                        continue
                    if terrain[ny, nx] == 1:  # 不能撞山
                        continue

                    target_army = army[ny, nx]
                    target_owner = owner[ny, nx]

                    # 中立城市: 必须兵力 > 城市兵力+1
                    if terrain[ny, nx] == 3 and target_owner != self.player_id:
                        if arm <= target_army + 1:
                            continue

                    # 敌方地盘: 必须兵力 > 敌方兵力+1
                    if target_owner == self.enemy_id and arm <= target_army + 1:
                        continue

                    # === V3 评分 ===
                    score = self._evaluate_v3(
                        sy, sx, ny, nx, arm,
                        owner, army, terrain, fog, front_map
                    )

                    if score > best_score:
                        best_score = score
                        best_action = (sy, sx, ny, nx, False)

        return best_action

    def _evaluate_v3(self, sy, sx, ny, nx, arm, owner, army, terrain, fog, front_map):
        """V3 评分函数: 角色分离 + 严格梯度"""
        score = 0
        target_owner = owner[ny, nx]
        front_s = front_map[sy, sx]
        front_n = front_map[ny, nx]

        # ============================================
        # 1. 攻击敌方 (最高优先级)
        # ============================================
        if target_owner == self.enemy_id:
            if terrain[ny, nx] == 2:  # 敌方将军
                score += 1000000  # 直接赢!
            else:
                # 大部队砸向敌人的得分碾压一切
                score += 50000 + arm * 20
            return score  # 攻击优先级最高，直接返回

        # ============================================
        # 2. 占领无主空地 (扩张) — V3: 角色分离
        # ============================================
        if target_owner == -1 and terrain[ny, nx] != 3:
            # 统计己方总领地数
            own_tiles = np.sum(owner == self.player_id)
            if own_tiles > 100:
                # 已经占了大片地盘, 不再盲目扩张, 只让小部队就近铺
                if arm > 5:
                    score -= arm * 10  # 大部队完全不参与后期扩张
                else:
                    score += 1000
            else:
                if arm <= 10:
                    score += 10000 + arm * 2
                elif arm <= 20:
                    score += 5000 - arm * 3
                else:
                    score -= arm * 5

        # ============================================
        # 3. 探索迷雾 (大部队优先)
        # ============================================
        if fog[ny, nx]:
            if arm > 30:
                # 大军团深入迷雾: 高奖励 (寻找敌人主力)
                score += 30000 + arm * 10
            elif arm > 10:
                # 中等部队探索
                score += 5000 + arm * 3
            else:
                # 小部队探路
                score += 2000

        # ============================================
        # 4. 己方合并 (兵力流动) — V3: 严格梯度
        # ============================================
        if target_owner == self.player_id:
            # 防喂食: 目标已有大量兵力时, 不要往里塞
            if army[ny, nx] > 20 and arm > 3:
                score -= 5000  # 严厉惩罚"喂兵"

            if front_n < front_s:
                # 流向更前线: 正确后勤
                score += 5000 + arm * 5
            elif front_n == front_s:
                # 同级平移: 无意义
                score -= 5000
            else:
                # 流向后方: 严惩!
                score -= 50000

        return score

    def _compute_frontmap(self, owner, fog):
        """计算每个格子的"前线度": 周围非己方/迷雾格子数"""
        front = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cnt = 0
                for dr, dc in self.dirs:
                    ny, nx = y + dr, x + dc
                    if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                        if owner[ny, nx] != self.player_id or fog[ny, nx]:
                            cnt += 1
                front[y, x] = cnt
        return front


# 保持向后兼容: ScriptBot = ScriptBotV3
ScriptBot = ScriptBotV3


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import ctypes

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
    lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.generals_step_dual.restype = ctypes.c_int
    lib.generals_get_winner.argtypes = [ctypes.c_void_p]
    lib.generals_get_winner.restype = ctypes.c_int
    lib.generals_destroy.argtypes = [ctypes.c_void_p]
    lib.generals_destroy.restype = None

    state = lib.generals_create(12, 12, 600, 42)

    o = (ctypes.c_int8 * 144)()
    a = (ctypes.c_int16 * 144)()
    t = (ctypes.c_uint8 * 144)()

    lib.generals_get_grid_data(state, o, a, t)
    owner = np.frombuffer(o, dtype=np.int8).reshape(12, 12).copy()
    army = np.frombuffer(a, dtype=np.int16).reshape(12, 12).copy()
    terrain = np.frombuffer(t, dtype=np.uint8).reshape(12, 12).copy()
    fog = np.zeros((12, 12), dtype=bool)

    bot = ScriptBotV3(player_id=1, grid_size=12)

    print("=== ScriptBot V3 Test (blue vs skip) ===\n")

    action_log = []
    feeding = 0
    backline_shuffle = 0
    max_army = 0

    for step in range(300):
        if lib.generals_get_winner(state) != -1:
            print(f"Game over at step {step}")
            break

        action = bot.get_action(owner, army, terrain, fog)

        if action:
            sy, sx, ny, nx, h = action
            enc = (sy * 12 + sx) * 8 + [(-1,0),(1,0),(0,-1),(0,1)].index((ny-sy, nx-sx)) * 2
            action_log.append((sx, sy, nx, ny, int(army[sy, sx])))

            # 统计问题
            if army[ny, nx] > 20 and army[sy, sx] > 3:
                feeding += 1

            if army[ny, nx] > max_army:
                max_army = army[ny, nx]
        else:
            enc = 1152

        lib.generals_step_dual(state, 1152, enc)
        lib.generals_get_grid_data(state, o, a, t)
        owner = np.frombuffer(o, dtype=np.int8).reshape(12, 12).copy()
        army = np.frombuffer(a, dtype=np.int16).reshape(12, 12).copy()
        terrain = np.frombuffer(t, dtype=np.uint8).reshape(12, 12).copy()

    blue_tiles = np.sum(owner == 1)
    blue_army = int(np.sum(army[owner == 1]))

    print(f"Total steps: {len(action_log)}")
    print(f"Final: Blue {blue_tiles} tiles / {blue_army} army")
    print(f"Max single tile army: {max_army}")
    print(f"Feeding moves (>20 target, >3 source): {feeding}")

    # 兵力集中度
    big_tiles = int(np.sum(army[owner == 1] > 30))
    print(f"Tiles with >30 army: {big_tiles}")

    # 最后10步
    print(f"\nLast 10 actions:")
    for sx, sy, nx, ny, arm in action_log[-10:]:
        print(f"  ({sx},{sy})->({nx},{ny}) arm={arm}")

    lib.generals_destroy(state)
    print("\n=== V3 Test Complete ===")
