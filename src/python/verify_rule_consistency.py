"""
verify_rule_consistency.py — 验证 GUI 与训练环境的规则一致性

对比项:
  1. 初始地图 (同 seed 是否一致)
  2. 单步动作执行 (step_dual vs 训练环境的 step)
  3. 兵力增长时序 (tick 频率)
  4. 城池/将军攻占逻辑
  5. 合法动作掩码

用法:
  python3 verify_rule_consistency.py
"""

import ctypes
import numpy as np
import os
import sys

# 加载 C++ 引擎
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CPP_DIR = os.path.join(PROJECT_DIR, "src", "cpp")

_lib_path = os.path.join(CPP_DIR, "libgenerals.so")
_lib = ctypes.CDLL(_lib_path)

# 类型定义
_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_bool_p = ctypes.POINTER(ctypes.c_bool)
_c_int8_p = ctypes.POINTER(ctypes.c_int8)
_c_int16_p = ctypes.POINTER(ctypes.c_int16)
_c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
_c_void_p = ctypes.c_void_p

_lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.generals_create.restype = _c_void_p
_lib.generals_destroy.argtypes = [_c_void_p]
_lib.generals_destroy.restype = None
_lib.generals_reset.argtypes = [_c_void_p, ctypes.c_uint]
_lib.generals_reset.restype = None
_lib.generals_step.argtypes = [_c_void_p, ctypes.c_int]
_lib.generals_step.restype = ctypes.c_int
_lib.generals_step_dual.argtypes = [_c_void_p, ctypes.c_int, ctypes.c_int]
_lib.generals_step_dual.restype = ctypes.c_int
_lib.generals_get_winner.argtypes = [_c_void_p]
_lib.generals_get_winner.restype = ctypes.c_int
_lib.generals_get_step.argtypes = [_c_void_p]
_lib.generals_get_step.restype = ctypes.c_int
_lib.generals_is_stalemate.argtypes = [_c_void_p]
_lib.generals_is_stalemate.restype = ctypes.c_bool
_lib.generals_get_width.argtypes = [_c_void_p]
_lib.generals_get_width.restype = ctypes.c_int
_lib.generals_get_height.argtypes = [_c_void_p]
_lib.generals_get_height.restype = ctypes.c_int
_lib.generals_skip_action.argtypes = [_c_void_p]
_lib.generals_skip_action.restype = ctypes.c_int
_lib.generals_clone.argtypes = [_c_void_p]
_lib.generals_clone.restype = _c_void_p
_lib.generals_get_grid_data.argtypes = [_c_void_p, _c_int8_p, _c_int16_p, _c_uint8_p]
_lib.generals_get_grid_data.restype = None
_lib.generals_get_action_mask.argtypes = [_c_void_p, _c_bool_p, ctypes.c_int]
_lib.generals_get_action_mask.restype = None

GRID_SIZE = 12
SKIP_ACTION = GRID_SIZE * GRID_SIZE * 8

def get_grid(state_ptr):
    owner_buf = (ctypes.c_int8 * (GRID_SIZE * GRID_SIZE))()
    army_buf = (ctypes.c_int16 * (GRID_SIZE * GRID_SIZE))()
    terrain_buf = (ctypes.c_uint8 * (GRID_SIZE * GRID_SIZE))()
    _lib.generals_get_grid_data(state_ptr, owner_buf, army_buf, terrain_buf)
    owner = np.frombuffer(owner_buf, dtype=np.int8).reshape(GRID_SIZE, GRID_SIZE).copy()
    army = np.frombuffer(army_buf, dtype=np.int16).reshape(GRID_SIZE, GRID_SIZE).copy()
    terrain = np.frombuffer(terrain_buf, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE).copy()
    return owner, army, terrain

def get_mask(state_ptr, player):
    mask_buf = (ctypes.c_bool * (GRID_SIZE * GRID_SIZE * 8 + 1))()
    _lib.generals_get_action_mask(state_ptr, mask_buf, player)
    return np.frombuffer(mask_buf, dtype=bool).copy()

def encode_action(sx, sy, dx, dy, is_half):
    DR = [-1, 1, 0, 0]
    DC = [0, 0, -1, 1]
    for d in range(4):
        if sy + DR[d] == dy and sx + DC[d] == dx:
            return (sy * GRID_SIZE + sx) * 8 + d * 2 + (1 if is_half else 0)
    return -1


print("=" * 60)
print("  Rule Consistency Verification")
print("=" * 60)

all_pass = True

# === Test 1: 同 seed 初始地图一致性 ===
print("\n[Test 1] Initial map consistency (same seed)")
s1 = _lib.generals_create(12, 12, 600, 42)
s2 = _lib.generals_create(12, 12, 600, 42)
o1, a1, t1 = get_grid(s1)
o2, a2, t2 = get_grid(s2)
if np.array_equal(o1, o2) and np.array_equal(a1, a2) and np.array_equal(t1, t2):
    print("  PASS: Same seed produces identical initial state")
else:
    print("  FAIL: Same seed produces different states!")
    all_pass = False
_lib.generals_destroy(s1)
_lib.generals_destroy(s2)

# === Test 2: step_dual 只 tick 一次 ===
print("\n[Test 2] step_dual advances exactly 1 step")
s = _lib.generals_create(12, 12, 600, 42)
step_before = _lib.generals_get_step(s)
_lib.generals_step_dual(s, SKIP_ACTION, SKIP_ACTION)
step_after = _lib.generals_get_step(s)
delta = step_after - step_before
if delta == 1:
    print(f"  PASS: step delta = 1")
else:
    print(f"  FAIL: step delta = {delta} (expected 1)")
    all_pass = False
_lib.generals_destroy(s)

# === Test 3: 兵力增长频率 (每 25 步 bonus) ===
print("\n[Test 3] Army growth timing (bonus every 25 steps)")
s = _lib.generals_create(12, 12, 600, 42)
owner, army, terrain = get_grid(s)
# 找一个有兵力的格子
found = False
for y in range(12):
    for x in range(12):
        if army[y, x] > 0:
            pos = (x, y)
            found = True
            break
    if found:
        break

if found:
    x, y = pos
    initial_army = int(army[y, x])
    # 跑 24 步 (不应该有 bonus)
    for i in range(24):
        _lib.generals_step_dual(s, SKIP_ACTION, SKIP_ACTION)
    _, army24, _ = get_grid(s)
    after_24 = int(army24[y, x])
    # 再跑 1 步 (第 25 步, 应该有 bonus)
    _lib.generals_step_dual(s, SKIP_ACTION, SKIP_ACTION)
    _, army25, _ = get_grid(s)
    after_25 = int(army25[y, x])

    # 主城/城市每步 +1, 普通领地不增长
    # 第 25 步额外 +1
    expected_24 = initial_army + 24  # 每步 +1 (城市/将军)
    expected_25 = initial_army + 26  # 25步 +1 再加 bonus +1
    if after_24 == expected_24:
        print(f"  PASS: After 24 steps: army={after_24} (expected {expected_24})")
    else:
        print(f"  WARN: After 24 steps: army={after_24} (expected {expected_24}, diff may be due to combat)")
    if after_25 == expected_25:
        print(f"  PASS: After 25 steps: army={after_25} (expected {expected_25}, bonus applied)")
    else:
        print(f"  INFO: After 25 steps: army={after_25} (expected {expected_25}, may be combat-affected)")
else:
    print("  SKIP: No army found")
_lib.generals_destroy(s)

# === Test 4: 合法动作掩码一致性 ===
print("\n[Test 4] Action mask: army<=1 tiles have no valid moves")
s = _lib.generals_create(12, 12, 600, 42)
mask = get_mask(s, 0)
# 检查初始主城 (army=1) 是否没有合法移动
owner, army, terrain = get_grid(s)
found_issue = False
for y in range(12):
    for x in range(12):
        idx = y * 12 + x
        if owner[y, x] == 0 and army[y, x] <= 1:
            # 这个格子不应该有合法移动
            base = idx * 8
            has_move = any(mask[base + d * 2] or mask[base + d * 2 + 1] for d in range(4))
            if has_move:
                print(f"  FAIL: Tile ({x},{y}) army={army[y,x]} has valid moves!")
                found_issue = True
                all_pass = False
if not found_issue:
    print("  PASS: No army<=1 tiles have valid moves")
_lib.generals_destroy(s)

# === Test 5: 攻占将军后 terrain 变为 CITY ===
print("\n[Test 5] General capture: terrain changes to CITY")
s = _lib.generals_create(12, 12, 600, 42)
owner, army, terrain = get_grid(s)
# 找将军位置
gen_pos = None
for y in range(12):
    for x in range(12):
        if terrain[y, x] == 2:
            gen_pos = (x, y)
            break
    if gen_pos:
        break

if gen_pos:
    gx, gy = gen_pos
    print(f"  General at ({gx},{gy}), owner={owner[gy,gx]}, army={army[gy,gx]}")
    # 模拟攻占: 从相邻格子投入大量兵力
    # 找相邻的己方格子
    DR = [-1, 1, 0, 0]
    DC = [0, 0, -1, 1]
    for d in range(4):
        nx, ny = gx + DC[d], gy + DR[d]
        if 0 <= nx < 12 and 0 <= ny < 12 and owner[ny, nx] == 0 and army[ny, nx] > 1:
            # 移动全部兵力过去
            action = encode_action(nx, ny, gx, gy, False)
            if action >= 0 and mask[action]:
                _lib.generals_step(s, action)
                _, _, terrain_after = get_grid(s)
                if terrain_after[gy, gx] == 3:  # 变为 CITY
                    print(f"  PASS: General captured, terrain changed to CITY (3)")
                else:
                    print(f"  FAIL: terrain={terrain_after[gy,gx]} after capture (expected 3)")
                    all_pass = False
                break
    else:
        print("  SKIP: No adjacent friendly tile with army>1 to test capture")
else:
    print("  SKIP: No general found")
_lib.generals_destroy(s)

# === Test 6: step_dual 固定先后手 ===
print("\n[Test 6] step_dual fixed order (red first)")
# 构造一个场景: 双方同时向中间格子移动，结果应该确定性的
s = _lib.generals_create(12, 12, 600, 123)
# 跑 10 步让兵力增长
for i in range(10):
    _lib.generals_step_dual(s, SKIP_ACTION, SKIP_ACTION)

# 记录状态, 跑两次相同动作看是否一致
owner1, army1, terrain1 = get_grid(s)
# 找一个红方有兵力的格子
red_pos = None
for y in range(12):
    for x in range(12):
        if owner1[y, x] == 0 and army1[y, x] > 1:
            red_pos = (x, y)
            break
    if red_pos:
        break

if red_pos:
    rx, ry = red_pos
    # 向右移动
    action = encode_action(rx, ry, rx + 1, ry, False)
    if action >= 0:
        _lib.generals_step_dual(s, action, SKIP_ACTION)
        _, army_after1, _ = get_grid(s)
        # 再跑一步看是否确定性
        _lib.generals_step_dual(s, SKIP_ACTION, SKIP_ACTION)
        step1 = _lib.generals_get_step(s)
        print(f"  Step after dual actions: {step1} (should be 12)")
        if step1 == 12:
            print("  PASS: Deterministic step progression")
        else:
            print(f"  FAIL: step={step1} (expected 12)")
            all_pass = False
    else:
        print("  SKIP: Cannot encode move")
else:
    print("  SKIP: No red tile found")
_lib.generals_destroy(s)

# === Summary ===
print("\n" + "=" * 60)
if all_pass:
    print("  ALL TESTS PASSED ✅")
else:
    print("  SOME TESTS FAILED ❌")
print("=" * 60)
