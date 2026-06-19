import numpy as np
import os, sys, ctypes
from collections import Counter

CPP_DIR = '/media/pyiming/C22AA0E82AA0DB25/project/generals.io/src/cpp'
sys.path.insert(0, '/media/pyiming/C22AA0E82AA0DB25/project/generals.io/src/python')

_lib = ctypes.CDLL(os.path.join(CPP_DIR, 'libgenerals.so'))

_lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.generals_create.restype = ctypes.c_void_p
_lib.generals_destroy.argtypes = [ctypes.c_void_p]
_lib.generals_destroy.restype = None
_lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.generals_step_dual.restype = ctypes.c_int
_lib.generals_get_winner.argtypes = [ctypes.c_void_p]
_lib.generals_get_winner.restype = ctypes.c_int
_lib.generals_get_step.argtypes = [ctypes.c_void_p]
_lib.generals_get_step.restype = ctypes.c_int
_lib.generals_get_grid_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint8)]
_lib.generals_get_grid_data.restype = None

GRID_SIZE = 12
SKIP = GRID_SIZE * GRID_SIZE * 8
DR = [-1, 1, 0, 0]
DC = [0, 0, -1, 1]

def get_grid(s):
    o = (ctypes.c_int8 * 144)()
    a = (ctypes.c_int16 * 144)()
    t = (ctypes.c_uint8 * 144)()
    _lib.generals_get_grid_data(s, o, a, t)
    return (np.frombuffer(o, dtype=np.int8).reshape(12,12).copy(),
            np.frombuffer(a, dtype=np.int16).reshape(12,12).copy(),
            np.frombuffer(t, dtype=np.uint8).reshape(12,12).copy())

def encode_action(sx, sy, dx, dy, half=False):
    for d in range(4):
        if sy + DR[d] == dy and sx + DC[d] == dx:
            return (sy * 12 + sx) * 8 + d * 2 + (1 if half else 0)
    return -1

from script_bot import ScriptBot

state = _lib.generals_create(12, 12, 600, 42)
blue_bot = ScriptBot(player_id=1, grid_size=12)

owner, army, terrain = get_grid(state)

print('=== ScriptBot vs Skip (red) ===')

action_log = []

for step in range(300):
    if _lib.generals_get_winner(state) != -1:
        print(f'Game over! Winner: {_lib.generals_get_winner(state)}, Step: {step}')
        break
    
    fog = np.zeros((12,12), dtype=bool)
    blue_action = blue_bot.get_action(owner, army, terrain, fog)
    
    if blue_action:
        sy, sx, ny, nx, h = blue_action
        blue_enc = encode_action(sx, sy, nx, ny, h)
        action_log.append((step, sx, sy, nx, ny, int(army[sy, sx])))
    else:
        blue_enc = SKIP
        action_log.append((step, None))
    
    _lib.generals_step_dual(state, SKIP, blue_enc)
    owner, army, terrain = get_grid(state)

# Analysis
print(f'Total steps: {len(action_log)}')
print(f'Total actions: {sum(1 for a in action_log if a[1] is not None)}')

# Max army
max_a = 0
max_pos = None
for y in range(12):
    for x in range(12):
        if army[y,x] > max_a:
            max_a = army[y,x]
            max_pos = (x, y)
print(f'Max army tile: {max_pos} = {max_a}')

# Top source tiles
src_counter = Counter()
for entry in action_log:
    if entry[1] is not None:
        _, sx, sy, nx, ny, arm = entry
        src_counter[(sx, sy)] += 1

print('\n=== High-frequency source tiles (>3 times) ===')
for (x, y), cnt in src_counter.most_common(10):
    if cnt > 3:
        print(f'  ({x},{y}): {cnt} times, current army={army[y,x]}')

# Detect backline shuffling (front=0 -> front=0 with >5 troops)
front_values = {}
for y in range(12):
    for x in range(12):
        if owner[y,x] == 1:
            front = 0
            for d in range(4):
                ny, nx = y+DR[d], x+DC[d]
                if 0 <= ny < 12 and 0 <= nx < 12:
                    if owner[ny,nx] != 1:
                        front += 1
            front_values[(x,y)] = front

backline_shuffle = 0
backline_details = []
for entry in action_log:
    if entry[1] is not None:
        _, sx, sy, nx, ny, arm = entry
        src_front = front_values.get((sx,sy), 0)
        dst_front = front_values.get((nx,ny), 0)
        if src_front == 0 and dst_front == 0 and arm > 5:
            backline_shuffle += 1
            if len(backline_details) < 10:
                backline_details.append(f'  Step {entry[0]}: ({sx},{sy})->({nx},{ny}) arm={arm} (both backline)')

print(f'\nBackline->backline moves (>5 troops): {backline_shuffle}')
if backline_details:
    print('Examples:')
    for d in backline_details:
        print(d)

# Detect "feeding" - moving to already-strong tiles
feeding = 0
for entry in action_log:
    if entry[1] is not None:
        _, sx, sy, nx, ny, arm = entry
        if army[ny,nx] > 20:
            feeding += 1

print(f'\nFeeding into >20 army tile: {feeding}')

# Last 20 actions
print('\n=== Last 20 actions ===')
for entry in action_log[-20:]:
    if entry[1] is not None:
        _, sx, sy, nx, ny, arm = entry
        print(f'  Step {entry[0]:3d}: ({sx},{sy})->({nx},{ny}) arm={arm}')
    else:
        print(f'  Step {entry[0]:3d}: SKIP')

# Final
blue_tiles = np.sum(owner == 1)
blue_army = int(np.sum(army[owner == 1]))
print(f'\nFinal: Blue {blue_tiles} tiles / {blue_army} army')

_lib.generals_destroy(state)
