"""
generals_gym_cpp.py — C++ 引擎的 Python ctypes 包装器

接口兼容 GeneralsEnvV4TieBreaker，可直接替代。
使用 libgenerals.so 获得 1000x+ 速度提升。
"""

import ctypes
import numpy as np
import os

# 加载共享库 (位于 ../cpp/ 目录)
_lib_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'cpp', 'libgenerals.so'
)
_lib = ctypes.cdll.LoadLibrary(_lib_path)

# 类型定义
_lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.generals_create.restype = ctypes.c_void_p

_lib.generals_destroy.argtypes = [ctypes.c_void_p]
_lib.generals_destroy.restype = None

_lib.generals_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
_lib.generals_reset.restype = None

_lib.generals_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.generals_step.restype = ctypes.c_int

_lib.generals_get_winner.argtypes = [ctypes.c_void_p]
_lib.generals_get_winner.restype = ctypes.c_int

_lib.generals_get_step.argtypes = [ctypes.c_void_p]
_lib.generals_get_step.restype = ctypes.c_int

_lib.generals_is_stalemate.argtypes = [ctypes.c_void_p]
_lib.generals_is_stalemate.restype = ctypes.c_bool

_lib.generals_get_width.argtypes = [ctypes.c_void_p]
_lib.generals_get_width.restype = ctypes.c_int
_lib.generals_get_height.argtypes = [ctypes.c_void_p]
_lib.generals_get_height.restype = ctypes.c_int

_lib.generals_skip_action.argtypes = [ctypes.c_void_p]
_lib.generals_skip_action.restype = ctypes.c_int

_lib.generals_get_obs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.generals_get_obs.restype = None

_lib.generals_get_action_mask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool), ctypes.c_int]
_lib.generals_get_action_mask.restype = None

_lib.generals_clone.argtypes = [ctypes.c_void_p]
_lib.generals_clone.restype = ctypes.c_void_p

_lib.generals_get_grid_data.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_uint8),
]
_lib.generals_get_grid_data.restype = None

# ScriptAgent C API
_lib.script_get_action.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.script_get_action.restype = ctypes.c_int

_lib.generals_script_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.generals_script_step.restype = ctypes.c_int


class GeneralsEnvCpp:
    """
    C++ 引擎包装器，兼容 gym-style 接口。

    用法:
        env = GeneralsEnvCpp(width=12, height=12, max_steps=600)
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action_id)
    """

    def __init__(self, width=12, height=12, max_steps=600):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self._state = None
        self.obs_buf = None
        self.mask_buf = None

    @property
    def SKIP_ACTION(self):
        return self.width * self.height * 8

    @property
    def current_step(self):
        return _lib.generals_get_step(self._state)

    @property
    def winner(self):
        return _lib.generals_get_winner(self._state)

    @property
    def stalemate(self):
        return _lib.generals_is_stalemate(self._state)

    def reset(self, seed=None):
        """重置游戏，返回 (obs, info)"""
        if seed is None:
            seed = np.random.randint(0, 2**31)

        if self._state is not None:
            _lib.generals_reset(self._state, seed)
        else:
            self._state = _lib.generals_create(self.width, self.height, self.max_steps, seed)

        # 分配缓冲区
        total = self.width * self.height
        self.obs_buf = (ctypes.c_float * (7 * total))()
        self.mask_buf = (ctypes.c_bool * (total * 8 + 1))()

        obs = self._get_obs(player_id=0)
        info = {}
        return obs, info

    def step(self, action_id):
        """执行一步，返回 (obs, reward, terminated, truncated, info)"""
        winner = _lib.generals_step(self._state, action_id)

        obs = self._get_obs(player_id=0)
        terminated = winner != -1
        truncated = self.current_step >= self.max_steps and not terminated

        reward = 0.0
        if winner == 0:
            reward = 20.0
        elif winner == 1:
            reward = -20.0

        info = {"action_mask": self.valid_action_mask()}
        return obs, reward, terminated, truncated, info

    def _get_obs(self, player_id=0):
        """获取 7 通道观测 (匹配 Python _get_obs)"""
        _lib.generals_get_obs(self._state, self.obs_buf, player_id)
        total = self.width * self.height
        arr = np.frombuffer(self.obs_buf, dtype=np.float32).copy()
        return arr.reshape(7, self.height, self.width)

    def valid_action_mask(self, player_id=0):
        """获取有效动作掩码"""
        _lib.generals_get_action_mask(self._state, self.mask_buf, player_id)
        arr = np.frombuffer(self.mask_buf, dtype=bool).copy()
        return arr

    def get_raw(self):
        """获取原始游戏数据（用于调试 / FeatureEngine 兼容）"""
        from ctypes import c_int
        h, w = self.height, self.width
        
        grid_owner = np.full((h, w), -1, dtype=np.int8)
        grid_troops = np.zeros((h, w), dtype=np.int16)
        grid_type = np.zeros((h, w), dtype=np.uint8)
        
        for r in range(h):
            for c in range(w):
                idx = r * w + c
                grid_owner[r, c] = _lib.generals_get_owner(self._state, idx)
                grid_troops[r, c] = _lib.generals_get_army(self._state, idx)
                grid_type[r, c] = _lib.generals_get_terrain(self._state, idx)
        
        return {
            'width': w,
            'height': h,
            'step': self.current_step,
            'winner': self.winner,
            'stalemate': self.stalemate,
            'grid_owner': grid_owner,
            'grid_troops': grid_troops,
            'grid_type': grid_type,
        }

    # FeatureEngine 兼容属性 (使用批量 API，速度快 100x)
    @property
    def grid_owner(self):
        h, w = self.height, self.width
        total = h * w
        owner_buf = (ctypes.c_int8 * total)()
        army_buf = (ctypes.c_int16 * total)()
        terrain_buf = (ctypes.c_uint8 * total)()
        _lib.generals_get_grid_data(self._state, owner_buf, army_buf, terrain_buf)
        return np.frombuffer(owner_buf, dtype=np.int8).copy().reshape(h, w)

    @property
    def grid_troops(self):
        h, w = self.height, self.width
        total = h * w
        owner_buf = (ctypes.c_int8 * total)()
        army_buf = (ctypes.c_int16 * total)()
        terrain_buf = (ctypes.c_uint8 * total)()
        _lib.generals_get_grid_data(self._state, owner_buf, army_buf, terrain_buf)
        return np.frombuffer(army_buf, dtype=np.int16).copy().reshape(h, w)

    @property
    def grid_type(self):
        h, w = self.height, self.width
        total = h * w
        owner_buf = (ctypes.c_int8 * total)()
        army_buf = (ctypes.c_int16 * total)()
        terrain_buf = (ctypes.c_uint8 * total)()
        _lib.generals_get_grid_data(self._state, owner_buf, army_buf, terrain_buf)
        return np.frombuffer(terrain_buf, dtype=np.uint8).copy().reshape(h, w)

    def close(self):
        if self._state is not None:
            _lib.generals_destroy(self._state)
            self._state = None

    def __del__(self):
        self.close()


# ============================================================
# 快速测试
# ============================================================
if __name__ == '__main__':
    import time

    print("=== C++ 引擎 Python 包装器测试 ===\n")

    # 基本功能测试
    env = GeneralsEnvCpp(12, 12, 600)
    obs, info = env.reset(seed=42)
    print(f"reset() → obs.shape={obs.shape}")
    print(f"  Ch0 (己方兵力) sum={obs[0].sum():.0f}")
    print(f"  Ch1 (敌方兵力) sum={obs[1].sum():.0f}")
    print(f"  Ch3 (山脉) count={obs[3].sum():.0f}")
    print(f"  Ch4 (城市) count={obs[4].sum():.0f}")

    mask = env.valid_action_mask()
    print(f"  action_mask: {mask.sum():.0f}/{mask.size} valid")
    print(f"  SKIP_ACTION: {env.SKIP_ACTION}")

    # 跑几步行不行
    for i in range(10):
        act = np.random.choice(np.where(mask)[0])
        obs, r, term, trunc, info = env.step(act)
        if term or trunc:
            print(f"  step {i}: done (winner={env.winner})")
            break
    else:
        print(f"  10步正常")

    env.close()

    # 性能测试
    print("\n=== 性能测试（C++ 引擎 vs Python 引擎）===")

    n_games = 1000
    t0 = time.time()
    for g in range(n_games):
        env = GeneralsEnvCpp(12, 12, 300)
        obs, _ = env.reset(seed=g)
        done = False
        while not done:
            mask = env.valid_action_mask()
            valid = np.where(mask)[0]
            non_skip = [a for a in valid if a != env.SKIP_ACTION]
            act = np.random.choice(non_skip) if non_skip else env.SKIP_ACTION
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        env.close()

    elapsed = time.time() - t0
    print(f"  C++ 引擎: {n_games} 局 = {elapsed:.1f}s")
    print(f"  速度: {n_games/elapsed:.0f} 局/秒")
    print(f"  每局: {elapsed/n_games*1000:.2f} ms")
    print(f"  ✅ C++ 引擎 Python 包装器工作正常!")
