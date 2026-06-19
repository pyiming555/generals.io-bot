"""
play_gui.py — Generals.io RL Debugger & Human-vs-AI对战界面

功能:
  - 12x12 网格渲染 (地形/兵力/颜色/迷雾)
  - 人类鼠标下棋 (左键选起点, 左键/右键移动)
  - AI 回合: C++ MCTS + NN 推理 (通过 BeliefState 保持迷雾)
  - 复盘模式: ← → 键穿梭历史
  - 迷雾切换: Tab 键开关全图视野
  - 状态栏: 当前步数/模式/AI 思考时间

用法:
  python3 play_gui.py [--model PATH] [--n-mcts 260] [--human-player 0]
"""

import pygame
import numpy as np
import ctypes
import os
import sys
import time
import argparse

# --- 加载 C++ 引擎 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CPP_DIR = os.path.join(PROJECT_DIR, "src", "cpp")

# libgenerals.so (游戏引擎)
_lib_path = os.path.join(CPP_DIR, "libgenerals.so")
_lib = ctypes.CDLL(_lib_path)

# libgenerals_nn.so (NN 推理 + MCTS)
_lib_nn_path = os.path.join(CPP_DIR, "libgenerals_nn.so")
_lib_nn = ctypes.CDLL(_lib_nn_path)

# --- C 类型定义 ---
_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_bool_p = ctypes.POINTER(ctypes.c_bool)
_c_int8_p = ctypes.POINTER(ctypes.c_int8)
_c_int16_p = ctypes.POINTER(ctypes.c_int16)
_c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
_c_void_p = ctypes.c_void_p

# generals_create
_lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.generals_create.restype = _c_void_p
_lib.generals_destroy.argtypes = [_c_void_p]
_lib.generals_destroy.restype = None
_lib.generals_reset.argtypes = [_c_void_p, ctypes.c_uint]
_lib.generals_reset.restype = None
_lib.generals_step.argtypes = [_c_void_p, ctypes.c_int]
_lib.generals_step.restype = ctypes.c_int
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

# belief_create / observe / destroy
_lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.belief_create.restype = _c_void_p
_lib.belief_destroy.argtypes = [_c_void_p]
_lib.belief_destroy.restype = None
_lib.belief_observe.argtypes = [_c_void_p, _c_void_p, ctypes.c_int]
_lib.belief_observe.restype = None

# mcts_create / mcts_destroy (在 libgenerals.so 中)
_lib.mcts_create.argtypes = [ctypes.c_uint]
_lib.mcts_create.restype = _c_void_p
_lib.mcts_destroy.argtypes = [_c_void_p]
_lib.mcts_destroy.restype = None

# mcts_search_flow (纯 MCTS, 在 libgenerals.so 中, 用于无 NN 时的 fallback)
_lib.mcts_search_flow.argtypes = [_c_void_p, _c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.mcts_search_flow.restype = ctypes.c_int

# mcts_search_nn_auto (在 libgenerals_nn.so 中)
_lib_nn.mcts_search_nn_auto.argtypes = [
    _c_void_p,  # ISMCTSEngine*
    _c_void_p,  # BeliefState*
    ctypes.c_int,  # player
    ctypes.c_int,  # n_det
    ctypes.c_int,  # n_mcts
    _c_void_p,  # NNPredictor*
]
_lib_nn.mcts_search_nn_auto.restype = ctypes.c_int

# nn_create / nn_destroy
_lib_nn.nn_create.argtypes = [ctypes.c_char_p]
_lib_nn.nn_create.restype = _c_void_p
_lib_nn.nn_destroy.argtypes = [_c_void_p]
_lib_nn.nn_destroy.restype = None

# --- 常量 ---
GRID_SIZE = 12
TILE_SIZE = 50
UI_HEIGHT = 100
WIDTH = GRID_SIZE * TILE_SIZE
HEIGHT = GRID_SIZE * TILE_SIZE + UI_HEIGHT

# 颜色
COLORS = {
    "bg": (10, 10, 10),
    "red": (200, 60, 60),
    "red_dark": (140, 30, 30),
    "blue": (60, 60, 200),
    "blue_dark": (30, 30, 140),
    "gray": (120, 120, 120),
    "mountain": (45, 45, 45),
    "city": (180, 160, 40),
    "general_red": (255, 100, 100),
    "general_blue": (100, 100, 255),
    "fog": (15, 15, 15),
    "select": (255, 255, 255),
    "hover": (255, 255, 100),
    "text": (255, 255, 255),
    "text_dim": (180, 180, 180),
    "ui_bg": (30, 30, 30),
    "ui_border": (60, 60, 60),
    "arrow_red": (255, 150, 150, 120),
    "arrow_blue": (150, 150, 255, 120),
}

# 方向向量 (上/下/左/右)
DR = [-1, 1, 0, 0]
DC = [0, 0, -1, 1]


class GeneralsGUI:
    def __init__(self, model_path, n_mcts, human_player, seed):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Generals.io RL Debugger - Human vs AI")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("arial", 22, bold=True)
        self.font = pygame.font.SysFont("arial", 16)
        self.font_small = pygame.font.SysFont("arial", 13)

        # --- 游戏配置 ---
        self.n_mcts = n_mcts
        self.human_player = human_player
        self.ai_player = 1 - human_player

        # --- C++ 引擎 ---
        self.state = _lib.generals_create(GRID_SIZE, GRID_SIZE, 600, seed)
        self.mcts_engine = _lib.mcts_create(seed)

        # --- AI BeliefState ---
        self.ai_belief = _lib.belief_create(GRID_SIZE, GRID_SIZE, self.ai_player)

        # --- NN Predictor ---
        self.nn_ptr = None
        # 如果未指定模型，自动搜索默认模型
        if model_path is None:
            for candidate in ["v4_attention.ptl", "v4_attention_80pct.ptl", "resnet_v4.ptl"]:
                alt = os.path.join(SCRIPT_DIR, "rl_models", candidate)
                if os.path.exists(alt):
                    model_path = alt
                    break
        if model_path:
            # 确保路径为绝对路径
            abs_path = model_path
            if not os.path.isabs(abs_path):
                abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, abs_path))
            # 如果路径不存在，尝试在 SCRIPT_DIR/rl_models/ 下查找
            if not os.path.exists(abs_path):
                alt = os.path.join(SCRIPT_DIR, "rl_models", os.path.basename(model_path))
                if os.path.exists(alt):
                    abs_path = alt
            if os.path.exists(abs_path):
                self.nn_ptr = _lib_nn.nn_create(abs_path.encode("utf-8"))
                if self.nn_ptr:
                    print(f"[GUI] NN 模型加载成功: {abs_path}")
                else:
                    print(f"[GUI] 警告: NN 模型加载失败，AI 将使用纯 MCTS")
            else:
                print(f"[GUI] 警告: 模型文件不存在 ({abs_path})，AI 将使用纯 MCTS")

        # --- 历史记录 (存 clone 的指针) ---
        self.history = []
        self.ai_think_time = []  # 每步 AI 思考耗时
        self.save_history()

        # --- 常量 ---
        self.SKIP_ACTION = GRID_SIZE * GRID_SIZE * 8

        # --- 交互状态 ---
        self.selected_tile = None
        self.hover_tile = None
        self.cursor_tile = None  # 键盘光标位置 (None=未启用, (x,y)=启用)
        self.mode = "PLAY"
        self.current_step_idx = 0
        self.show_fog = True
        self.game_over = False
        self.winner = -1
        self.ai_thinking = False

        # --- 预分配缓冲区 ---
        self.grid_owner_buf = (ctypes.c_int8 * (GRID_SIZE * GRID_SIZE))()
        self.grid_army_buf = (ctypes.c_int16 * (GRID_SIZE * GRID_SIZE))()
        self.grid_terrain_buf = (ctypes.c_uint8 * (GRID_SIZE * GRID_SIZE))()
        self.mask_buf = (ctypes.c_bool * (GRID_SIZE * GRID_SIZE * 8 + 1))()

    def save_history(self):
        """保存当前 GameState 的 clone 到历史"""
        clone = _lib.generals_clone(self.state)
        self.history.append(clone)
        self.current_step_idx = len(self.history) - 1

    def get_grid_data(self, state_ptr):
        """从 C++ GameState 获取网格数据"""
        _lib.generals_get_grid_data(state_ptr, self.grid_owner_buf, self.grid_army_buf, self.grid_terrain_buf)
        owner = np.frombuffer(self.grid_owner_buf, dtype=np.int8).reshape(GRID_SIZE, GRID_SIZE).copy()
        army = np.frombuffer(self.grid_army_buf, dtype=np.int16).reshape(GRID_SIZE, GRID_SIZE).copy()
        terrain = np.frombuffer(self.grid_terrain_buf, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE).copy()
        return owner, army, terrain

    def get_action_mask(self, state_ptr, player_id):
        _lib.generals_get_action_mask(state_ptr, self.mask_buf, player_id)
        return np.frombuffer(self.mask_buf, dtype=bool).copy()

    def get_tile_at(self, pos):
        """鼠标坐标 → 网格坐标"""
        x, y = pos[0] // TILE_SIZE, pos[1] // TILE_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            return (x, y)
        return None

    def is_adjacent(self, t1, t2):
        """检查是否相邻 (上下左右)"""
        return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1]) == 1

    def encode_action(self, sx, sy, dx, dy, is_half):
        """编码动作: (src, dst, half) → action_id"""
        direction = -1
        for d in range(4):
            if sy + DR[d] == dy and sx + DC[d] == dx:
                direction = d
                break
        if direction < 0:
            return -1
        return (sy * GRID_SIZE + sx) * 8 + direction * 2 + (1 if is_half else 0)

    def decode_action(self, action_id):
        """解码 action_id → (sx, sy, dx, dy, is_half)"""
        if action_id == GRID_SIZE * GRID_SIZE * 8:
            return None  # SKIP
        is_half = action_id % 2
        action_id //= 2
        direction = action_id % 4
        action_id //= 4
        sx = action_id % GRID_SIZE
        sy = action_id // GRID_SIZE
        dx = sx + DR[direction]
        dy = sy + DC[direction]
        return (sx, sy, dx, dy, is_half)

    def trigger_ai_turn(self):
        """AI 回合: 通过 BeliefState 做 MCTS+NN 推理"""
        if self.game_over:
            return

        self.ai_thinking = True
        t0 = time.time()

        # 1. 让 AI 观察真实世界 (更新 BeliefState)
        current_step = _lib.generals_get_step(self.state)
        _lib.belief_observe(self.ai_belief, self.state, current_step)

        # 2. MCTS + NN 搜索 (在 libgenerals_nn.so 中)
        if self.nn_ptr:
            action_id = _lib_nn.mcts_search_nn_auto(
                self.mcts_engine,
                self.ai_belief,
                self.ai_player,
                4,  # n_det
                self.n_mcts,
                self.nn_ptr,
            )
        else:
            # 纯 MCTS (无 NN) — 在 libgenerals.so 中
            action_id = _lib.mcts_search_flow(self.mcts_engine, self.ai_belief, self.ai_player, 4, self.n_mcts)

        elapsed = time.time() - t0
        self.ai_think_time.append(elapsed)
        self.ai_thinking = False

        # 3. 执行动作
        if action_id >= 0:
            _lib.generals_step(self.state, action_id)
            self.save_history()

        # 检查游戏结束
        self.winner = _lib.generals_get_winner(self.state)
        if self.winner != -1 or _lib.generals_is_stalemate(self.state):
            self.game_over = True

    def handle_human_action(self, pos, button):
        """处理人类玩家的鼠标动作"""
        if self.game_over or self.ai_thinking:
            return

        tile = self.get_tile_at(pos)
        if tile is None:
            self.selected_tile = None
            return

        # 如果键盘光标已激活, 鼠标点击优先
        self.cursor_tile = None

        if self.selected_tile is None:
            # 选中起点: 必须是己方领地 (主城1兵也可以选，用于后续skip)
            owner, army, terrain = self.get_grid_data(self.state)
            x, y = tile
            if owner[y, x] == self.human_player and army[y, x] >= 1:
                self.selected_tile = tile
        else:
            # 移动或取消选中
            sx, sy = self.selected_tile
            dx, dy = tile

            if (dx, dy) == (sx, sy):
                # 点击自己 = 取消选中
                self.selected_tile = None
                return

            if self.is_adjacent((sx, sy), (dx, dy)):
                is_half = (button == 3)  # 右键 = 半兵
                action_id = self.encode_action(sx, sy, dx, dy, is_half)

                if action_id >= 0:
                    # 检查合法性
                    mask = self.get_action_mask(self.state, self.human_player)
                    if mask[action_id]:
                        _lib.generals_step(self.state, action_id)
                        self.save_history()
                        self.selected_tile = None

                        # 检查游戏结束
                        self.winner = _lib.generals_get_winner(self.state)
                        if self.winner != -1 or _lib.generals_is_stalemate(self.state):
                            self.game_over = True
                            return

                        # 触发 AI 回合
                        self.trigger_ai_turn()
                    else:
                        self.selected_tile = None
            else:
                # 点击不相邻的格子: 如果点击己方领地则切换选中
                owner, army, terrain = self.get_grid_data(self.state)
                if owner[dy, dx] == self.human_player and army[dy, dx] >= 1:
                    self.selected_tile = tile
                else:
                    self.selected_tile = None

    def handle_keyboard_move(self, direction):
        """处理键盘移动: 0=上 1=下 2=左 3=右, 返回是否成功"""
        if self.game_over or self.ai_thinking or self.mode != "PLAY":
            return False

        # 获取当前光标位置 (首次使用选己方有兵力的格子)
        if self.cursor_tile is None:
            owner, army, terrain = self.get_grid_data(self.state)
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if owner[y, x] == self.human_player and army[y, x] > 1:
                        self.cursor_tile = (x, y)
                        break
                if self.cursor_tile:
                    break
            if self.cursor_tile is None:
                return False

        cx, cy = self.cursor_tile
        nx, ny = cx + DC[direction], cy + DR[direction]

        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            return False

        # 如果当前有选中格子, 键盘移动方向 = 确认移动
        if self.selected_tile is not None:
            sx, sy = self.selected_tile
            if (nx, ny) == (sx, sy):
                self.selected_tile = None
                self.cursor_tile = (nx, ny)
                return True
            if self.is_adjacent((sx, sy), (nx, ny)):
                action_id = self.encode_action(sx, sy, nx, ny, False)
                mask = self.get_action_mask(self.state, self.human_player)
                if mask[action_id]:
                    _lib.generals_step(self.state, action_id)
                    self.save_history()
                    self.selected_tile = None
                    self.cursor_tile = (nx, ny)
                    self.winner = _lib.generals_get_winner(self.state)
                    if self.winner != -1 or _lib.generals_is_stalemate(self.state):
                        self.game_over = True
                        return True
                    self.trigger_ai_turn()
                    return True
            # 不可移动: 切换选中目标
            owner, army, terrain = self.get_grid_data(self.state)
            if owner[ny, nx] == self.human_player and army[ny, nx] >= 1:
                self.selected_tile = (nx, ny)
                self.cursor_tile = (nx, ny)
            else:
                self.selected_tile = None
                self.cursor_tile = (nx, ny)
            return True
        else:
            # 无选中: 移动光标
            self.cursor_tile = (nx, ny)
            return True

    def do_skip(self):
        """人类玩家跳过回合"""
        if self.game_over or self.ai_thinking:
            return
        skip_action = self.SKIP_ACTION
        _lib.generals_step(self.state, skip_action)
        self.save_history()
        self.selected_tile = None

        # 检查游戏结束
        self.winner = _lib.generals_get_winner(self.state)
        if self.winner != -1 or _lib.generals_is_stalemate(self.state):
            self.game_over = True
            return

        # 触发 AI 回合
        self.trigger_ai_turn()

    def get_skip_button_rect(self):
        """返回 Skip 按钮的 pygame.Rect (如果当前是人类回合则显示)"""
        if self.mode != "PLAY" or self.game_over or self.ai_thinking:
            return None
        # 按钮在右下角
        btn_w, btn_h = 80, 30
        btn_x = WIDTH - btn_w - 10
        btn_y = GRID_SIZE * TILE_SIZE + (UI_HEIGHT - btn_h) // 2
        return pygame.Rect(btn_x, btn_y, btn_w, btn_h)

    def draw_tile(self, x, y, owner, army, terrain, fog_mask):
        """绘制单个格子"""
        rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 2, TILE_SIZE - 2)

        is_fogged = self.show_fog and fog_mask[y, x]

        if is_fogged:
            # 迷雾区: 只画地形轮廓 (暗色)，不画兵力
            t = terrain[y, x]
            if t == 1:  # MOUNTAIN
                color = (25, 25, 25)
            elif t == 3:  # CITY
                color = (60, 55, 20)
            elif t == 2:  # GENERAL
                color = (40, 40, 40)
            else:
                color = COLORS["fog"]
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            # 迷雾标记: 画一个半透明层
            s = pygame.Surface((TILE_SIZE - 2, TILE_SIZE - 2), pygame.SRCALPHA)
            s.fill((0, 0, 0, 120))
            self.screen.blit(s, rect)
            return

        # 非迷雾区: 正常绘制
        t = terrain[y, x]
        o = owner[y, x]
        a = army[y, x]

        if t == 1:  # MOUNTAIN
            color = COLORS["mountain"]
        elif t == 3:  # CITY
            color = COLORS["city"]
        elif t == 2:  # GENERAL
            color = COLORS["general_red"] if o == 0 else COLORS["general_blue"]
        elif o == 0:
            color = COLORS["red"] if a > 0 else COLORS["red_dark"]
        elif o == 1:
            color = COLORS["blue"] if a > 0 else COLORS["blue_dark"]
        else:
            color = COLORS["gray"]

        pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # 兵力数字
        if a > 0 and t != 1:
            # 根据背景亮度选文字色
            brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
            text_color = (255, 255, 255) if brightness < 128 else (20, 20, 20)
            txt = self.font.render(str(int(a)), True, text_color)
            txt_rect = txt.get_rect(center=rect.center)
            self.screen.blit(txt, txt_rect)

        # 选中框 (白色粗框)
        if self.selected_tile == (x, y):
            pygame.draw.rect(self.screen, COLORS["select"], rect, 3, border_radius=3)

        # 键盘光标框 (青色细框)
        if self.cursor_tile == (x, y) and self.mode == "PLAY" and not self.game_over:
            pygame.draw.rect(self.screen, (0, 200, 200), rect, 2, border_radius=3)

        # 悬停高亮
        if self.hover_tile == (x, y) and self.mode == "PLAY" and not self.game_over:
            pygame.draw.rect(self.screen, COLORS["hover"], rect, 2, border_radius=3)

    def draw_arrow(self, sx, sy, dx, dy, direction, alpha=120):
        """绘制移动方向箭头 (用于显示 AI 上一步动作)"""
        if self.show_fog and self.human_player == 0:
            # 简化: 迷雾模式下不画箭头
            pass

        cx = sx * TILE_SIZE + TILE_SIZE // 2
        cy = sy * TILE_SIZE + TILE_SIZE // 2
        ex = dx * TILE_SIZE + TILE_SIZE // 2
        ey = dy * TILE_SIZE + TILE_SIZE // 2

        # 箭头线
        color = (*COLORS["select"][:3], alpha)
        pygame.draw.line(self.screen, color[:3], (cx, cy), (ex, ey), 3)

        # 箭头头部
        arrow_size = 8
        angle = np.arctan2(ey - cy, ex - cx)
        a1 = angle + np.pi * 0.8
        a2 = angle - np.pi * 0.8
        p1 = (ex + int(arrow_size * np.cos(a1)), ey + int(arrow_size * np.sin(a1)))
        p2 = (ex + int(arrow_size * np.cos(a2)), ey + int(arrow_size * np.sin(a2)))
        pygame.draw.polygon(self.screen, color[:3], [(ex, ey), p1, p2])

    def draw_ui(self):
        """绘制底部 UI"""
        ui_y = GRID_SIZE * TILE_SIZE
        ui_rect = pygame.Rect(0, ui_y, WIDTH, UI_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["ui_bg"], ui_rect)
        pygame.draw.line(self.screen, COLORS["ui_border"], (0, ui_y), (WIDTH, ui_y), 2)

        state_ptr = self.history[self.current_step_idx]
        owner, army, terrain = self.get_grid_data(state_ptr)
        current_step = _lib.generals_get_step(state_ptr) if self.mode == "PLAY" else self.current_step_idx

        # --- 左侧: 模式信息 ---
        mode_text = f"Mode: {self.mode}"
        if self.ai_thinking:
            mode_text += " | AI Thinking..."
        if self.game_over:
            if self.winner == self.human_player:
                mode_text += " | 🏆 You Win!"
            elif self.winner == self.ai_player:
                mode_text += " | 🤖 AI Wins!"
            elif self.winner == 2:
                mode_text += " | 🤝 Draw!"

        txt_mode = self.font.render(mode_text, True, COLORS["text"])
        self.screen.blit(txt_mode, (10, ui_y + 8))

        step_text = f"Step: {current_step} | History: {self.current_step_idx}/{len(self.history)-1}"
        txt_step = self.font_small.render(step_text, True, COLORS["text_dim"])
        self.screen.blit(txt_step, (10, ui_y + 32))

        fog_text = f"Fog: {'ON' if self.show_fog else 'OFF'} (F to toggle)"
        txt_fog = self.font_small.render(fog_text, True, COLORS["text_dim"])
        self.screen.blit(txt_fog, (10, ui_y + 50))

        # --- 中间: 玩家信息 ---
        # 红方 (玩家0)
        red_army = np.sum(owner[owner == 0]) if np.any(owner == 0) else 0
        red_tiles = np.sum(owner == 0)
        p0_label = "🔴 You" if self.human_player == 0 else "🔴 AI"
        txt_red = self.font.render(f"{p0_label}: {red_tiles} tiles, {red_army} army", True, COLORS["red"])
        self.screen.blit(txt_red, (WIDTH // 2 - 150, ui_y + 8))

        # 蓝方 (玩家1)
        blue_army = np.sum(army[owner == 1]) if np.any(owner == 1) else 0
        blue_tiles = np.sum(owner == 1)
        p1_label = "🔵 AI" if self.ai_player == 1 else "🔵 You"
        txt_blue = self.font.render(f"{p1_label}: {blue_tiles} tiles, {blue_army} army", True, COLORS["blue"])
        self.screen.blit(txt_blue, (WIDTH // 2 - 150, ui_y + 30))

        # --- 右侧: AI 思考时间 ---
        if self.ai_think_time:
            last_t = self.ai_think_time[-1]
            avg_t = np.mean(self.ai_think_time[-10:])
            t_text = f"AI: {last_t*1000:.0f}ms (avg {avg_t*1000:.0f}ms)"
            txt_time = self.font_small.render(t_text, True, COLORS["text_dim"])
            self.screen.blit(txt_time, (WIDTH - 200, ui_y + 8))

        # --- 底部: 操作提示 ---
        # --- Skip 按钮 ---
        skip_rect = self.get_skip_button_rect()
        if skip_rect:
            # 按钮背景
            pygame.draw.rect(self.screen, (80, 80, 80), skip_rect, border_radius=5)
            pygame.draw.rect(self.screen, COLORS["text_dim"], skip_rect, 1, border_radius=5)
            # 按钮文字
            txt_skip = self.font.render("Skip", True, COLORS["text"])
            txt_rect = txt_skip.get_rect(center=skip_rect.center)
            self.screen.blit(txt_skip, txt_rect)

        hints = "WASD/Arrows: move | Space: skip | Enter/Click: select | <: replay | F: fog | R: restart | Q: quit"
        txt_hint = self.font_small.render(hints, True, COLORS["text_dim"])
        self.screen.blit(txt_hint, (WIDTH // 2 - 160, ui_y + 72))

    def render(self):
        """渲染一帧"""
        self.screen.fill(COLORS["bg"])

        state_ptr = self.history[self.current_step_idx]
        owner, army, terrain = self.get_grid_data(state_ptr)

        # 计算迷雾 (人类玩家的视角)
        # 规则: 己方领地 + 与己方领地相邻的所有格子均可见
        fog_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        if self.show_fog:
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if owner[y, x] == self.human_player:
                        fog_mask[y, x] = False  # 自己可见
                        # 上下左右相邻格子也可见
                        for d in range(4):
                            ny, nx = y + DR[d], x + DC[d]
                            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                                fog_mask[ny, nx] = False

        # 绘制所有格子
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.draw_tile(x, y, owner, army, terrain, fog_mask)

        # 绘制 AI 上一步的移动箭头
        if self.mode == "PLAY" and len(self.ai_think_time) > 0 and not self.game_over:
            # 从最近一步历史推断 AI 的动作
            if len(self.history) >= 2:
                prev_owner, prev_army, _ = self.get_grid_data(self.history[-2])
                curr_owner, curr_army, _ = self.get_grid_data(self.history[-1])
                # 找到兵力变化最大的格子作为 AI 移动目标
                diff = curr_army.astype(int) - prev_army.astype(int)
                # AI 是蓝色方 (player=1)
                ai_increases = np.where((diff > 0) & (curr_owner == 1))
                if len(ai_increases[0]) > 0:
                    # 取最大增加量
                    best = np.argmax(diff[ai_increases])
                    ty, tx = ai_increases[0][best], ai_increases[1][best]
                    # 找来源 (相邻的蓝方格子且兵力减少)
                    for d in range(4):
                        sy, sx = ty + DR[d], tx + DC[d]
                        if 0 <= sy < GRID_SIZE and 0 <= sx < GRID_SIZE:
                            if prev_owner[sy, sx] == 1:
                                if prev_army[sy, sx] > curr_army[sy, sx]:
                                    self.draw_arrow(sx, sy, tx, ty, d, alpha=180)
                                    break

        self.draw_ui()
        pygame.display.flip()

    def run(self):
        """主循环"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.mode == "PLAY":
                        # 检查是否点击了 Skip 按钮
                        skip_rect = self.get_skip_button_rect()
                        if skip_rect and skip_rect.collidepoint(event.pos):
                            self.do_skip()
                        else:
                            self.handle_human_action(event.pos, event.button)

                elif event.type == pygame.MOUSEMOTION:
                    self.hover_tile = self.get_tile_at(event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

                    elif event.key == pygame.K_f:
                        self.show_fog = not self.show_fog

                    elif event.key == pygame.K_r:
                        # 重新开始
                        seed = np.random.randint(0, 2**31)
                        _lib.generals_reset(self.state, seed)
                        self.ai_belief = _lib.belief_create(GRID_SIZE, GRID_SIZE, self.ai_player)
                        self.history = []
                        self.ai_think_time = []
                        self.save_history()
                        self.game_over = False
                        self.winner = -1
                        self.selected_tile = None
                        self.mode = "PLAY"

                    elif event.key == pygame.K_SPACE:
                        self.do_skip()

                    elif event.key == pygame.K_RETURN:
                        # 回车 = 选中/取消当前光标位置
                        if self.cursor_tile:
                            if self.selected_tile is None:
                                owner, army, terrain = self.get_grid_data(self.state)
                                x, y = self.cursor_tile
                                if owner[y, x] == self.human_player and army[y, x] >= 1:
                                    self.selected_tile = self.cursor_tile
                            elif self.selected_tile == self.cursor_tile:
                                self.selected_tile = None

                    elif event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.handle_keyboard_move(0)  # 上
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        self.handle_keyboard_move(1)  # 下
                    elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.handle_keyboard_move(2)  # 左
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.handle_keyboard_move(3)  # 右

                    elif event.key == pygame.K_LEFT:
                        if self.mode == "PLAY":
                            self.mode = "REPLAY"
                        self.current_step_idx = max(0, self.current_step_idx - 1)

                    elif event.key == pygame.K_RIGHT:
                        self.current_step_idx = min(len(self.history) - 1, self.current_step_idx + 1)
                        if self.current_step_idx == len(self.history) - 1:
                            self.mode = "PLAY"

                    elif event.key == pygame.K_UP:
                        # 加速: 跳到最前/最后
                        pass

                    elif event.key == pygame.K_DOWN:
                        pass

            self.render()
            self.clock.tick(60)

        # 清理
        for h in self.history:
            _lib.generals_destroy(h)
        _lib.generals_destroy(self.state)
        _lib.mcts_destroy(self.mcts_engine)
        _lib.belief_destroy(self.ai_belief)
        if self.nn_ptr:
            _lib_nn.nn_destroy(self.nn_ptr)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Generals.io RL Debugger — Human vs AI")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to TorchScript model (.ptl). If not given, uses pure MCTS.")
    parser.add_argument("--n-mcts", type=int, default=260,
                        help="MCTS iterations per move (default: 260)")
    parser.add_argument("--human-player", type=int, default=0, choices=[0, 1],
                        help="Human player ID: 0=red, 1=blue (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # 模型路径 — 转为绝对路径 (C++ 层需要绝对路径)
    model_path = args.model
    if model_path is not None:
        if not os.path.isabs(model_path):
            model_path = os.path.normpath(os.path.join(SCRIPT_DIR, model_path))
        model_path = os.path.abspath(model_path)
    if model_path is None or not os.path.exists(model_path):
        for candidate in ["v4_attention.ptl", "v4_attention_80pct.ptl", "resnet_v4.ptl"]:
            default_path = os.path.join(SCRIPT_DIR, "rl_models", candidate)
            if os.path.exists(default_path):
                model_path = os.path.abspath(default_path)
                print(f"[GUI] 使用默认模型: {model_path}")
                break

    app = GeneralsGUI(
        model_path=model_path,
        n_mcts=args.n_mcts,
        human_player=args.human_player,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
