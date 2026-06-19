/**
 * game_state.h — 完美信息游戏状态（服务器真实地图）
 */
#pragma once
#include <cstring>
#include <random>
#include "common.h"

struct GameState {
    // === 配置 ===
    int width;
    int height;
    int max_steps;

    // === 运行时 ===
    int current_step;
    int winner;       // -1=无, 0=红, 1=蓝
    bool stalemate;

    // === 地图 (1D 扁平数组) ===
    int8_t  owner[MAX_TILES];   // -1=中立, 0=红, 1=蓝
    int16_t army[MAX_TILES];
    uint8_t terrain[MAX_TILES]; // Terrain 枚举

    // === 玩家 ===
    bool is_alive[2];

    // === 随机数 ===
    std::mt19937 rng;

    inline int idx(int r, int c) const { return r * width + c; }

    void init(int w, int h, int max_steps_val, unsigned int seed) {
        width = w; height = h; max_steps = max_steps_val;
        current_step = 0; winner = -1; stalemate = false;
        is_alive[0] = is_alive[1] = true;
        rng.seed(seed);
        std::memset(owner, -1, sizeof(owner));
        std::memset(army, 0, sizeof(army));
        std::memset(terrain, T_EMPTY, sizeof(terrain));
    }

    void generate_map() {
        std::memset(owner, -1, sizeof(owner));
        std::memset(army, 0, sizeof(army));
        std::memset(terrain, T_EMPTY, sizeof(terrain));
        is_alive[0] = is_alive[1] = true;

        // 山脉 (15%)
        int n_mountains = int(width * height * 0.15);
        for (int i = 0; i < n_mountains; ++i) {
            int r = rng() % height, c = rng() % width;
            terrain[idx(r, c)] = T_MOUNTAIN;
        }

        // 中立城市 (4-6 个, 15-25 兵力)
        int n_cities = 4 + (rng() % 3);
        for (int i = 0; i < n_cities; ++i) {
            int r = 1 + (rng() % (height - 2));
            int c = 1 + (rng() % (width - 2));
            int pos = idx(r, c);
            if (terrain[pos] == T_EMPTY) {
                terrain[pos] = T_CITY;
                army[pos] = 15 + (rng() % 11);
            }
        }

        // 红方将军 (左上)
        int p0_r = rng() % 4, p0_c = rng() % 4;
        int p0_pos = idx(p0_r, p0_c);
        terrain[p0_pos] = T_GENERAL; owner[p0_pos] = 0; army[p0_pos] = 1;

        // 蓝方将军 (右下)
        int p1_r = height - 4 + (rng() % 4);
        int p1_c = width  - 4 + (rng() % 4);
        while (p1_r == p0_r && p1_c == p0_c) {
            p1_r = height - 4 + (rng() % 4);
            p1_c = width  - 4 + (rng() % 4);
        }
        int p1_pos = idx(p1_r, p1_c);
        terrain[p1_pos] = T_GENERAL; owner[p1_pos] = 1; army[p1_pos] = 1;
    }

    void get_action_mask(bool* mask, int player_id) const {
        int total_actions = width * height * 8 + 1;
        std::memset(mask, 0, total_actions * sizeof(bool));
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                int pos = idx(r, c);
                if (owner[pos] == player_id && army[pos] > 1) {
                    int base = (r * width + c) * 8;
                    if (r > 0 && terrain[idx(r-1, c)] != T_MOUNTAIN)
                        mask[base+0] = mask[base+1] = true;
                    if (r < height-1 && terrain[idx(r+1, c)] != T_MOUNTAIN)
                        mask[base+2] = mask[base+3] = true;
                    if (c > 0 && terrain[idx(r, c-1)] != T_MOUNTAIN)
                        mask[base+4] = mask[base+5] = true;
                    if (c < width-1 && terrain[idx(r, c+1)] != T_MOUNTAIN)
                        mask[base+6] = mask[base+7] = true;
                }
            }
        }
        mask[width * height * 8] = true; // SKIP
    }

    bool decode_action(int action_id, int& r, int& c, int& direction, int& is_half) const {
        int skip = width * height * 8;
        if (action_id == skip) return false;
        is_half = action_id % 2;
        direction = (action_id / 2) % 4;
        c = (action_id / 8) % width;
        r = action_id / (width * 8);
        return true;
    }

    /** 编码动作 (src_r, src_c, direction, is_half) → action_id */
    int encode_action(int r, int c, int direction, int is_half) const {
        return (r * width + c) * 8 + direction * 2 + is_half;
    }

    void apply_move(int player, int r, int c, int direction, int is_half) {
        int src = idx(r, c);
        if (owner[src] != player || army[src] <= 1) return;
        int nr = r + DR[direction], nc = c + DC[direction];
        if (nr < 0 || nr >= height || nc < 0 || nc >= width) return;
        int dst = idx(nr, nc);
        if (terrain[dst] == T_MOUNTAIN) return;

        int total = army[src];
        int moving = (is_half == 1) ? (total / 2) : (total - 1);
        if (moving <= 0) return;
        army[src] -= moving;

        if (owner[dst] == player) {
            army[dst] += moving;
        } else if (moving > army[dst]) {
            army[dst] = moving - army[dst];
            int old_owner = owner[dst]; owner[dst] = player;
            if (terrain[dst] == T_GENERAL && old_owner != -1) {
                is_alive[old_owner] = false; winner = player;
                for (int i = 0; i < width * height; ++i)
                    if (owner[i] == old_owner) { owner[i] = player; army[i] = (army[i] + 1) / 2; }
                terrain[dst] = T_CITY;
            }
        } else {
            army[dst] -= moving;
        }
    }

    void opponent_turn() {
        if (winner != -1) return;
        int total_actions = width * height * 8 + 1, skip_action = width * height * 8;
        int* valid_non_skip = new int[total_actions - 1];
        int n_valid = 0;
        for (int a = 0; a < total_actions; ++a) {
            if (a == skip_action) continue;
            int r, c, dir, half;
            if (!decode_action(a, r, c, dir, half)) continue;
            int src = idx(r, c);
            if (owner[src] != 1 || army[src] <= 1) continue;
            int nr = r + DR[dir], nc = c + DC[dir];
            if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
            if (terrain[idx(nr, nc)] == T_MOUNTAIN) continue;
            valid_non_skip[n_valid++] = a;
        }
        if (n_valid > 0) {
            int chosen = valid_non_skip[rng() % n_valid];
            delete[] valid_non_skip;
            int r, c, dir, half; decode_action(chosen, r, c, dir, half);
            apply_move(1, r, c, dir, half);
        } else delete[] valid_non_skip;
    }

    void tick() {
        current_step++;
        bool bonus = (current_step % 25 == 0);
        for (int i = 0; i < width * height; ++i) {
            if (owner[i] != -1) {
                if (terrain[i] == T_GENERAL || terrain[i] == T_CITY) army[i]++;
                if (bonus) army[i]++;
            }
        }
    }

    int tiebreaker_score(int player_id) const {
        int t = 0, tiles = 0;
        for (int i = 0; i < width * height; ++i)
            if (owner[i] == player_id) { t += army[i]; tiles++; }
        return t + tiles * 10;
    }

    int step(int action_id) {
        bool red_first = (rng() % 2 == 0);
        if (red_first) {
            int r, c, dir, half;
            if (decode_action(action_id, r, c, dir, half)) apply_move(0, r, c, dir, half);
            opponent_turn();
        } else {
            opponent_turn();
            int r, c, dir, half;
            if (decode_action(action_id, r, c, dir, half)) apply_move(0, r, c, dir, half);
        }
        tick();
        if (current_step >= max_steps && winner == -1) {
            stalemate = true;
            int s0 = tiebreaker_score(0), s1 = tiebreaker_score(1);
            if (s0 > s1) winner = 0; else if (s1 > s0) winner = 1;
        }
        return winner;
    }

    void get_grid_data(int8_t* owner_out, int16_t* army_out, uint8_t* terrain_out) const {
        int total = width * height;
        std::memcpy(owner_out, owner, total * sizeof(int8_t));
        std::memcpy(army_out, army, total * sizeof(int16_t));
        std::memcpy(terrain_out, terrain, total * sizeof(uint8_t));
    }

    void get_obs(float* buffer, int player_id) const {
        int enemy_id = 1 - player_id, total = width * height;
        std::memset(buffer, 0, 7 * total * sizeof(float));
        for (int i = 0; i < total; ++i) {
            if (owner[i] == player_id) buffer[i] = army[i];
            if (owner[i] == enemy_id) buffer[total + i] = army[i];
            if (owner[i] == -1) buffer[2*total + i] = army[i];
            if (terrain[i] == T_MOUNTAIN) buffer[3*total + i] = 1.0f;
            if (terrain[i] == T_CITY) buffer[4*total + i] = 1.0f;
            if (terrain[i] == T_GENERAL && owner[i] == player_id) buffer[5*total + i] = 1.0f;
            if (terrain[i] == T_GENERAL && owner[i] == enemy_id) buffer[6*total + i] = 1.0f;
        }
    }
};
