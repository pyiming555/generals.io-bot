/**
 * belief_state.h — 贝叶斯滤波 + 信念状态（迷雾记忆 + 去迷雾化）
 */
#pragma once
#include <cstring>
#include <cmath>
#include "common.h"
#include "game_state.h"

// ============================================================
// 贝叶斯滤波：推断敌方将军位置
// ============================================================
struct BayesTracker {
    float prob[MAX_TILES];
    int bfs_dist[MAX_TILES];

    void init() {
        float p = 1.0f / MAX_TILES;
        for (int i = 0; i < MAX_TILES; ++i) prob[i] = p;
    }

    void normalize(int total) {
        float sum = 0.0f;
        for (int i = 0; i < total; ++i) sum += prob[i];
        if (sum > 0.0f) for (int i = 0; i < total; ++i) prob[i] /= sum;
    }

    void update_negative_vision(const bool* visible, const int8_t* terrain_mem, int w, int h) {
        for (int i = 0; i < w * h; ++i) {
            if (visible[i] || terrain_mem[i] == MEM_MOUNTAIN) {
                bool is_gen = (visible[i] && terrain_mem[i] == MEM_GENERAL);
                if (!is_gen) prob[i] = 0.0f;
            }
        }
        normalize(w * h);
    }

    void compute_bfs_dist(int target, const int8_t* terrain_mem, int w, int h) {
        std::memset(bfs_dist, -1, sizeof(int) * MAX_TILES);
        int q[MAX_TILES], qh = 0, qt = 0;
        q[qt++] = target; bfs_dist[target] = 0;
        while (qh < qt) {
            int cur = q[qh++], cx = cur % w, cy = cur / w;
            for (int d = 0; d < 4; ++d) {
                int nx = cx + DC[d], ny = cy + DR[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int ni = ny * w + nx;
                if (bfs_dist[ni] >= 0 || terrain_mem[ni] == MEM_MOUNTAIN) continue;
                bfs_dist[ni] = bfs_dist[cur] + 1; q[qt++] = ni;
            }
        }
    }

    void update_army_observation(int obs_idx, int army_size, int turn,
                                 const int8_t* terrain_mem, int w, int h) {
        compute_bfs_dist(obs_idx, terrain_mem, w, h);
        for (int i = 0; i < w * h; ++i) {
            if (prob[i] <= 0.0f || terrain_mem[i] == MEM_MOUNTAIN) { prob[i] = 0.0f; continue; }
            int dist = bfs_dist[i];
            if (dist > turn || dist < 0) { prob[i] = 0.0f; continue; }
            float sigma = 2.0f + army_size / 20.0f;
            float likelihood = std::exp(-float(dist * dist) / (2.0f * sigma * sigma));
            int x = i % w, y = i / w;
            float corner = 1.0f;
            if (x == 0 || x == w - 1) corner += 0.5f;
            if (y == 0 || y == h - 1) corner += 0.5f;
            prob[i] *= likelihood * corner;
        }
        normalize(w * h);
    }

    int get_best_pos(int w, int h) const {
        int best = -1; float maxp = -1.0f;
        for (int i = 0; i < w * h; ++i)
            if (prob[i] > maxp) { maxp = prob[i]; best = i; }
        return best;
    }

    float get_prob(int i) const { return prob[i]; }
};

// ============================================================
// 信念状态：AI 眼中带迷雾的世界
// ============================================================
struct BeliefState {
    int width, height, player_id;
    int8_t  owner[MAX_TILES];
    int16_t army[MAX_TILES];
    int8_t  terrain_mem[MAX_TILES]; // MemTerrain
    bool    visible[MAX_TILES];
    int16_t guessed_army[MAX_TILES];
    BayesTracker bayes;

    void init(int w, int h, int pid) {
        width = w; height = h; player_id = pid;
        std::memset(owner, -1, sizeof(owner));
        std::memset(army, 0, sizeof(army));
        std::memset(terrain_mem, MEM_UNKNOWN, sizeof(terrain_mem));
        std::memset(visible, 0, sizeof(visible));
        std::memset(guessed_army, 0, sizeof(guessed_army));
        bayes.init();
    }

    inline int idx(int r, int c) const { return r * width + c; }

    void compute_visibility(const GameState& truth) {
        std::memset(visible, 0, sizeof(bool) * width * height);
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                if (truth.owner[truth.idx(r, c)] == player_id) {
                    visible[idx(r, c)] = true;
                    for (int dr = -1; dr <= 1; ++dr)
                        for (int dc = -1; dc <= 1; ++dc) {
                            int nr = r + dr, nc = c + dc;
                            if (nr >= 0 && nr < height && nc >= 0 && nc < width)
                                visible[idx(nr, nc)] = true;
                        }
                }
    }

    void observe(const GameState& truth, int current_turn) {
        compute_visibility(truth);
        for (int i = 0; i < width * height; ++i) {
            if (visible[i]) {
                owner[i] = truth.owner[i]; army[i] = truth.army[i]; guessed_army[i] = army[i];
                if (truth.terrain[i] == T_MOUNTAIN) terrain_mem[i] = MEM_MOUNTAIN;
                else if (truth.terrain[i] == T_CITY) terrain_mem[i] = MEM_CITY;
                else if (truth.terrain[i] == T_GENERAL) terrain_mem[i] = MEM_GENERAL;
                else terrain_mem[i] = MEM_EMPTY;
            } else { army[i] = 0; }
        }
        bayes.update_negative_vision(visible, terrain_mem, width, height);
        for (int i = 0; i < width * height; ++i)
            if (visible[i] && owner[i] == 1 - player_id && army[i] >= 5)
                bayes.update_army_observation(i, army[i], current_turn, terrain_mem, width, height);
    }

    /** 去迷雾化：生成一个"猜测"的完整 GameState */
    void determinize(GameState& guess, std::mt19937& rng) const {
        guess.width = width; guess.height = height;
        std::memcpy(guess.owner, owner, sizeof(int8_t) * width * height);
        std::memcpy(guess.army, army, sizeof(int16_t) * width * height);
        guess.is_alive[0] = guess.is_alive[1] = true;
        guess.winner = -1; guess.stalemate = false; guess.current_step = 0;

        for (int i = 0; i < width * height; ++i) {
            if (terrain_mem[i] == MEM_MOUNTAIN) guess.terrain[i] = T_MOUNTAIN;
            else if (terrain_mem[i] == MEM_CITY) guess.terrain[i] = T_CITY;
            else if (terrain_mem[i] == MEM_GENERAL) guess.terrain[i] = T_GENERAL;
            else guess.terrain[i] = T_EMPTY;
        }

        // 根据贝叶斯概率放置敌方将军
        int enemy_id = 1 - player_id;
        float rnd = float(rng() % 10000) / 10000.0f, cum = 0.0f;
        int placed = -1;
        for (int i = 0; i < width * height; ++i) {
            if (!visible[i] && terrain_mem[i] != MEM_MOUNTAIN) {
                cum += bayes.get_prob(i);
                if (rnd < cum && placed < 0) {
                    guess.terrain[i] = T_GENERAL; guess.owner[i] = enemy_id; placed = i;
                }
            }
        }
        if (placed < 0) {
            int best = bayes.get_best_pos(width, height);
            if (best >= 0 && !visible[best])
                guess.terrain[best] = T_GENERAL; guess.owner[best] = enemy_id;
        }

        // 填充迷雾兵力猜测
        for (int i = 0; i < width * height; ++i) {
            if (!visible[i] && owner[i] == enemy_id && terrain_mem[i] != MEM_GENERAL)
                guess.army[i] = std::max(guess.army[i], int16_t(1));
            if (!visible[i] && terrain_mem[i] == MEM_UNKNOWN && guess.owner[i] == -1)
                if ((rng() % 100) < 15) guess.terrain[i] = T_MOUNTAIN;
        }
    }

    void get_fog_obs(float* buffer) const {
        int total = width * height, enemy_id = 1 - player_id;
        std::memset(buffer, 0, 7 * total * sizeof(float));
        for (int i = 0; i < total; ++i) {
            if (!visible[i]) continue;
            if (owner[i] == player_id) buffer[i] = army[i];
            if (owner[i] == enemy_id) buffer[total + i] = army[i];
            if (owner[i] == -1) buffer[2*total + i] = army[i];
            if (terrain_mem[i] == MEM_MOUNTAIN) buffer[3*total + i] = 1.0f;
            if (terrain_mem[i] == MEM_CITY) buffer[4*total + i] = 1.0f;
            if (terrain_mem[i] == MEM_GENERAL && owner[i] == player_id) buffer[5*total + i] = 1.0f;
            if (terrain_mem[i] == MEM_GENERAL && owner[i] == enemy_id) buffer[6*total + i] = 1.0f;
        }
    }

    void get_bayes_heatmap(float* buffer) const {
        int total = width * height;
        for (int i = 0; i < total; ++i)
            buffer[i] = visible[i] ? 0.0f : bayes.get_prob(i) * 100.0f;
    }
};
