/**
 * flow_field.h — 矢量流场寻路
 * 全图势能场，所有部队像水流一样同时向目标推进
 */
#pragma once
#include <cstring>
#include <vector>
#include <algorithm>
#include "common.h"
#include "belief_state.h"

static constexpr uint16_t COST_INF = 65535;

struct MoveCmd {
    int src_idx;
    int dest_idx;
    bool is_half;
};

struct VectorFlowField {
    uint16_t cost_field[MAX_TILES];
    uint16_t integration_field[MAX_TILES];
    int16_t  vector_field[MAX_TILES]; // -1 = 无流向

    void init() { std::memset(vector_field, -1, sizeof(vector_field)); }

    void build_cost_field(const BeliefState& state) {
        int w = state.width, h = state.height;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int i = y * w + x;
                if (state.terrain_mem[i] == MEM_MOUNTAIN) cost_field[i] = COST_INF;
                else if (state.owner[i] == 1 - state.player_id)
                    cost_field[i] = 1 + (state.guessed_army[i] / 10);
                else if (state.terrain_mem[i] == MEM_CITY && state.owner[i] != state.player_id)
                    cost_field[i] = 80;
                else if (!state.visible[i] && state.terrain_mem[i] == MEM_UNKNOWN)
                    cost_field[i] = 5;
                else cost_field[i] = 1;
            }
    }

    void build_integration_field(int target_idx, int w, int h) {
        for (int i = 0; i < w * h; ++i) integration_field[i] = COST_INF;
        integration_field[target_idx] = 0;

        struct Item { uint16_t cost; int idx; };
        auto cmp = [](const Item& a, const Item& b) { return a.cost > b.cost; };
        std::vector<Item> pq;
        pq.push_back({0, target_idx}); std::push_heap(pq.begin(), pq.end(), cmp);

        while (!pq.empty()) {
            std::pop_heap(pq.begin(), pq.end(), cmp);
            auto [cc, ci] = pq.back(); pq.pop_back();
            if (cc != integration_field[ci]) continue;
            int cx = ci % w, cy = ci / w;
            for (int d = 0; d < 4; ++d) {
                int nx = cx + DC[d], ny = cy + DR[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int ni = ny * w + nx;
                if (cost_field[ci] == COST_INF) continue;
                uint16_t nc = cc + cost_field[ci];
                if (nc < integration_field[ni]) {
                    integration_field[ni] = nc;
                    pq.push_back({nc, ni}); std::push_heap(pq.begin(), pq.end(), cmp);
                }
            }
        }
    }

    void build_vector_field(int w, int h) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int i = y * w + x;
                if (integration_field[i] == COST_INF) { vector_field[i] = -1; continue; }
                int best = -1; uint16_t best_cost = integration_field[i];
                for (int d = 0; d < 4; ++d) {
                    int nx = x + DC[d], ny = y + DR[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    int ni = ny * w + nx;
                    if (integration_field[ni] < best_cost) { best_cost = integration_field[ni]; best = ni; }
                }
                vector_field[i] = best;
            }
    }

    int get_sync_moves(const BeliefState& state, MoveCmd* cmds, int max_cmds) {
        int w = state.width, h = state.height, n = 0;
        for (int y = 0; y < h && n < max_cmds; ++y)
            for (int x = 0; x < w && n < max_cmds; ++x) {
                int i = y * w + x;
                if (state.owner[i] == state.player_id && state.army[i] > 1 && vector_field[i] >= 0) {
                    int dst = vector_field[i];
                    bool suicide = (state.owner[dst] != state.player_id && state.owner[dst] != -1
                                    && state.army[i] - 1 <= state.guessed_army[dst]);
                    if (!suicide) cmds[n++] = {i, dst, false};
                }
            }
        return n;
    }

    int compute(const BeliefState& state, int target_idx, MoveCmd* cmds, int max_cmds) {
        build_cost_field(state);
        build_integration_field(target_idx, state.width, state.height);
        build_vector_field(state.width, state.height);
        return get_sync_moves(state, cmds, max_cmds);
    }
};
