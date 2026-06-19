/**
 * script_agent.h — 多性格规则脚本智能体
 *
 * 三种性格：
 *   A (扩张流) — 快速扩张占中立格，用半兵留守
 *   B (城市流) — 优先抢城市，憋兵经济流
 *   C (进攻流) — 全力攻击敌方，直捅将军
 *
 * 所有逻辑在 GameState（完美信息）上运行，用于评估 IS-MCTS。
 */
#pragma once
#include <algorithm>
#include <cstring>
#include "common.h"
#include "game_state.h"

// ============================================================
// 性格枚举
// ============================================================
enum ScriptPersonality {
    SCRIPT_A_EXPANSION  = 0,  // 扩张流
    SCRIPT_B_CITY       = 1,  // 城市流
    SCRIPT_C_AGGRESSIVE = 2,  // 进攻流
};

// ============================================================
// 临时 BFS 工具（栈分配，不跨调用）
// ============================================================
struct ScriptBFS {
    int dist[MAX_TILES];
    int q[MAX_TILES];

    void compute(const GameState& gs, int start) {
        std::memset(dist, -1, sizeof(int) * gs.width * gs.height);
        int qh = 0, qt = 0;
        q[qt++] = start; dist[start] = 0;
        while (qh < qt) {
            int cur = q[qh++];
            int cx = cur % gs.width, cy = cur / gs.width;
            for (int d = 0; d < 4; ++d) {
                int nx = cx + DC[d], ny = cy + DR[d];
                if (nx < 0 || nx >= gs.width || ny < 0 || ny >= gs.height) continue;
                int ni = ny * gs.width + nx;
                if (dist[ni] >= 0 || gs.terrain[ni] == T_MOUNTAIN) continue;
                dist[ni] = dist[cur] + 1;
                q[qt++] = ni;
            }
        }
    }
};

// ============================================================
// 脚本智能体核心
// ============================================================
struct ScriptAgent {
    /**
     * 对从 src 向 dest 移动兵力的动作打分。
     * 分数越高 = 这步棋越好。
     */
    float score_move(const GameState& gs, int player, int src, int dest,
                     ScriptPersonality personality,
                     const ScriptBFS& bfs_enemy, const ScriptBFS& bfs_my_gen,
                     bool is_enemy_tile) const {
        float score = 0.0f;
        int enemy = 1 - player;

        // ---- 基础属性 ----
        int army_at_src = gs.army[src];
        int army_at_dest = gs.army[dest];
        int owner_dest = gs.owner[dest];
        int terrain_dest = gs.terrain[dest];
        int dist_to_enemy = (bfs_enemy.dist[dest] >= 0) ? bfs_enemy.dist[dest] : 999;
        int dist_to_gen = (bfs_my_gen.dist[dest] >= 0) ? bfs_my_gen.dist[dest] : 999;

        switch (personality) {

        // ======== A — 扩张流 ========
        case SCRIPT_A_EXPANSION: {
            // 占领中立格/城市：高优先级
            if (owner_dest == -1 && terrain_dest == T_CITY)
                score += 8.0f;
            else if (owner_dest == -1)
                score += 5.0f;
            // 攻击敌方：中等
            if (is_enemy_tile) {
                score += 4.0f;
                if (terrain_dest == T_GENERAL) score += 15.0f;
            }
            // 近敌分
            if (dist_to_enemy <= 5) score += 3.0f;
            // 留守：越靠近将军，留防守分越高
            if (dist_to_gen <= 3) score += 2.0f;
            // 兵力够才打敌方
            if (is_enemy_tile && army_at_src <= army_at_dest && terrain_dest != T_GENERAL)
                score -= 4.0f;
            break;
        }

        // ======== B — 城市流 ========
        case SCRIPT_B_CITY: {
            // 城市是第一优先级
            if (terrain_dest == T_CITY) {
                if (owner_dest == -1) score += 12.0f;
                else if (owner_dest == enemy) score += 10.0f;
            }
            // 扩张：中等
            if (owner_dest == -1) score += 4.0f;
            // 攻击：仅在兵力碾压时
            if (is_enemy_tile) {
                if (army_at_src > army_at_dest * 1.5f)
                    score += 5.0f;
                else
                    score -= 3.0f;
                if (terrain_dest == T_GENERAL && army_at_src > army_at_dest)
                    score += 20.0f;
            }
            // 防守分：重防守
            if (dist_to_gen <= 2) score += 4.0f;
            if (dist_to_gen <= 4) score += 2.0f;
            // 不送兵
            if (army_at_src <= 3 && owner_dest == player) score -= 1.0f;
            break;
        }

        // ======== C — 进攻流 ========
        case SCRIPT_C_AGGRESSIVE: {
            // 攻击是一切
            if (is_enemy_tile) {
                score += 8.0f;
                if (terrain_dest == T_GENERAL) score += 25.0f;
                // 兵力多加分
                score += army_at_src * 0.1f;
            }
            // 近敌分：越近越好
            if (dist_to_enemy <= 3) score += 6.0f;
            else if (dist_to_enemy <= 6) score += 3.0f;
            // 占领中立（次要）
            if (owner_dest == -1) score += 2.0f;
            // 不防守
            if (dist_to_gen <= 2) score -= 2.0f;
            // 不管兵力比，莽就对了
            break;
        }
        }

        // ---- 共用修正 ----
        // 不要往自己地盘深送兵（除非是城市或前线）
        if (owner_dest == player && terrain_dest != T_CITY) {
            int my_army = army_at_src;
            int cur_army = army_at_dest;
            if (my_army > cur_army + 5) score -= 1.0f; // 集结过多了
        }
        // 不要走回头路（src 与 dest 相邻但 dest 兵力更多且是自己领土）
        if (owner_dest == player && army_at_dest >= army_at_src * 2) score -= 2.0f;

        return score;
    }

    /** 获取最佳动作的 action_id */
    int get_action(const GameState& gs, int player, ScriptPersonality personality) {
        int total = gs.width * gs.height;
        int skip = total * 8;
        int enemy = 1 - player;

        // 找己方将军位置用于 BFS
        int my_general = -1;
        for (int i = 0; i < total; ++i)
            if (gs.terrain[i] == T_GENERAL && gs.owner[i] == player)
                { my_general = i; break; }

        // 计算距离（找最近敌方格和到己方将军距离）
        ScriptBFS bfs_enemy;
        int enemy_seed = -1;
        for (int i = 0; i < total; ++i)
            if (gs.owner[i] == enemy && gs.army[i] > 0)
                { enemy_seed = i; break; }
        if (enemy_seed >= 0) bfs_enemy.compute(gs, enemy_seed);
        else std::memset(bfs_enemy.dist, 99, sizeof(int) * total);

        ScriptBFS bfs_my_gen;
        if (my_general >= 0) bfs_my_gen.compute(gs, my_general);
        else std::memset(bfs_my_gen.dist, 99, sizeof(int) * total);

        // 枚举所有合法动作，打分
        float best_score = -99999.0f;
        int best_action = skip;

        for (int i = 0; i < total; ++i) {
            if (gs.owner[i] != player || gs.army[i] <= 1) continue;

            int r = i / gs.width, c = i % gs.width;
            for (int d = 0; d < 4; ++d) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr < 0 || nr >= gs.height || nc < 0 || nc >= gs.width) continue;
                int dest = nr * gs.width + nc;
                if (gs.terrain[dest] == T_MOUNTAIN) continue;

                bool is_enemy = (gs.owner[dest] == enemy);

                // 全兵 (is_half=0)
                float score = score_move(gs, player, i, dest, personality,
                                         bfs_enemy, bfs_my_gen, is_enemy);

                int action_half0 = (r * gs.width + c) * 8 + d * 2 + 0;
                if (score > best_score) { best_score = score; best_action = action_half0; }

                // 半兵 (is_half=1) — 只有兵多或防守时才考虑
                if (gs.army[i] >= 5) {
                    float score_half = score - 1.0f; // 半兵减分
                    // 对 A 扩张流：半兵优先（想留守备）
                    if (personality == SCRIPT_A_EXPANSION && !is_enemy) score_half = score + 1.0f;
                    // 对 B 城市流：半兵可以
                    if (personality == SCRIPT_B_CITY) score_half = score + 0.5f;
                    // 对 C 进攻流：全兵更好
                    if (personality == SCRIPT_C_AGGRESSIVE) score_half = score - 2.0f;

                    int action_half1 = (r * gs.width + c) * 8 + d * 2 + 1;
                    if (score_half > best_score) { best_score = score_half; best_action = action_half1; }
                }
            }
        }

        return best_action; // 无合法动作→skip
    }
};
