/**
 * is_mcts.h — 信息集蒙特卡洛树搜索 (IS-MCTS)
 *
 * 核心思想：
 *   每轮搜索从 BeliefState 去迷雾化生成 N 个"平行宇宙"，
 *   在每个宇宙中跑 UCT 树搜索，取所有宇宙中统计最优的动作。
 *
 * 预分配内存池，零动态分配。
 */
#pragma once
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include "common.h"
#include "game_state.h"
#include "belief_state.h"
#include "nn_predictor.h"

// ============================================================
// 常量
// ============================================================
constexpr int MAX_MCTS_NODES = 100000;   // 内存池大小
constexpr int MAX_ROLLOUT_DEPTH = 25;    // 推演深度
constexpr int MAX_ACTIONS_PER_NODE = 20; // 每个节点的最大分支数（扩宽策略覆盖）
constexpr float UCT_C = 1.414f;          // 探索常数
constexpr float PUCT_C = 2.0f;           // PUCT 探索常数（带先验）

// ============================================================
// MCTS 动作
// ============================================================
struct MCTSAction {
    bool is_null;    // true = wait/skip
    int src_idx;     // 出兵格
    int dest_idx;    // 目标格
};

// ============================================================
// 树节点 (32 字节)
// ============================================================
struct MCTSNode {
    int parent_id;
    int first_child_id;
    int next_sibling_id;
    MCTSAction action;
    int visits;
    float total_score;
    float prior;             // PUCT 先验概率（NN 输出）
    int untried_start;       // 待探索动作在 action_buf 中的索引
    int untried_count;

    void init(int parent, const MCTSAction& act, float prior_val = 0.0f) {
        parent_id = parent; first_child_id = -1; next_sibling_id = -1;
        action = act; visits = 0; total_score = 0.0f;
        prior = prior_val;
        untried_start = 0; untried_count = 0;
    }
};

// ============================================================
// IS-MCTS 引擎
// ============================================================
struct ISMCTSEngine {
    MCTSNode node_pool[MAX_MCTS_NODES];
    int node_count;

    // 动作缓冲池
    MCTSAction action_buf[MAX_MCTS_NODES * MAX_ACTIONS_PER_NODE];
    int action_buf_count;

    std::mt19937 rng;

    // ============================================================
    // 内部方法
    // ============================================================

    int alloc_node(int parent, const MCTSAction& act, float prior_val = 0.0f) {
        if (node_count >= MAX_MCTS_NODES) return -1;
        int id = node_count++;
        node_pool[id].init(parent, act, prior_val);
        return id;
    }

    /** 生成当前 GameState 下所有有意义的动作 */
    void generate_actions(int node_id, const GameState& gs, int player) {
        int start = action_buf_count;
        int count = 0;

        // 收集所有可移动格子的兵力
        struct Opt { int src, dest, army; };
        Opt opts[MAX_TILES * 4];
        int n_opts = 0;

        for (int i = 0; i < gs.width * gs.height; ++i) {
            if (gs.owner[i] == player && gs.army[i] > 1) {
                int cx = i % gs.width, cy = i / gs.width;
                for (int d = 0; d < 4; ++d) {
                    int nx = cx + DC[d], ny = cy + DR[d];
                    if (nx < 0 || nx >= gs.width || ny < 0 || ny >= gs.height) continue;
                    int dest = ny * gs.width + nx;
                    if (gs.terrain[dest] != T_MOUNTAIN)
                        opts[n_opts++] = {i, dest, gs.army[i]};
                }
            }
        }

        // 按兵力排序，取前 MAX_ACTIONS_PER_NODE-1 个
        std::sort(opts, opts + n_opts, [](const Opt& a, const Opt& b) { return a.army > b.army; });
        int limit = std::min(MAX_ACTIONS_PER_NODE - 1, n_opts);
        for (int j = 0; j < limit; ++j)
            action_buf[action_buf_count++] = {false, opts[j].src, opts[j].dest};

        // 加入 Wait 动作
        action_buf[action_buf_count++] = {true, -1, -1};
        count = limit + 1;

        node_pool[node_id].untried_start = start;
        node_pool[node_id].untried_count = count;
    }

    /** 快速执行一个 MCTSAction 到 GameState 上 */
    void apply_action(GameState& gs, const MCTSAction& act, int player) {
        if (act.is_null) return; // wait
        int sx = act.src_idx % gs.width, sy = act.src_idx / gs.width;
        int dx = act.dest_idx % gs.width, dy = act.dest_idx / gs.width;
        // 计算方向
        int dir = -1;
        for (int d = 0; d < 4; ++d)
            if (DR[d] == dy - sy && DC[d] == dx - sx) { dir = d; break; }
        if (dir >= 0) gs.apply_move(player, sy, sx, dir, 0); // 全兵
    }

    /** 启发式评估函数 */
    float evaluate(const GameState& gs, int player) {
        if (gs.winner == player) return 10000.0f;
        if (gs.winner == 1 - player) return -10000.0f;
        if (!gs.is_alive[player]) return -10000.0f;
        if (!gs.is_alive[1 - player]) return 10000.0f;

        int my_army = 0, en_army = 0, my_land = 0, en_land = 0;
        for (int i = 0; i < gs.width * gs.height; ++i) {
            if (gs.owner[i] == player) { my_army += gs.army[i]; my_land++; }
            else if (gs.owner[i] == 1 - player) { en_army += gs.army[i]; en_land++; }
        }
        return (my_army - en_army) * 1.0f + (my_land - en_land) * 0.5f;
    }

    /** 快速随机推演 */
    float rollout(GameState gs, int player, int depth) {
        std::uniform_int_distribution<int> dir_dist(0, 3);
        for (int d = 0; d < depth; ++d) {
            if (gs.winner != -1 || !gs.is_alive[0] || !gs.is_alive[1]) break;

            // 我方：随机合法动作
            int skip = gs.width * gs.height * 8;
            bool mask[MAX_TILES * 8 + 1];
            gs.get_action_mask(mask, player);
            int valid[256], nv = 0;
            for (int a = 0; a < skip; ++a) if (mask[a]) valid[nv++] = a;
            int act = (nv > 0) ? valid[rng() % nv] : skip;
            gs.step(act);
        }
        return evaluate(gs, player);
    }

    // ============================================================
    // 流场推演 (Flow-field Rollout)
    // ============================================================

    /** BFS 到目标格的距离 */
    void bfs_to_target(const GameState& gs, int target, int* dist_out) {
        std::memset(dist_out, -1, sizeof(int) * gs.width * gs.height);
        if (target < 0) return;
        int q[MAX_TILES], qh = 0, qt = 0;
        q[qt++] = target; dist_out[target] = 0;
        while (qh < qt) {
            int cur = q[qh++], cx = cur % gs.width, cy = cur / gs.width;
            for (int d = 0; d < 4; ++d) {
                int nx = cx + DC[d], ny = cy + DR[d];
                if (nx < 0 || nx >= gs.width || ny < 0 || ny >= gs.height) continue;
                int ni = ny * gs.width + nx;
                if (dist_out[ni] >= 0 || gs.terrain[ni] == T_MOUNTAIN) continue;
                dist_out[ni] = dist_out[cur] + 1; q[qt++] = ni;
            }
        }
    }

    /** 流场评分：给某个玩家的某个动作打分 */
    float score_flow_action(const GameState& gs, int src, int dest, int player,
                            int target, const int* bfs_dist, int my_general) const {
        float score = 0.0f;
        int enemy = 1 - player;
        int owner_dest = gs.owner[dest];
        int terrain_dest = gs.terrain[dest];
        int army_src = gs.army[src];

        // 占领/攻击加分
        if (owner_dest == enemy) {
            score += 4.0f;
            if (terrain_dest == T_GENERAL) score += 15.0f;
        } else if (owner_dest == -1) {
            if (terrain_dest == T_CITY) score += 3.0f;
            else score += 0.5f; // 中立空地
        }

        // 流场方向：向目标推进
        if (target >= 0 && bfs_dist[src] >= 0 && bfs_dist[dest] >= 0) {
            int d_src = bfs_dist[src], d_dst = bfs_dist[dest];
            if (d_dst < d_src) score += 2.0f;            // 靠近目标
            else if (d_dst == d_src) score += 0.5f;      // 平行移动
            else score -= 1.0f;                           // 远离目标
            if (owner_dest == enemy && d_dst <= 4) score += 3.0f;
        }

        // 兵力权重：大兵优先移动
        score += army_src * 0.01f;

        // ====== 防守分量 ======
        if (my_general >= 0) {
            int gen_dist_src = -1, gen_dist_dst = -1;
            // 计算 src 和 dest 到己方将军的距离
            // 用棋盘距离（不精确但够用）
            int sx = src % gs.width, sy = src / gs.width;
            int gx = my_general % gs.width, gy = my_general / gs.width;
            gen_dist_src = abs(sx - gx) + abs(sy - gy);
            int dx = dest % gs.width, dy = dest / gs.width;
            gen_dist_dst = abs(dx - gx) + abs(dy - gy);

            // 如果从离将军近的地方往远处移动 → 需要保留防守
            if (gen_dist_src <= 2) {
                // 将军旁边，保留至少 3 兵
                if (gen_dist_dst > gen_dist_src && army_src <= 4)
                    score -= 3.0f; // 有防守责任，不移动
                // 如果移动的是将军格本身 → 绝对不移动
                if (src == my_general) score -= 10.0f;
            }
            // 如果离将军很远 → 可以大胆进攻
            if (gen_dist_src >= 6) score += 1.0f;
        }

        // 不要向后倒退（往后方自己领土）
        if (owner_dest == player) {
            if (target >= 0 && bfs_dist[dest] > bfs_dist[src] + 1)
                score -= 1.5f;
        }

        return score;
    }

    /** 用流场获取某玩家的最佳动作（带随机噪声的随机版本） */
    int flow_get_action_stochastic(const GameState& gs, int player, int target,
                                    std::mt19937& rng) {
        int bfs_dist[MAX_TILES];
        bfs_to_target(gs, target, bfs_dist);

        // 找己方将军
        int my_general = -1;
        int total = gs.width * gs.height;
        for (int i = 0; i < total; ++i)
            if (gs.terrain[i] == T_GENERAL && gs.owner[i] == player)
                { my_general = i; break; }

        // 收集所有合法动作及其分数
        struct SA { float score; int action_id; };
        SA acts[MAX_TILES * 4];
        int na = 0;
        int skip = total * 8;

        for (int i = 0; i < total; ++i) {
            if (gs.owner[i] != player || gs.army[i] <= 1) continue;
            int r = i / gs.width, c = i % gs.width;
            for (int d = 0; d < 4; ++d) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr < 0 || nr >= gs.height || nc < 0 || nc >= gs.width) continue;
                int dest = nr * gs.width + nc;
                if (gs.terrain[dest] == T_MOUNTAIN) continue;

                float sc = score_flow_action(gs, i, dest, player, target, bfs_dist, my_general);
                // 关键：加随机噪声！让 rollout 有多样性
                sc += (float(rng() % 10000) / 5000.0f - 1.0f) * 2.0f; // [-2.0, 2.0]
                int act_id = (r * gs.width + c) * 8 + d * 2 + 0;
                acts[na++] = {sc, act_id};
            }
        }

        if (na == 0) return skip;
        // 找最高分
        int best = 0;
        for (int a = 1; a < na; ++a)
            if (acts[a].score > acts[best].score) best = a;
        return acts[best].action_id;
    }

    /** 流场推演：前 N 步流场引导，之后随机（兼顾领域知识和多样性） */
    float flow_rollout(GameState gs, int player, int depth) {
        int enemy = 1 - player;
        int flow_depth = std::min(5, depth); // 前5步流场引导，之后随机
        for (int d = 0; d < depth; ++d) {
            if (gs.winner != -1 || !gs.is_alive[0] || !gs.is_alive[1]) break;

            int total = gs.width * gs.height;
            int skip = total * 8;

            // 己方动作
            int act_me;
            if (d < flow_depth) {
                // 流场引导阶段：找敌方将军位置
                int en_gen = -1;
                for (int i = 0; i < total; ++i) {
                    if (gs.terrain[i] == T_GENERAL && gs.owner[i] == enemy)
                        { en_gen = i; break; }
                }
                act_me = flow_get_action_stochastic(gs, player, en_gen, rng);
            } else {
                // 随机阶段
                bool mask[MAX_TILES * 8 + 1];
                gs.get_action_mask(mask, player);
                int valid[256], nv = 0;
                for (int a = 0; a < skip; ++a) if (mask[a]) valid[nv++] = a;
                act_me = (nv > 0) ? valid[rng() % nv] : skip;
            }

            // 对方：随机动作
            bool mask2[MAX_TILES * 8 + 1];
            gs.get_action_mask(mask2, enemy);
            int valid2[256], nv2 = 0;
            for (int a = 0; a < skip; ++a) if (mask2[a]) valid2[nv2++] = a;
            int act_en = (nv2 > 0) ? valid2[rng() % nv2] : skip;

            // 随机先后手
            bool red_first = (rng() % 2 == 0);
            int r0, c0, dir0, half0;
            int r1, c1, dir1, half1;
            bool valid0 = gs.decode_action(act_me, r0, c0, dir0, half0);
            bool valid1 = gs.decode_action(act_en, r1, c1, dir1, half1);

            if (player == 0) {
                if (red_first) {
                    if (valid0) gs.apply_move(0, r0, c0, dir0, half0);
                    if (valid1) gs.apply_move(1, r1, c1, dir1, half1);
                } else {
                    if (valid1) gs.apply_move(1, r1, c1, dir1, half1);
                    if (valid0) gs.apply_move(0, r0, c0, dir0, half0);
                }
            } else {
                if (red_first) {
                    if (valid1) gs.apply_move(0, r1, c1, dir1, half1);
                    if (valid0) gs.apply_move(1, r0, c0, dir0, half0);
                } else {
                    if (valid0) gs.apply_move(1, r0, c0, dir0, half0);
                    if (valid1) gs.apply_move(0, r1, c1, dir1, half1);
                }
            }

            gs.tick();
            gs.current_step++;

            // 检查将军被占
            bool alive_changed = false;
            for (int p = 0; p < 2; ++p) {
                if (gs.is_alive[p]) {
                    bool found = false;
                    for (int i = 0; i < total; ++i)
                        if (gs.terrain[i] == T_GENERAL && gs.owner[i] == p) { found = true; break; }
                    if (!found) { gs.is_alive[p] = false; alive_changed = true; }
                }
            }
            if (alive_changed) {
                if (!gs.is_alive[0] && gs.is_alive[1]) gs.winner = 1;
                else if (gs.is_alive[0] && !gs.is_alive[1]) gs.winner = 0;
                else if (!gs.is_alive[0] && !gs.is_alive[1]) gs.winner = 2;
            }

            if (gs.current_step >= gs.max_steps && gs.winner == -1) {
                int t0 = gs.tiebreaker_score(0), t1 = gs.tiebreaker_score(1);
                if (t0 > t1) gs.winner = 0;
                else if (t1 > t0) gs.winner = 1;
                else gs.winner = 2;
            }
        }
        return evaluate(gs, player);
    }

    // ============================================================
    // 主搜索接口
    // ============================================================

    /** 在单次去迷雾化的 GameState 上跑 MCTS，返回最佳动作 */
    MCTSAction search_on_state(GameState& gs, int player, int iterations) {
        node_count = 0;
        action_buf_count = 0;

        MCTSAction null_act = {true, -1, -1};
        int root_id = alloc_node(-1, null_act);
        generate_actions(root_id, gs, player);

        for (int iter = 0; iter < iterations; ++iter) {
            // 克隆一份用于本次推演
            GameState sim = gs;
            sim.rng.seed(rng());

            int nid = root_id;

            // ---- 1. SELECT ----
            while (node_pool[nid].untried_count == 0 && node_pool[nid].first_child_id != -1) {
                int best = -1; float best_uct = -1e9f;
                int cid = node_pool[nid].first_child_id;
                while (cid != -1) {
                    auto& ch = node_pool[cid];
                    float uct = ch.total_score / ch.visits
                        + UCT_C * std::sqrt(std::log(float(node_pool[nid].visits)) / ch.visits);
                    if (uct > best_uct) { best_uct = uct; best = cid; }
                    cid = ch.next_sibling_id;
                }
                nid = best;
                apply_action(sim, node_pool[nid].action, player);
                sim.tick();
            }

            // ---- 2. EXPAND ----
            if (node_pool[nid].untried_count > 0 && sim.winner == -1
                && sim.is_alive[0] && sim.is_alive[1]) {
                int idx = node_pool[nid].untried_start + (--node_pool[nid].untried_count);
                MCTSAction act = action_buf[idx];
                apply_action(sim, act, player);
                sim.tick();

                int child = alloc_node(nid, act);
                if (child >= 0) {
                    node_pool[child].next_sibling_id = node_pool[nid].first_child_id;
                    node_pool[nid].first_child_id = child;
                    generate_actions(child, sim, player);
                    nid = child;
                }
            }

            // ---- 3. ROLLOUT ----
            float reward = rollout(sim, player, MAX_ROLLOUT_DEPTH);

            // ---- 4. BACKPROPAGATE ----
            int bid = nid;
            while (bid != -1) {
                node_pool[bid].visits++;
                node_pool[bid].total_score += reward;
                bid = node_pool[bid].parent_id;
            }
        }

        // ---- 提取最佳动作 ----
        int best = -1, max_visits = -1;
        int cid = node_pool[root_id].first_child_id;
        while (cid != -1) {
            if (node_pool[cid].visits > max_visits) { max_visits = node_pool[cid].visits; best = cid; }
            cid = node_pool[cid].next_sibling_id;
        }
        if (best < 0) return null_act;
        return node_pool[best].action;
    }

    // ============================================================
    // IS-MCTS 主入口
    // ============================================================

    /**
     * 从 BeliefState 出发，运行 IS-MCTS 搜索。
     * @param belief  当前信念状态
     * @param player  当前玩家 ID
     * @param n_det   去迷雾化次数（每轮生成多少个平行宇宙）
     * @param n_mcts  每个宇宙中的 MCTS 迭代次数
     * @return 最佳动作的 (src_idx, dest_idx)，is_null=true 表示 Wait
     */
    MCTSAction search(const BeliefState& belief, int player,
                      int n_det = 8, int n_mcts = 500) {
        // 统计每个动作在所有宇宙中的总访问量
        struct Vote { int src, dest; int visits; };
        Vote votes[MAX_TILES * 4];
        int n_votes = 0;

        for (int det = 0; det < n_det; ++det) {
            // 去迷雾化生成一个猜测地图
            GameState guess;
            belief.determinize(guess, rng);
            guess.max_steps = 999; // 搜索模式不限步数

            // 跑 MCTS
            MCTSAction best = search_on_state(guess, player, n_mcts / n_det);
            if (!best.is_null) {
                // 累加投票
                bool found = false;
                for (int v = 0; v < n_votes; ++v)
                    if (votes[v].src == best.src_idx && votes[v].dest == best.dest_idx) {
                        votes[v].visits++; found = true; break;
                    }
                if (!found && n_votes < MAX_TILES * 4)
                    votes[n_votes++] = {best.src_idx, best.dest_idx, 1};
            }
        }

        // 取票数最高的动作
        int best_v = -1, max_v = -1;
        for (int v = 0; v < n_votes; ++v)
            if (votes[v].visits > max_v) { max_v = votes[v].visits; best_v = v; }

        if (best_v < 0) return {true, -1, -1};
        return {false, votes[best_v].src, votes[best_v].dest};
    }
    /** 将 MCTSAction 转为 action_id */
    int to_action_id(const MCTSAction& act, int w, int h) const {
        if (act.is_null) return w * h * 8; // SKIP
        int src_r = act.src_idx / w, src_c = act.src_idx % w;
        int dst_r = act.dest_idx / w, dst_c = act.dest_idx % w;
        int dir = -1;
        for (int d = 0; d < 4; ++d)
            if (DR[d] == dst_r - src_r && DC[d] == dst_c - src_c) { dir = d; break; }
        if (dir < 0) return w * h * 8;
        return (src_r * w + src_c) * 8 + dir * 2 + 0; // 全兵 (half=0)
    }

    // ============================================================
    // 流场 MCTS (Flow-field Rollout 版)
    // ============================================================

    /** 使用流场推演的 search_on_state */
    MCTSAction search_on_state_flow(GameState& gs, int player, int iterations) {
        node_count = 0;
        action_buf_count = 0;

        MCTSAction null_act = {true, -1, -1};
        int root_id = alloc_node(-1, null_act);
        generate_actions(root_id, gs, player);

        for (int iter = 0; iter < iterations; ++iter) {
            GameState sim = gs;
            sim.rng.seed(rng());

            int nid = root_id;

            // SELECT
            while (node_pool[nid].untried_count == 0 && node_pool[nid].first_child_id != -1) {
                int best = -1; float best_uct = -1e9f;
                int cid = node_pool[nid].first_child_id;
                while (cid != -1) {
                    auto& ch = node_pool[cid];
                    float uct = ch.total_score / ch.visits
                        + UCT_C * std::sqrt(std::log(float(node_pool[nid].visits)) / ch.visits);
                    if (uct > best_uct) { best_uct = uct; best = cid; }
                    cid = ch.next_sibling_id;
                }
                nid = best;
                apply_action(sim, node_pool[nid].action, player);
                sim.tick();
            }

            // EXPAND
            if (node_pool[nid].untried_count > 0 && sim.winner == -1
                && sim.is_alive[0] && sim.is_alive[1]) {
                int idx = node_pool[nid].untried_start + (--node_pool[nid].untried_count);
                MCTSAction act = action_buf[idx];
                apply_action(sim, act, player);
                sim.tick();

                int child = alloc_node(nid, act);
                if (child >= 0) {
                    node_pool[child].next_sibling_id = node_pool[nid].first_child_id;
                    node_pool[nid].first_child_id = child;
                    generate_actions(child, sim, player);
                    nid = child;
                }
            }

            // ROLLOUT (流场版)
            float reward = flow_rollout(sim, player, MAX_ROLLOUT_DEPTH);

            // BACKPROPAGATE
            int bid = nid;
            while (bid != -1) {
                node_pool[bid].visits++;
                node_pool[bid].total_score += reward;
                bid = node_pool[bid].parent_id;
            }
        }

        int best = -1, max_visits = -1;
        int cid = node_pool[root_id].first_child_id;
        while (cid != -1) {
            if (node_pool[cid].visits > max_visits) { max_visits = node_pool[cid].visits; best = cid; }
            cid = node_pool[cid].next_sibling_id;
        }
        if (best < 0) return null_act;
        return node_pool[best].action;
    }

    /** IS-MCTS 流场版入口 */
    MCTSAction search_flow(const BeliefState& belief, int player,
                           int n_det = 8, int n_mcts = 500) {
        return search_flow_with_policy(belief, player, n_det, n_mcts, nullptr);
    }

    /** IS-MCTS 流场版入口（含策略输出） */
    MCTSAction search_flow_with_policy(const BeliefState& belief, int player,
                                       int n_det, int n_mcts,
                                       float* policy_out) {
        int n_actions = belief.width * belief.height * 8 + 1;
        if (policy_out) std::memset(policy_out, 0, sizeof(float) * n_actions);

        struct Vote { int src, dest; int visits; };
        Vote votes[MAX_TILES * 4];
        int n_votes = 0;

        for (int det = 0; det < n_det; ++det) {
            GameState guess;
            belief.determinize(guess, rng);
            guess.max_steps = 999;

            MCTSAction best = search_on_state_flow(guess, player, n_mcts / n_det);

            // 提取根节点子节点的访问量 -> 策略分布
            if (policy_out) {
                int cid = node_pool[0].first_child_id;
                while (cid != -1) {
                    auto& ch = node_pool[cid];
                    if (ch.visits > 0) {
                        int aid = to_action_id(ch.action, belief.width, belief.height);
                        if (aid >= 0 && aid < n_actions)
                            policy_out[aid] += ch.visits;
                    }
                    cid = ch.next_sibling_id;
                }
            }

            // 投票（同 search_flow）
            if (!best.is_null) {
                bool found = false;
                for (int v = 0; v < n_votes; ++v)
                    if (votes[v].src == best.src_idx && votes[v].dest == best.dest_idx) {
                        votes[v].visits++; found = true; break;
                    }
                if (!found && n_votes < MAX_TILES * 4)
                    votes[n_votes++] = {best.src_idx, best.dest_idx, 1};
            }
        }

        // 归一化策略
        if (policy_out) {
            float total = 0.0f;
            for (int i = 0; i < n_actions; ++i) total += policy_out[i];
            if (total > 0.0f)
                for (int i = 0; i < n_actions; ++i) policy_out[i] /= total;
        }

        // 找最佳动作（投票制）
        int best_v = -1, max_v = -1;
        for (int v = 0; v < n_votes; ++v)
            if (votes[v].visits > max_v) { max_v = votes[v].visits; best_v = v; }

        if (best_v < 0) return {true, -1, -1};
        return {false, votes[best_v].src, votes[best_v].dest};
    }
    // ============================================================
    // NN-guided 搜索 (PUCT + 网络价值)
    // ============================================================

    /** 生成动作 + 从 NN policy 中读取先验 */
    void generate_actions_nn(int node_id, const GameState& gs, int player,
                             const float* policy_prior) {
        int start = action_buf_count;
        int count = 0;

        struct Opt { int src, dest, army, action_id; float prior; };
        Opt opts[MAX_TILES * 4];
        int n_opts = 0;

        for (int i = 0; i < gs.width * gs.height; ++i) {
            if (gs.owner[i] == player && gs.army[i] > 1) {
                int cx = i % gs.width, cy = i / gs.width;
                for (int d = 0; d < 4; ++d) {
                    int nx = cx + DC[d], ny = cy + DR[d];
                    if (nx < 0 || nx >= gs.width || ny < 0 || ny >= gs.height) continue;
                    int dest = ny * gs.width + nx;
                    if (gs.terrain[dest] != T_MOUNTAIN) {
                        int aid = (cy * gs.width + cx) * 8 + d * 2 + 0;
                        float p = policy_prior ? policy_prior[aid] : 0.0f;
                        opts[n_opts++] = {i, dest, gs.army[i], aid, p};
                    }
                }
            }
        }

        // 按先验概率排序（有 NN 先验的优先）
        if (policy_prior) {
            std::sort(opts, opts + n_opts,
                      [](const Opt& a, const Opt& b) { return a.prior > b.prior; });
        } else {
            std::sort(opts, opts + n_opts,
                      [](const Opt& a, const Opt& b) { return a.army > b.army; });
        }

        int limit = std::min(MAX_ACTIONS_PER_NODE - 1, n_opts);
        for (int j = 0; j < limit; ++j) {
            action_buf[action_buf_count++] = {false, opts[j].src, opts[j].dest};
            // 在 action_buf 后面存 prior（把 prior 编码为 float 数组）
            // 实际上，prior 会在 alloc_node 时传入
        }

        // Wait 动作，先验为小常数
        action_buf[action_buf_count++] = {true, -1, -1};
        count = limit + 1;

        node_pool[node_id].untried_start = start;
        node_pool[node_id].untried_count = count;
    }

    /** NN-guided 单宇宙搜索 */
    MCTSAction search_on_state_nn(GameState& gs, int player, int iterations,
                                   const float* policy_prior, float value_prior) {
        node_count = 0;
        action_buf_count = 0;

        MCTSAction null_act = {true, -1, -1};
        int root_id = alloc_node(-1, null_act);
        generate_actions_nn(root_id, gs, player, policy_prior);
        // 为根节点的每个子动作设置先验
        // 这是关键：root 的 children 用 NN 先验
        float prior_sum = 0.0f;
        {
            int cid = node_pool[root_id].first_child_id;
            while (cid != -1) {
                auto& ch = node_pool[cid];
                // 从 action_buf 中找对应动作的 prior
                // 用 policy_prior 查
                if (!ch.action.is_null) {
                    int aid = to_action_id(ch.action, gs.width, gs.height);
                    if (policy_prior && aid >= 0)
                        ch.prior = policy_prior[aid];
                }
                prior_sum += ch.prior;
                cid = ch.next_sibling_id;
            }
            // 归一化根节点先验
            if (prior_sum > 0.0f) {
                cid = node_pool[root_id].first_child_id;
                while (cid != -1) {
                    node_pool[cid].prior /= prior_sum;
                    cid = node_pool[cid].next_sibling_id;
                }
            }
        }

        for (int iter = 0; iter < iterations; ++iter) {
            GameState sim = gs;
            sim.rng.seed(rng());

            int nid = root_id;

            // SELECT: 用 PUCT
            while (node_pool[nid].untried_count == 0 && node_pool[nid].first_child_id != -1) {
                int best = -1; float best_val = -1e9f;
                int cid = node_pool[nid].first_child_id;
                while (cid != -1) {
                    auto& ch = node_pool[cid];
                    float q = ch.visits > 0 ? ch.total_score / ch.visits : 0.0f;
                    float puct = q + PUCT_C * ch.prior *
                        std::sqrt(float(node_pool[nid].visits)) / (1.0f + ch.visits);
                    if (puct > best_val) { best_val = puct; best = cid; }
                    cid = ch.next_sibling_id;
                }
                nid = best;
                apply_action(sim, node_pool[nid].action, player);
                sim.tick();
            }

            // EXPAND
            if (node_pool[nid].untried_count > 0 && sim.winner == -1
                && sim.is_alive[0] && sim.is_alive[1]) {
                int idx = node_pool[nid].untried_start + (--node_pool[nid].untried_count);
                MCTSAction act = action_buf[idx];
                apply_action(sim, act, player);
                sim.tick();

                float prior_val = 0.0f;
                if (!act.is_null && policy_prior) {
                    int aid = to_action_id(act, gs.width, gs.height);
                    if (aid >= 0) prior_val = policy_prior[aid];
                }
                int child = alloc_node(nid, act, prior_val);
                if (child >= 0) {
                    node_pool[child].next_sibling_id = node_pool[nid].first_child_id;
                    node_pool[nid].first_child_id = child;
                    generate_actions_nn(child, sim, player, policy_prior);
                    nid = child;
                }
            }

            // ROLLOUT: 混合 NN 价值 + 流场 rollout
            float rollout_val = flow_rollout(sim, player, MAX_ROLLOUT_DEPTH);
            float reward = 0.3f * value_prior + 0.7f * rollout_val;

            // BACKPROPAGATE
            int bid = nid;
            while (bid != -1) {
                node_pool[bid].visits++;
                node_pool[bid].total_score += reward;
                bid = node_pool[bid].parent_id;
            }
        }

        int best = -1, max_visits = -1;
        int cid = node_pool[root_id].first_child_id;
        while (cid != -1) {
            if (node_pool[cid].visits > max_visits) { max_visits = node_pool[cid].visits; best = cid; }
            cid = node_pool[cid].next_sibling_id;
        }
        if (best < 0) return null_act;
        return node_pool[best].action;
    }

    /** IS-MCTS NN 版入口 */
    MCTSAction search_nn(const BeliefState& belief, int player,
                         int n_det, int n_mcts,
                         const float* policy_prior, float value_prior) {
        struct Vote { int src, dest; int visits; };
        Vote votes[MAX_TILES * 4];
        int n_votes = 0;

        for (int det = 0; det < n_det; ++det) {
            GameState guess;
            belief.determinize(guess, rng);
            guess.max_steps = 999;

            MCTSAction best = search_on_state_nn(guess, player, n_mcts / n_det,
                                                  policy_prior, value_prior);
            if (!best.is_null) {
                bool found = false;
                for (int v = 0; v < n_votes; ++v)
                    if (votes[v].src == best.src_idx && votes[v].dest == best.dest_idx) {
                        votes[v].visits++; found = true; break;
                    }
                if (!found && n_votes < MAX_TILES * 4)
                    votes[n_votes++] = {best.src_idx, best.dest_idx, 1};
            }
        }

        int best_v = -1, max_v = -1;
        for (int v = 0; v < n_votes; ++v)
            if (votes[v].visits > max_v) { max_v = votes[v].visits; best_v = v; }

        if (best_v < 0) return {true, -1, -1};
        return {false, votes[best_v].src, votes[best_v].dest};
    }

    /** IS-MCTS NN 版入口（含策略输出） */
    MCTSAction search_nn_with_policy(const BeliefState& belief, int player,
                                     int n_det, int n_mcts,
                                     const float* policy_prior, float value_prior,
                                     float* policy_out) {
        int n_actions = belief.width * belief.height * 8 + 1;
        if (policy_out) std::memset(policy_out, 0, sizeof(float) * n_actions);

        struct Vote { int src, dest; int visits; };
        Vote votes[MAX_TILES * 4];
        int n_votes = 0;

        for (int det = 0; det < n_det; ++det) {
            GameState guess;
            belief.determinize(guess, rng);
            guess.max_steps = 999;

            MCTSAction best = search_on_state_nn(guess, player, n_mcts / n_det,
                                                  policy_prior, value_prior);

            // 提取根节点子节点的访问量 -> 策略分布
            if (policy_out) {
                int cid = node_pool[0].first_child_id;
                while (cid != -1) {
                    auto& ch = node_pool[cid];
                    if (ch.visits > 0) {
                        int aid = to_action_id(ch.action, belief.width, belief.height);
                        if (aid >= 0 && aid < n_actions)
                            policy_out[aid] += ch.visits;
                    }
                    cid = ch.next_sibling_id;
                }
            }

            // 投票（同 search_nn）
            if (!best.is_null) {
                bool found = false;
                for (int v = 0; v < n_votes; ++v)
                    if (votes[v].src == best.src_idx && votes[v].dest == best.dest_idx) {
                        votes[v].visits++; found = true; break;
                    }
                if (!found && n_votes < MAX_TILES * 4)
                    votes[n_votes++] = {best.src_idx, best.dest_idx, 1};
            }
        }

        // 归一化策略
        if (policy_out) {
            float total = 0.0f;
            for (int i = 0; i < n_actions; ++i) total += policy_out[i];
            if (total > 0.0f)
                for (int i = 0; i < n_actions; ++i) policy_out[i] /= total;
        }

        int best_v = -1, max_v = -1;
        for (int v = 0; v < n_votes; ++v)
            if (votes[v].visits > max_v) { max_v = votes[v].visits; best_v = v; }

        if (best_v < 0) return {true, -1, -1};
        return {false, votes[best_v].src, votes[best_v].dest};
    }

    /** IS-MCTS NN 版入口（自动用 NNPredictor 推理） */
    MCTSAction search_nn_auto(const BeliefState& belief, int player,
                              int n_det, int n_mcts,
                              NNPredictor& nn) {
        int n_actions = belief.width * belief.height * 8 + 1;
        int h = belief.height, w = belief.width;

        // ============================================================
        // 方案A：NN推理只做一次，从belief提取迷雾观测
        // 与Python版逻辑对齐 — 先验来自信念状态，而非完美信息
        // ============================================================
        float obs_buf[7 * MAX_TILES];
        belief.get_fog_obs(obs_buf);

        float policy_buf[MAX_TILES * 8 + 1];
        float value_buf;
        nn.predict(obs_buf, h, w, n_actions, policy_buf, value_buf);

        // ============================================================
        // determinization循环：所有宇宙共享同一个NN先验
        // ============================================================
        struct Vote { int src, dest; int visits; };
        Vote votes[MAX_TILES * 4];
        int n_votes = 0;

        for (int det = 0; det < n_det; ++det) {
            GameState guess;
            belief.determinize(guess, rng);
            guess.max_steps = 999;

            MCTSAction best = search_on_state_nn(guess, player, n_mcts / n_det,
                                                  policy_buf, value_buf);

            // 投票
            if (!best.is_null) {
                bool found = false;
                for (int v = 0; v < n_votes; ++v)
                    if (votes[v].src == best.src_idx && votes[v].dest == best.dest_idx) {
                        votes[v].visits++; found = true; break;
                    }
                if (!found && n_votes < MAX_TILES * 4)
                    votes[n_votes++] = {best.src_idx, best.dest_idx, 1};
            }
        }

        int best_v = -1, max_v = -1;
        for (int v = 0; v < n_votes; ++v)
            if (votes[v].visits > max_v) { max_v = votes[v].visits; best_v = v; }

        if (best_v < 0) return {true, -1, -1};
        return {false, votes[best_v].src, votes[best_v].dest};
    }
}; // struct ISMCTSEngine
