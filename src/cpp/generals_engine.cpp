/**
 * generals_engine.cpp — C API + 独立性能测试
 *
 * 所有逻辑在以下头文件中定义：
 *   common.h        — 共享类型和常量
 *   game_state.h    — 完美信息游戏状态
 *   belief_state.h  — 迷雾 + 贝叶斯滤波
 *   flow_field.h    — 矢量流场寻路
 *   is_mcts.h       — IS-MCTS 搜索树
 */
#include <iostream>
#include <chrono>
#include "game_state.h"
#include "belief_state.h"
#include "flow_field.h"
#include "is_mcts.h"
#include "script_agent.h"

// ============================================================
// C API
// ============================================================
extern "C" {

    // ---------- GameState ----------
    GameState* generals_create(int width, int height, int max_steps, unsigned int seed) {
        GameState* s = new GameState(); s->init(width, height, max_steps, seed); s->generate_map(); return s;
    }
    void generals_destroy(GameState* s) { delete s; }
    void generals_reset(GameState* s, unsigned int seed) { s->init(s->width, s->height, s->max_steps, seed); s->generate_map(); }
    int generals_step(GameState* s, int a) { return s->step(a); }
    int generals_get_winner(GameState* s) { return s->winner; }
    int generals_get_step(GameState* s) { return s->current_step; }
    bool generals_is_stalemate(GameState* s) { return s->stalemate; }
    int generals_get_width(GameState* s) { return s->width; }
    int generals_get_height(GameState* s) { return s->height; }
    int generals_skip_action(GameState* s) { return s->width * s->height * 8; }
    int generals_get_owner(GameState* s, int i) { return s->owner[i]; }
    int generals_get_army(GameState* s, int i) { return s->army[i]; }
    int generals_get_terrain(GameState* s, int i) { return s->terrain[i]; }

    void generals_get_obs(GameState* s, float* buf, int pid) { s->get_obs(buf, pid); }
    void generals_get_action_mask(GameState* s, bool* buf, int pid) { s->get_action_mask(buf, pid); }
    void generals_get_grid_data(GameState* s, int8_t* o, int16_t* a, uint8_t* t) { s->get_grid_data(o, a, t); }

    GameState* generals_clone(GameState* s) {
        GameState* c = new GameState(); std::memcpy(c, s, sizeof(GameState)); c->rng = s->rng; return c;
    }

    // ---------- BeliefState ----------
    BeliefState* belief_create(int w, int h, int pid) {
        BeliefState* b = new BeliefState(); b->init(w, h, pid); return b;
    }
    void belief_destroy(BeliefState* b) { delete b; }
    void belief_observe(BeliefState* b, GameState* t, int turn) { b->observe(*t, turn); }
    void belief_determinize(BeliefState* b, GameState* g, unsigned int seed) {
        std::mt19937 rng(seed); b->determinize(*g, rng);
    }
    void belief_get_obs(BeliefState* b, float* buf) { b->get_fog_obs(buf); }
    void belief_get_heatmap(BeliefState* b, float* buf) { b->get_bayes_heatmap(buf); }
    int belief_best_general(BeliefState* b) { return b->bayes.get_best_pos(b->width, b->height); }
    void belief_get_probs(BeliefState* b, float* buf) {
        int t = b->width * b->height; for (int i = 0; i < t; ++i) buf[i] = b->bayes.get_prob(i);
    }

    // ---------- VectorFlowField ----------
    VectorFlowField* flow_create() { VectorFlowField* f = new VectorFlowField(); f->init(); return f; }
    void flow_destroy(VectorFlowField* f) { delete f; }
    int flow_compute(VectorFlowField* f, BeliefState* b, int target, MoveCmd* cmds, int maxc) {
        return f->compute(*b, target, cmds, maxc);
    }

    // ---------- IS-MCTS ----------
    ISMCTSEngine* mcts_create(unsigned int seed) {
        ISMCTSEngine* m = new ISMCTSEngine(); m->rng.seed(seed); return m;
    }
    void mcts_destroy(ISMCTSEngine* m) { delete m; }
    int mcts_search(ISMCTSEngine* m, BeliefState* b, int player, int n_det, int n_mcts) {
        MCTSAction act = m->search(*b, player, n_det, n_mcts);
        return m->to_action_id(act, b->width, b->height);
    }
    int mcts_search_flow(ISMCTSEngine* m, BeliefState* b, int player, int n_det, int n_mcts) {
        MCTSAction act = m->search_flow(*b, player, n_det, n_mcts);
        return m->to_action_id(act, b->width, b->height);
    }
    int mcts_search_flow_with_policy(ISMCTSEngine* m, BeliefState* b, int player,
                                      int n_det, int n_mcts, float* policy_buf) {
        MCTSAction act = m->search_flow_with_policy(*b, player, n_det, n_mcts, policy_buf);
        return m->to_action_id(act, b->width, b->height);
    }
    int mcts_search_nn(ISMCTSEngine* m, BeliefState* b, int player,
                       int n_det, int n_mcts,
                       const float* policy_prior, float value_prior) {
        MCTSAction act = m->search_nn(*b, player, n_det, n_mcts, policy_prior, value_prior);
        return m->to_action_id(act, b->width, b->height);
    }
    int mcts_search_nn_auto(ISMCTSEngine* m, BeliefState* b, int player,
                            int n_det, int n_mcts, NNPredictor* nn) {
        MCTSAction act = m->search_nn_auto(*b, player, n_det, n_mcts, *nn);
        return m->to_action_id(act, b->width, b->height);
    }

    // ---------- NNPredictor 创建和销毁 ----------
    NNPredictor* nn_create(const char* model_path) {
        NNPredictor* nn = new NNPredictor();
        if (!nn->load(model_path)) {
            delete nn;
            return nullptr;
        }
        return nn;
    }
    void nn_destroy(NNPredictor* nn) { delete nn; }

    // ---------- BotMode 查询 ----------
    int bot_get_default_mcts(int mode) {
        return get_default_mcts(static_cast<BotMode>(mode));
    }

    int mcts_search_nn_with_policy(ISMCTSEngine* m, BeliefState* b, int player,
                                   int n_det, int n_mcts,
                                   const float* policy_prior, float value_prior,
                                   float* policy_out) {
        MCTSAction act = m->search_nn_with_policy(*b, player, n_det, n_mcts,
                                                   policy_prior, value_prior, policy_out);
        return m->to_action_id(act, b->width, b->height);
    }

    // ---------- 双动作步进 ----------
    int generals_step_dual(GameState* s, int action0, int action1) {
        // 随机先后手，apply 双方动作，tick，检查胜负
        bool red_first = (s->rng() % 2 == 0);
        if (red_first) {
            int r0, c0, dir0, half0;
            if (s->decode_action(action0, r0, c0, dir0, half0))
                s->apply_move(0, r0, c0, dir0, half0);
            int r1, c1, dir1, half1;
            if (s->decode_action(action1, r1, c1, dir1, half1))
                s->apply_move(1, r1, c1, dir1, half1);
        } else {
            int r1, c1, dir1, half1;
            if (s->decode_action(action1, r1, c1, dir1, half1))
                s->apply_move(1, r1, c1, dir1, half1);
            int r0, c0, dir0, half0;
            if (s->decode_action(action0, r0, c0, dir0, half0))
                s->apply_move(0, r0, c0, dir0, half0);
        }
        s->tick();
        s->current_step++;

        bool alive_changed = false;
        for (int p = 0; p < 2; ++p) {
            if (s->is_alive[p]) {
                bool found_gen = false;
                for (int i = 0; i < s->width * s->height; ++i)
                    if (s->terrain[i] == T_GENERAL && s->owner[i] == p) { found_gen = true; break; }
                if (!found_gen) { s->is_alive[p] = false; alive_changed = true; }
            }
        }
        if (alive_changed) {
            if (!s->is_alive[0] && s->is_alive[1]) s->winner = 1;
            else if (s->is_alive[0] && !s->is_alive[1]) s->winner = 0;
            else if (!s->is_alive[0] && !s->is_alive[1]) s->winner = 2;
        }
        if (s->current_step >= s->max_steps && s->winner == -1) {
            int t0 = s->tiebreaker_score(0), t1 = s->tiebreaker_score(1);
            if (t0 > t1) s->winner = 0;
            else if (t1 > t0) s->winner = 1;
            else s->winner = 2;
        }
        return s->winner;
    }

    // ---------- ScriptAgent ----------
    int script_get_action(GameState* s, int player, int personality) {
        ScriptAgent agent;
        return agent.get_action(*s, player, ScriptPersonality(personality));
    }
    int generals_script_step(GameState* s, int player0_action, int personality) {
        // 同 step() 但玩家1使用脚本 AI 替代随机
        bool red_first = (s->rng() % 2 == 0);
        ScriptAgent agent;

        if (red_first) {
            int r, c, dir, half;
            if (s->decode_action(player0_action, r, c, dir, half))
                s->apply_move(0, r, c, dir, half);
            // 玩家1: 脚本 AI
            int act1 = agent.get_action(*s, 1, ScriptPersonality(personality));
            int r1, c1, dir1, half1;
            if (s->decode_action(act1, r1, c1, dir1, half1))
                s->apply_move(1, r1, c1, dir1, half1);
        } else {
            // 玩家1 (蓝方) 先走
            int act1 = agent.get_action(*s, 1, ScriptPersonality(personality));
            int r1, c1, dir1, half1;
            if (s->decode_action(act1, r1, c1, dir1, half1))
                s->apply_move(1, r1, c1, dir1, half1);
            // 玩家0 (红方)
            int r, c, dir, half;
            if (s->decode_action(player0_action, r, c, dir, half))
                s->apply_move(0, r, c, dir, half);
        }

        s->tick();
        s->current_step++;

        // 军队自然增长后检查将军是否被攻占后死亡
        bool alive_changed = false;
        for (int p = 0; p < 2; ++p) {
            if (s->is_alive[p]) {
                bool found_gen = false;
                for (int i = 0; i < s->width * s->height; ++i)
                    if (s->terrain[i] == T_GENERAL && s->owner[i] == p) { found_gen = true; break; }
                if (!found_gen) { s->is_alive[p] = false; alive_changed = true; }
            }
        }

        if (alive_changed) {
            if (!s->is_alive[0] && s->is_alive[1]) s->winner = 1;
            else if (s->is_alive[0] && !s->is_alive[1]) s->winner = 0;
            else if (!s->is_alive[0] && !s->is_alive[1]) s->winner = 2; // draw
        }

        if (s->current_step >= s->max_steps && s->winner == -1) {
            int t0 = s->tiebreaker_score(0), t1 = s->tiebreaker_score(1);
            if (t0 > t1) s->winner = 0;
            else if (t1 > t0) s->winner = 1;
            else s->winner = 2;
        }
        return s->winner;
    }

} // extern "C"


// ============================================================
// 独立性能测试
// ============================================================
int main() {
    constexpr int N_GAMES = 100000;
    constexpr int MAX_STEPS = 300;

    std::cout << "C++ generals.io 引擎性能测试\n";
    std::cout << "模拟 " << N_GAMES << " 局 (每局最多 " << MAX_STEPS << " 步)\n";
    std::cout << "地图: 12x12, 对手: 随机 Bot\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    long long total_steps = 0;
    int wins = 0, losses = 0, draws = 0;

    for (int g = 0; g < N_GAMES; ++g) {
        GameState* state = generals_create(12, 12, MAX_STEPS, g);
        int winner = -1;

        while (winner == -1) {
            int skip = 12 * 12 * 8;
            bool mask[12 * 12 * 8 + 1];
            state->get_action_mask(mask, 0);

            int* valid = new int[skip];
            int nv = 0;
            for (int a = 0; a < skip; ++a) if (mask[a]) valid[nv++] = a;

            int action = (nv > 0) ? valid[state->rng() % nv] : skip;
            delete[] valid;
            winner = state->step(action);
            total_steps++;
        }

        if (winner == 0) wins++;
        else if (winner == 1) losses++;
        else draws++;
        generals_destroy(state);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "结果:\n";
    std::cout << "  总耗时: " << elapsed << " 秒\n";
    std::cout << "  每秒: " << (N_GAMES / elapsed) << " 局/秒\n";
    std::cout << "  每秒: " << (total_steps / elapsed) << " 步/秒\n";
    std::cout << "  每局: " << (elapsed / N_GAMES * 1000) << " ms\n";
    std::cout << "  胜率: " << (100.0 * wins / N_GAMES) << "%\n";
    std::cout << "  红胜 " << wins << ", 蓝胜 " << losses << ", 平局 " << draws << "\n";

    // --- IS-MCTS 快速性能测试 ---
    std::cout << "\n--- IS-MCTS 性能测试 ---\n";
    GameState* gs = generals_create(12, 12, 200, 42);
    BeliefState* bs = belief_create(12, 12, 0);

    // 打 50 步再搜索
    for (int i = 0; i < 50; ++i) {
        int skip = 12 * 12 * 8;
        bool mask[12 * 12 * 8 + 1];
        gs->get_action_mask(mask, 0);
        int valid[256], nv = 0;
        for (int a = 0; a < skip; ++a) if (mask[a]) valid[nv++] = a;
        int act = (nv > 0) ? valid[gs->rng() % nv] : skip;
        gs->step(act);
        bs->observe(*gs, gs->current_step);
    }

    ISMCTSEngine mcts;
    mcts.rng.seed(42);

    auto mt0 = std::chrono::high_resolution_clock::now();
    int n_searches = 5;
    for (int i = 0; i < n_searches; ++i) {
        MCTSAction act = mcts.search(*bs, 0, 4, 200);  // 4宇宙×200迭代=800总迭代
        int aid = mcts.to_action_id(act, 12, 12);
        if (act.is_null) std::cout << "  search " << i << ": Wait\n";
        else std::cout << "  search " << i << ": (" << act.src_idx/12 << "," << act.src_idx%12
                       << ") -> (" << act.dest_idx/12 << "," << act.dest_idx%12 << ")\n";
    }
    auto mt1 = std::chrono::high_resolution_clock::now();
    double mcts_elapsed = std::chrono::duration<double>(mt1 - mt0).count();
    std::cout << "  " << n_searches << " 次搜索耗时 " << mcts_elapsed << "s (每次 " << (mcts_elapsed/n_searches*1000) << "ms)\n";

    belief_destroy(bs);
    generals_destroy(gs);

    return 0;
}
