#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

// ================================================================
// 地图常量
// ================================================================
constexpr int MAX_W = 25;
constexpr int MAX_H = 25;
constexpr int MAX_TILES = MAX_W * MAX_H;  // 625

constexpr int T_EMPTY    = 0;
constexpr int T_MOUNTAIN = 1;
constexpr int T_GENERAL  = 2;
constexpr int T_CITY     = 3;

constexpr int MEM_UNKNOWN   = 0;
constexpr int MEM_MOUNTAIN  = 1;
constexpr int MEM_CITY      = 2;
constexpr int MEM_GENERAL   = 3;
constexpr int MEM_EMPTY     = 4;

// ================================================================
// 方向向量
// ================================================================
constexpr int DR[4] = {-1, 1, 0, 0};
constexpr int DC[4] = {0, 0, -1, 1};

// ================================================================
// AI 模式枚举 — 不同场景使用不同的 MCTS 迭代预算
// ================================================================
enum class BotMode : int {
    FAST_TRAIN = 0,  // 快速训练/自对弈 (n_mcts=200, 18.8ms)
    LADDER     = 1,  // 天梯排位       (n_mcts=260, 23.9ms, Sweet Spot!)
    ANALYSIS   = 2,  // 深度复盘分析   (n_mcts=300, 26.9ms)
};

/// 根据 BotMode 返回推荐的 MCTS 迭代数
inline int get_default_mcts(BotMode mode) {
    switch (mode) {
        case BotMode::FAST_TRAIN: return 200;
        case BotMode::LADDER:     return 260;
        case BotMode::ANALYSIS:   return 300;
        default:                  return 260;
    }
}

#endif // COMMON_H
