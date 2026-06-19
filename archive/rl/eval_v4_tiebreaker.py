"""
V4 绝杀平局 — 评估脚本
评估结果基于纯净奖励规则：赢=+20，输=-20，绝杀平局按分算
"""
import time
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from generals_gym_v4_tiebreaker import GeneralsEnvV4TieBreaker
from train_v3_base import GeneralsCNNV3, mask_fn_v3


def evaluate_v4(model_name, num_games=100, deterministic=True):
    print("=" * 50)
    label = "deterministic=True" if deterministic else "stochastic"
    print(f"  🏆 V4 绝杀平局评估 ({label})")
    print(f"  vs 随机 Bot | {num_games}局")
    print("=" * 50)

    base_env = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    print(f"\n📦 加载 {model_name} ...")
    model = MaskablePPO.load(model_name, env=env)

    wins, losses, draws, tiebreaker_wins = 0, 0, 0, 0
    steps_total = 0
    win_durations = []

    print(f"\n=== 开始 {num_games} 局测试 ===")
    start_time = time.time()

    for i in range(num_games):
        obs, info = env.reset()
        terminated, truncated = False, False
        game_steps = 0

        while not (terminated or truncated):
            action_masks = mask_fn_v3(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            game_steps += 1

        steps_total += game_steps
        is_tiebreaker = base_env.stalemate

        if base_env.winner == 0:
            wins += 1
            win_durations.append(game_steps)
            if is_tiebreaker:
                tiebreaker_wins += 1
                mark = "📐"  # 绝杀胜出
            else:
                mark = "⚔️"
        elif base_env.winner == 1:
            losses += 1
            mark = "💀"
        else:
            draws += 1
            mark = "❓"

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{num_games}] {mark} | 胜{wins} 负{losses} 平{draws} | "
                  f"胜率 {wins/(i+1)*100:.1f}%")

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"  📊 {num_games}局最终结果 ({label})")
    print("=" * 50)
    print(f"  ⚔️  斩首胜利: {wins - tiebreaker_wins} 局")
    print(f"  📐  绝杀胜利: {tiebreaker_wins} 局")
    print(f"  🔴  总胜利:   {wins} 局 ({wins:.1f}%)")
    print(f"  🔵  失败:     {losses} 局 ({losses:.1f}%)")
    print(f"  ❓  绝对平局: {draws} 局 ({draws:.1f}%)")
    print(f"  ⏱️  总耗时: {elapsed:.1f}s | 均局 {elapsed/num_games:.2f}s")
    print(f"  📏  平均对局步数: {steps_total/num_games:.0f}")
    if wins > 0:
        print(f"  ⚔️  平均胜局步数: {sum(win_durations)/len(win_durations):.0f}")
    print("=" * 50)

    if wins >= 95:
        print("💡 🏆 绝对碾压！")
    elif wins >= 80:
        print("💡 ✅ 非常强力")
    elif wins >= 50:
        print("💡 ⚡ 刚过及格线")
    else:
        print("💡 🔄 还需继续训练")

    return wins, losses, draws, tiebreaker_wins


if __name__ == "__main__":
    import sys
    det = True
    model_name = "generals_ppo_v4_master"
    if len(sys.argv) > 1:
        det = sys.argv[1] != "stochastic"
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    evaluate_v4(model_name, num_games=100, deterministic=det)
