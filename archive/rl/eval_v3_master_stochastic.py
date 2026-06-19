"""
V3 大师模型百局评估 — 随机策略版 (deterministic=False)
"""
import time
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from generals_gym_v3_selfplay import GeneralsEnvV3SelfPlay
from train_v3_base import GeneralsCNNV3, mask_fn_v3


def evaluate_100_stochastic():
    print("=" * 50)
    print("  🏆 V3 大师模型百局盲测 (deterministic=False)")
    print("=" * 50)

    base_env = GeneralsEnvV3SelfPlay(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    print(f"\n📦 加载 generals_ppo_v3_master ...")
    model = MaskablePPO.load("generals_ppo_v3_master", env=env)

    wins, losses, draws = 0, 0, 0
    steps_total = 0
    win_durations = []

    print("\n=== 开始 100 局测试 (随机策略) ===")
    start_time = time.time()

    for i in range(100):
        obs, info = env.reset()
        terminated, truncated = False, False
        game_steps = 0

        while not (terminated or truncated):
            action_masks = mask_fn_v3(env)
            # 🌟 关键改动：deterministic=False
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            game_steps += 1

        steps_total += game_steps

        if base_env.winner == 0:
            wins += 1
            win_durations.append(game_steps)
            mark = "✅"
        elif base_env.winner == 1:
            losses += 1
            mark = "❌"
        else:
            draws += 1
            mark = "🤝"

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/100] {mark} | 当前: 胜 {wins} 负 {losses} 平 {draws} | "
                  f"胜率 {wins/(i+1)*100:.1f}%")

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("  📊 100 局最终结果 (deterministic=False)")
    print("=" * 50)
    print(f"  🔴 AI 胜利 (斩首): {wins} 局 ({wins:.1f}%)")
    print(f"  🔵 随机 Bot 胜利 : {losses} 局 ({losses:.1f}%)")
    print(f"  🤝 平局 (600步)   : {draws} 局 ({draws:.1f}%)")
    print(f"  ⏱️  总耗时: {elapsed:.1f}s | 平均每局 {elapsed/100:.2f}s")
    print(f"  📏 平均对局步数: {steps_total/100:.0f} 步")
    if wins > 0:
        print(f"  ⚔️  平均斩首步数: {sum(win_durations)/len(win_durations):.0f} 步")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_100_stochastic()
