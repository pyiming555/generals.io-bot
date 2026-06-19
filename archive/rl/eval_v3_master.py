"""
V3 大师级模型百局评估脚本
vs 随机 Bot（不加载对手模型）
"""
import time
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from generals_gym_v3_selfplay import GeneralsEnvV3SelfPlay
from train_v3_base import GeneralsCNNV3, mask_fn_v3


def evaluate_100():
    print("=" * 50)
    print("  🏆 V3 大师模型百局盲测 (vs 随机 Bot)")
    print("=" * 50)

    base_env = GeneralsEnvV3SelfPlay(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    print(f"\n📦 加载 models/generals_ppo_v3_master ...")
    model = MaskablePPO.load("generals_ppo_v3_master", env=env)

    wins, losses, draws = 0, 0, 0
    steps_total = 0
    game_lengths = []
    win_durations = []

    print("\n=== 开始 100 局测试 ===")
    start_time = time.time()

    for i in range(100):
        obs, info = env.reset()
        terminated, truncated = False, False
        game_steps = 0

        while not (terminated or truncated):
            action_masks = mask_fn_v3(env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            game_steps += 1

        steps_total += game_steps
        game_lengths.append(game_steps)

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
    print("  📊 100 局最终结果")
    print("=" * 50)
    print(f"  🔴 AI 胜利 (斩首): {wins} 局 ({wins:.1f}%)")
    print(f"  🔵 随机 Bot 胜利 : {losses} 局 ({losses:.1f}%)")
    print(f"  🤝 平局 (600步)   : {draws} 局 ({draws:.1f}%)")
    print(f"  ⏱️  总耗时: {elapsed:.1f}s | 平均每局 {elapsed/100:.2f}s")
    print(f"  📏 平均对局步数: {steps_total/100:.0f} 步")
    if wins > 0:
        print(f"  ⚔️  平均斩首局步数: {sum(win_durations)/len(win_durations):.0f} 步")
    if losses > 0:
        print(f"  😵 平均被斩首步数: {sum(game_lengths[i] for i, w in enumerate([base_env.winner == 1] * 100) if w)/losses:.0f} 步" if losses > 0 else "")
    print("=" * 50)

    if wins >= 95:
        print("💡 评价: 🏆 绝对碾压！大师级模型已经出神入化")
    elif wins >= 80:
        print("💡 评价: ✅ 非常强力，随机 Bot 几乎没有还手之力")
    elif wins >= 60:
        print("💡 评价: ⚡ 有一定优势，但偶尔会失误")
    else:
        print("💡 评价: 🔄 还需要继续训练")


if __name__ == "__main__":
    evaluate_100()
