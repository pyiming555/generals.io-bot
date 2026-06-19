"""
V6 评估脚本 — 区分斩首胜 vs 绝杀胜
"""
import time, sys
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from generals_gym_v6 import GeneralsEnvV6
from train_v3_base import GeneralsCNNV3, mask_fn_v3


def evaluate(model_name, num_games=100, deterministic=True, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = dict(max_steps=450, decap_reward=50.0, tiebreaker_reward=20.0, step_penalty=0.01)
    label = "deterministic" if deterministic else "stochastic"

    print("=" * 55)
    print(f"  🏆 V6 评估 ({label})")
    print(f"  model={model_name} | {num_games}局 vs 随机Bot")
    print("=" * 55)

    base_env = GeneralsEnvV6(width=12, height=12, **env_kwargs)
    env = ActionMasker(base_env, mask_fn_v3)
    model = MaskablePPO.load(model_name, env=env)

    decap_wins, tie_wins, losses, draws = 0, 0, 0, 0
    decap_steps = []
    total_steps = 0

    print(f"\n=== 开始 {num_games} 局 ===")
    start = time.time()

    for i in range(num_games):
        obs, info = env.reset()
        term, trunc = False, False
        gs = 0
        while not (term or trunc):
            masks = mask_fn_v3(env)
            a, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            gs += 1

        total_steps += gs

        if base_env.winner == 0:
            if base_env.stalemate:
                tie_wins += 1
                mark = "📐"
            else:
                decap_wins += 1
                decap_steps.append(gs)
                mark = "⚔️"
        elif base_env.winner == 1:
            losses += 1
            mark = "💀"
        else:
            draws += 1
            mark = "❓"

        if (i+1) % 10 == 0:
            wr = (decap_wins + tie_wins) / (i+1) * 100
            print(f"  [{i+1:3d}] {mark} | ⚔️{decap_wins} 📐{tie_wins} 💀{losses} | {wr:.0f}%")

    elapsed = time.time() - start

    print("\n" + "=" * 55)
    print(f"  📊 {num_games}局结果 ({label})")
    print("=" * 55)
    print(f"  ⚔️  斩首胜:  {decap_wins} 局 ({decap_wins/num_games*100:.1f}%)")
    print(f"  📐  绝杀胜:  {tie_wins} 局 ({tie_wins/num_games*100:.1f}%)")
    print(f"  🏆  总胜率:  {(decap_wins+tie_wins)/num_games*100:.1f}%")
    print(f"  💀  失败:    {losses} 局 ({losses/num_games*100:.1f}%)")
    print(f"  ❓  绝对平局: {draws} 局")
    print(f"  ⏱   耗时: {elapsed:.1f}s | 均局 {elapsed/num_games:.2f}s")
    print(f"  📏  平均步数: {total_steps/num_games:.0f}")
    if decap_steps:
        print(f"  ⚔️  平均斩首步数: {sum(decap_steps)/len(decap_steps):.0f}")
    print("=" * 55)

    if decap_wins >= 80:
        print("💡 🏆 斩首大师！")
    elif decap_wins >= 50:
        print("💡 ⚔️ 开始学会斩首了！")
    elif decap_wins >= 10:
        print("💡 👶 刚刚体会到斩首的甜头")
    elif tie_wins >= 50:
        print("💡 📐 靠绝杀赢，还需更激进")
    else:
        print("💡 🔄 仍需训练")

    return decap_wins, tie_wins, losses, draws


if __name__ == "__main__":
    det = True
    model_name = "generals_ppo_v6_phaseA"
    games = 100
    if len(sys.argv) > 1:
        det = sys.argv[1] != "stochastic"
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    if len(sys.argv) > 3:
        games = int(sys.argv[3])
    evaluate(model_name, games, det)
