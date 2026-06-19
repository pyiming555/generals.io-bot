"""
V6 Phase A — 斩首重赏微调训练

基于 V5 master 微调，分阶段引入斩首重赏和时间窗口压缩。

Phase A (100K 步):
  - max_steps=450（从 600 砍到 450）
  - 斩首奖励 = +50 / 绝杀保底 = +20
  - 步数惩罚 = -0.01/步
  - 对手池: 10%随机 + 10%V5大师 + 40%最新 + 40%历史
"""
import os
import glob
import random
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from generals_gym_v6 import GeneralsEnvV6
from train_v3_base import GeneralsCNNV3, mask_fn_v3

V5_SEED = "generals_ppo_v5_master"
POOL_DIR = "league_pool_v6"
TOTAL_STEPS = 100000


class V6LeagueCallback(BaseCallback):
    def __init__(self, pool_dir=POOL_DIR, v5_seed=V5_SEED,
                 update_freq=20480, verbose=1):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.update_freq = update_freq
        self.v5_seed = os.path.join(pool_dir, "model_v5_seed.zip")
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.update_freq == 0:
            step = self.num_timesteps
            model_path = os.path.join(self.pool_dir, f"model_step_{step}.zip")
            self.model.save(model_path)

            models = glob.glob(os.path.join(self.pool_dir, "*.zip"))

            # 🌟 V6 多样化对手池
            rand_val = random.random()
            if rand_val < 0.10:
                chosen = "random"
                opp_name = "Random Bot"
            elif rand_val < 0.20:
                chosen = self.v5_seed  # V5 大师作为固定强敌
                opp_name = "V5 Master"
            elif rand_val < 0.60:
                chosen = model_path  # 最新的自己 (40%)
                opp_name = "Latest Self"
            else:
                chosen = random.choice(models) if models else model_path
                opp_name = os.path.basename(chosen) if chosen != "random" else "Random"

            if self.verbose > 0:
                print(f"\n[V6 League] {step:>7d}: 对手 -> {opp_name}")

            self.training_env.env_method("load_opponent", chosen)
        return True


def start_v6_phase_a():
    print("=" * 60)
    print("  🚀 V6 Phase A — 斩首重赏微调训练")
    print("  ⚔️  斩首=+50 | 绝杀=+20 | 步数惩罚=-0.01")
    print(f"  📏  max_steps=450 | 目标 {TOTAL_STEPS:,} 步")
    print("=" * 60)

    base_env = GeneralsEnvV6(
        width=12, height=12, max_steps=450,
        decap_reward=50.0, tiebreaker_reward=20.0,
        step_penalty=0.01, adv_scale=0.001,
    )
    env = ActionMasker(base_env, mask_fn_v3)

    os.makedirs(POOL_DIR, exist_ok=True)

    print(f"\n📦 加载种子模型: {V5_SEED}")
    model = MaskablePPO.load(V5_SEED, env=env, custom_objects={
        "learning_rate": 1e-4,
        "ent_coef": 0.02,
    })

    # 保存 V5 种子到联赛池
    v5_seed_path = os.path.join(POOL_DIR, "model_v5_seed.zip")
    model.save(v5_seed_path)
    print(f"✅ V5 种子已保存到联赛池")

    # 初始对手：V5 种子
    try:
        base_env.load_opponent(v5_seed_path)
        print(f"✅ 初始对手: V5 Master")
    except:
        print(f"⚠️ 初始对手加载失败")

    callback = V6LeagueCallback(pool_dir=POOL_DIR, v5_seed=v5_seed_path)

    print(f"\n=== 🚀 Phase A 开始 ({TOTAL_STEPS:,} 步) ===")
    print("   对手池: 10%随机 + 10%V5大师 + 40%最新 + 40%历史")
    print("   预计 ~20-30 分钟\n")

    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    model.save("generals_ppo_v6_phaseA")
    print("\n" + "=" * 60)
    print("  🏆 V6 Phase A 完成！")
    print("  模型: generals_ppo_v6_phaseA.zip")
    print("=" * 60)


if __name__ == "__main__":
    start_v6_phase_a()
