"""
V5 混合联赛训练脚本 — 绝杀平局 + 微密引导 + 多样化对手池

对手池:
  20% Random Bot  — 虐菜防遗忘
  20% V3 Farmer   — 防守反击（generals_ppo_v3_base.zip）
  30% 最新自己    — 上限提升
  30% 历史版本    — 防遗忘

模型: 从 generals_ppo_v4_master 继续训练
"""
import os
import glob
import random
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from generals_gym_v5_league import GeneralsEnvV5League
from train_v3_base import GeneralsCNNV3, mask_fn_v3


class V5LeagueCallback(BaseCallback):
    """V5 多样化联赛对手池"""
    def __init__(self, pool_dir="league_pool_v5", farmer_model="generals_ppo_v3_base.zip",
                 update_freq=20480, verbose=1):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.update_freq = update_freq
        self.farmer_model = farmer_model
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.update_freq == 0:
            # 保存 checkpoint
            model_path = os.path.join(self.pool_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)

            models = glob.glob(os.path.join(self.pool_dir, "*.zip"))

            # 多样化抽签
            r = random.random()
            if r < 0.20:
                chosen = "random"
                opp_name = "Random Bot 🎲"
            elif r < 0.40:
                chosen = self.farmer_model
                opp_name = "V3 Farmer 🌾"
            elif r < 0.70:
                chosen = model_path  # 打最新的自己
                opp_name = "Latest Self 🆕"
            else:
                chosen = random.choice(models) if models else "random"
                opp_name = os.path.basename(chosen) if chosen != "random" else "Random Bot"

            if self.verbose > 0:
                print(f"\n[V5 League] Step {self.num_timesteps}: 对手 -> {opp_name}")

            self.training_env.env_method("load_opponent", chosen)

        return True


def start_v5_training():
    print("=" * 60)
    print("  🚀 V5 混合联赛训练")
    print("  🌟 微密引导 + 绝杀平局 + 多样化对手池")
    print("=" * 60)
    print("  对手池:")
    print("    • 20% Random Bot")
    print("    • 20% V3 Farmer (generals_ppo_v3_base)")
    print("    • 30% Latest Self")
    print("    • 30% Historical Self")
    print("  奖励:")
    print("    • 优势差×0.001（微密引导）")
    print("    • Win +20 / Lose -20")
    print("=" * 60)

    base_env = GeneralsEnvV5League(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    pool_dir = "league_pool_v5"
    os.makedirs(pool_dir, exist_ok=True)

    # 模型
    start_model = "generals_ppo_v4_master"
    v0_path = os.path.join(pool_dir, "model_step_0.zip")

    print(f"\n📦 加载种子模型: {start_model}")
    model = MaskablePPO.load(start_model, env=env, custom_objects={
        "learning_rate": 1e-4,
        "ent_coef": 0.02,  # 保持探索
    })
    model.save(v0_path)

    callback = V5LeagueCallback(
        pool_dir=pool_dir,
        farmer_model="generals_ppo_v3_base.zip",
        update_freq=20480,
    )

    total_steps = 500000
    print(f"\n=== 🚀 开始 V5 混合联赛训练（{total_steps:,} 步）===")
    print(f"   预计耗时: ~1-2 小时 (CPU)\n")

    model.learn(total_timesteps=total_steps, callback=callback)

    model.save("generals_ppo_v5_master")
    print("\n" + "=" * 60)
    print("  🏆 V5 训练完成！")
    print(f"  模型: generals_ppo_v5_master.zip")
    print("=" * 60)


if __name__ == "__main__":
    start_v5_training()
