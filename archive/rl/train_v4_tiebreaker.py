"""
V4 绝杀平局 Self-Play 训练脚本

核心改动 vs V3 Self-Play:
  - 环境: GeneralsEnvV4TieBreaker (绝杀平局 + 纯净奖励)
  - 模型: 从 generals_ppo_v3_master 继续训练
  - 联赛: 每 20480 步保存 checkpoint
  - 对手: 70% 历史模型 / 30% 最新
"""
import os
import glob
import random
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from generals_gym_v4_tiebreaker import GeneralsEnvV4TieBreaker
from train_v3_base import GeneralsCNNV3, mask_fn_v3


class V4LeagueCallback(BaseCallback):
    """联赛对手池：70% 历史 / 30% 最新，每 20480 步轮换"""
    def __init__(self, pool_dir="league_pool_v4", update_freq=20480, verbose=1):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.update_freq = update_freq
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.update_freq == 0:
            model_path = os.path.join(self.pool_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)

            models = glob.glob(os.path.join(self.pool_dir, "*.zip"))

            # 70% 历史 / 30% 最新
            if random.random() < 0.3:
                chosen_opponent = model_path
            else:
                chosen_opponent = random.choice(models)

            if self.verbose > 0:
                print(f"\n[V4 League] 步数 {self.num_timesteps}: 对手 -> {os.path.basename(chosen_opponent)}")

            self.training_env.env_method("load_opponent", chosen_opponent)

        return True


def start_v4_training():
    print("=" * 60)
    print("  🚀 V4 绝杀平局 Self-Play 训练")
    print("  🌟 纯净奖励 | 绝杀平局 | 联赛对手池")
    print("=" * 60)

    base_env = GeneralsEnvV4TieBreaker(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    pool_dir = "league_pool_v4"
    os.makedirs(pool_dir, exist_ok=True)

    # 从 V3 大师模型开始
    start_model = "generals_ppo_v3_master"
    v0_path = os.path.join(pool_dir, "model_step_0.zip")

    print(f"\n📦 加载种子模型: {start_model}")
    model = MaskablePPO.load(start_model, env=env, custom_objects={
        "learning_rate": 1e-4,
        "ent_coef": 0.02,  # 保持探索
    })
    model.save(v0_path)

    # 种子对手
    try:
        base_env.load_opponent(v0_path)
        print(f"✅ 种子对手: model_step_0.zip（自己）")
    except:
        print(f"⚠️ 种子对手加载失败，使用随机 Bot")

    callback = V4LeagueCallback(pool_dir=pool_dir, update_freq=20480)

    total_steps = 500000  # 先跑 50 万步看看效果
    print(f"\n=== 🚀 开始 V4 绝杀平局训练（{total_steps:,} 步）===")
    print("   奖励: 仅 win=+20 / lose=-20 / 绝杀平局算分")
    print("   预计耗时: ~1-2 小时 (CPU)\n")

    model.learn(total_timesteps=total_steps, callback=callback)

    model.save("generals_ppo_v4_master")
    print("\n" + "=" * 60)
    print("  🏆 V4 训练完成！")
    print(f"  模型: generals_ppo_v4_master.zip")
    print("=" * 60)


if __name__ == "__main__":
    start_v4_training()
