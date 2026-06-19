"""
V3 自对弈联赛训练脚本 (Phase 2: 自我进化)
============================================
将种田流大师丢进黑暗森林 —— 当两个 AI 把地图瓜分完毕后，
为了获得更多分数，它们被迫开始互相攻击，最终演变成斩首修罗场。

改动点 vs train_v3_base.py (Phase 1):
  - 环境: generals_gym_v3_selfplay (占地 0.01, 城市 15-25)
  - ent_coef: 0.02 -> 0.01 (基础已稳定，降低探索让模型打磨杀人技)
  - 训练: 最大池联赛回调（70% 历史版本 / 30% 最新版本）
  - total_timesteps: 500k -> 1M (自对弈需要更多步数)
"""
import os
import glob
import random
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# V3 CNN + 掩码函数 (从 Phase 1 的课程脚本导入，与模型架构一致)
from train_v3_base import GeneralsCNNV3, mask_fn_v3

# Phase 2 专用自对弈环境
from generals_gym_v3_selfplay import GeneralsEnvV3SelfPlay


class V3LeagueCallback(BaseCallback):
    """
    联赛训练回调（V3 加强版）

    每 update_freq 步：
      1. 保存当前模型到 pool
      2. 70% 概率打历史随机版本 / 30% 打最新版本
         （相比 Phase 1 的 80/20，提高自战比例以加速进化）
    """
    def __init__(self, pool_dir="league_pool_v3", update_freq=20480, verbose=1):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.update_freq = update_freq
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.update_freq == 0:
            # 1. 保存当前模型快照
            model_path = os.path.join(self.pool_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)

            # 2. 从池中选择对手
            models = glob.glob(os.path.join(self.pool_dir, "*.zip"))

            # 70% 历史随机 / 30% 最新版本（逼 AI 尽快适应最新战术）
            if random.random() < 0.3:
                chosen_opponent = model_path  # 打最新版自己
            else:
                chosen_opponent = random.choice(models)  # 打历史版

            if self.verbose > 0:
                print(
                    f"\n[V3 League] 步数 {self.num_timesteps}: "
                    f"蓝方切换为 -> {os.path.basename(chosen_opponent)}"
                )

            # 3. 注入对手到环境中
            self.training_env.env_method("load_opponent", chosen_opponent)

        return True


def start_v3_selfplay():
    print("=" * 60)
    print("  🥷 Phase 2: V3 黑暗森林 — 自对弈联赛")
    print("=" * 60)
    print("  地图: 12x12 | 城市兵力: 15-25 (课程进阶)")
    print("  奖励: 占地+0.01/格 (削减) | 破城+5.0/座 | 斩首±20.0")
    print("  训练步数: 1,000,000 | ent_coef=0.01")
    print("  对战池: league_pool_v3/ (每 20480 步保存)")
    print("  对手选择: 70% 历史随机 / 30% 最新版本")
    print("=" * 60)

    # ---- 初始化环境 ----
    base_env = GeneralsEnvV3SelfPlay(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    obs, info = env.reset()
    print(f"\n📐 观测空间: {obs.shape}")
    print(f"🕹️  动作空间: {env.action_space.n}")

    # ---- 准备联赛池 ----
    pool_dir = "league_pool_v3"
    os.makedirs(pool_dir, exist_ok=True)

    # ---- 加载种田流大师 (Phase 1 成果) ----
    starting_model = "generals_ppo_v3_base_curriculum"
    v0_path = os.path.join(pool_dir, "model_step_0.zip")

    print(f"\n=== 🧑‍🌾 加载种田流大师 {starting_model} ===")

    # 加载模型，微调超参：降低 ent_coef 让它开始专注战斗
    model = MaskablePPO.load(
        starting_model,
        env=env,
        custom_objects={
            "learning_rate": 1e-4,
            "ent_coef": 0.01,  # 从 0.02 降到 0.01
        },
    )

    # 第一个对手 = 自己（然后联赛回调会不断进化对手池）
    model.save(v0_path)
    base_env.load_opponent(v0_path)

    # ---- 联赛回调 ----
    callback = V3LeagueCallback(pool_dir=pool_dir, update_freq=20480)

    # ---- 🚀 开练！----
    print("\n=== 🚀 开始 V3 左右互搏！让它们卷起来！(100 万步) ===")
    print("   预计耗时: 3-4 小时 (CPU)\n")
    model.learn(total_timesteps=1000000, callback=callback)

    # ---- 保存最终模型 ----
    model.save("generals_ppo_v3_master")
    print("\n" + "=" * 60)
    print("  🏆 V3 大师级训练完成！")
    print(f"  模型已保存为 generals_ppo_v3_master.zip")
    print("=" * 60)


if __name__ == "__main__":
    start_v3_selfplay()
