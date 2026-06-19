"""
从最近 checkpoint 恢复 V3 自对弈联赛训练
用法: python resume_v3_selfplay.py [checkpoint_path]
"""
import sys
import os
import glob
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from train_v3_base import GeneralsCNNV3, mask_fn_v3
from generals_gym_v3_selfplay import GeneralsEnvV3SelfPlay

# ---- 配置 ----
TOTAL_TARGET = 1_000_000   # 原始目标总步数
POOL_DIR = "league_pool_v3"  # V3 自对弈对战池
UPDATE_FREQ = 20480        # 保存频率

class V3LeagueCallback(BaseCallback):
    """联赛训练回调（复用原逻辑）"""
    def __init__(self, pool_dir=POOL_DIR, update_freq=UPDATE_FREQ, verbose=1):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.update_freq = update_freq
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.update_freq == 0:
            model_path = os.path.join(self.pool_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)

            models = glob.glob(os.path.join(self.pool_dir, "*.zip"))
            if random.random() < 0.3:
                chosen_opponent = model_path
            else:
                chosen_opponent = random.choice(models)

            if self.verbose > 0:
                print(f"\n[V3 League] 步数 {self.num_timesteps}: 蓝方切换为 -> {os.path.basename(chosen_opponent)}")
            self.training_env.env_method("load_opponent", chosen_opponent)
        return True


def resume_training(checkpoint):
    print("=" * 60)
    print(f"  🔄 恢复 V3 自对弈训练 (从 {os.path.basename(checkpoint)})")
    print("=" * 60)

    # 解析当前步数
    base_model_name = os.path.splitext(os.path.basename(checkpoint))[0]
    step_current = 0
    for part in base_model_name.split("_"):
        if part.isdigit():
            step_current = int(part)
            break
    steps_remaining = TOTAL_TARGET - step_current
    if steps_remaining <= 0:
        print(f"  ✅ 训练目标 {TOTAL_TARGET} 步已完成! 无需继续.")
        return

    print(f"  当前步数: {step_current}")
    print(f"  还需训练: {steps_remaining} 步")
    print(f"  总目标:   {TOTAL_TARGET} 步")
    print(f"  对战池:   {POOL_DIR}/ (每 {UPDATE_FREQ} 步保存)")
    print(f"  对手选择: 70% 历史 / 30% 最新")
    print("=" * 60)

    # ---- 初始化环境 ----
    base_env = GeneralsEnvV3SelfPlay(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)
    obs, info = env.reset()
    print(f"\n📐 观测空间: {obs.shape}")
    print(f"🕹️  动作空间: {env.action_space.n}")

    # ---- 加载 checkpoint ----
    print(f"\n=== 📦 加载 checkpoint: {checkpoint} ===")
    model = MaskablePPO.load(
        checkpoint,
        env=env,
        custom_objects={
            "learning_rate": 1e-4,
            "ent_coef": 0.01,
        },
    )

    # 确保模型有联赛池里的对手可以打
    models_in_pool = glob.glob(os.path.join(POOL_DIR, "*.zip"))
    if models_in_pool:
        chosen = random.choice(models_in_pool)
        print(f"  初始对手: {os.path.basename(chosen)}")
        base_env.load_opponent(chosen)
    else:
        print("  ⚠️ 对战池为空，保存当前模型作为第一个对手")
        model.save(os.path.join(POOL_DIR, f"model_step_{step_current}.zip"))
        base_env.load_opponent(os.path.join(POOL_DIR, f"model_step_{step_current}.zip"))

    # ---- 联赛回调 ----
    callback = V3LeagueCallback(pool_dir=POOL_DIR, update_freq=UPDATE_FREQ)

    # ---- 🚀 继续训练 ----
    print(f"\n=== 🚀 继续训练 {steps_remaining} 步! ===")
    print(f"   预计耗时: ~{steps_remaining // 80000} 小时 (CPU)\n")

    model.learn(
        total_timesteps=steps_remaining,
        reset_num_timesteps=False,  # 从当前步数继续
        callback=callback,
    )

    # ---- 保存最终模型 ----
    model.save("generals_ppo_v3_master")
    print("\n" + "=" * 60)
    print("  🏆 V3 大师级训练完成！")
    print(f"  模型已保存为 generals_ppo_v3_master.zip")
    print("=" * 60)


if __name__ == "__main__":
    # 找最新的 V3 checkpoint
    checkpoint = "league_pool_v3/model_step_778240.zip"
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]

    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint 不存在: {checkpoint}")
        # 尝试自动找最新的完整模型（排除正在写入的不完整文件）
        models = sorted(glob.glob(os.path.join(POOL_DIR, "model_step_*.zip")))
        # 过滤掉小于 10MB 的不完整文件
        models = [m for m in models if os.path.getsize(m) > 10_000_000]
        if models:
            checkpoint = models[-1]
            print(f"   使用最新完整模型: {checkpoint}")
        else:
            print("   对战池为空或所有模型不完整，无法恢复训练！")
            sys.exit(1)

    resume_training(checkpoint)
