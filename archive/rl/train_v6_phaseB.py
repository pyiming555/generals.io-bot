"""
V6 Phase B — 斩首重赏微调 (max_steps=300, 斩首+100)
基于 Phase A 继续训练
"""
import os, glob, random, time, sys
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generals_gym_v6 import GeneralsEnvV6
from train_v3_base import mask_fn_v3


class V6LeagueCallback(BaseCallback):
    def __init__(self, pool_dir="league_pool_v6b", v5_master="generals_ppo_v5_master",
                 update_freq=20480, verbose=1):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.update_freq = update_freq
        self.v5_master = v5_master
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.update_freq != 0:
            return True

        model_path = os.path.join(self.pool_dir, f"model_step_{self.num_timesteps}.zip")
        self.model.save(model_path)
        models = glob.glob(os.path.join(self.pool_dir, "*.zip"))

        r = random.random()
        if r < 0.10:
            opponent = "random"
            opp_name = "Random Bot"
        elif r < 0.30:
            opponent = self.v5_master
            opp_name = "V5 Master"
        elif r < 0.65:
            opponent = model_path  # 最新自己
            opp_name = "Latest Self"
        else:
            opponent = random.choice([m for m in models if m != model_path]) if models else model_path
            opp_name = f"History {os.path.basename(opponent)}"

        if self.verbose > 0:
            print(f"\n[V6B League] 步数 {self.num_timesteps}: 对手 -> {opp_name}")

        self.training_env.env_method("load_opponent", opponent)
        return True


def train_phaseB():
    print("=" * 60)
    print("  🚀 V6 Phase B: 斩首重赏微调")
    print("  max_steps=300 | 斩首+100 | 步罚-0.01")
    print("=" * 60)

    base_env = GeneralsEnvV6(width=12, height=12, max_steps=300)
    env = ActionMasker(base_env, mask_fn_v3)

    pool_dir = "league_pool_v6b"
    os.makedirs(pool_dir, exist_ok=True)

    start_model = "generals_ppo_v6_phaseA"
    print(f"\n📦 加载 {start_model}.zip ...")
    v0_path = os.path.join(pool_dir, "model_step_0.zip")

    model = MaskablePPO.load(start_model, env=env, custom_objects={
        "learning_rate": 1e-4,
        "ent_coef": 0.03,  # 提高探索
    })
    model.save(v0_path)
    base_env.load_opponent(v0_path)

    callback = V6LeagueCallback(pool_dir=pool_dir, v5_master="generals_ppo_v5_master",
                                update_freq=20480)

    print(f"\n=== 🚀 Phase B: 400,000 步 ===")
    print("   斩首+100 | 绝杀+20 | 每步-0.01 | max_steps=300")
    print("   预计耗时: ~1.5-2 小时 (CPU)\n")

    model.learn(total_timesteps=400000, callback=callback)
    model.save("generals_ppo_v6_phaseB")

    print("\n" + "=" * 60)
    print("  🏆 V6 Phase B 完成！")
    print("  模型: generals_ppo_v6_phaseB.zip")
    print("=" * 60)


if __name__ == "__main__":
    train_phaseB()
