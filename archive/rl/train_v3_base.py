"""
generals.io V3 课程学习训练脚本
===============================
改动点 vs train_cnn_v3.py：
  - n_steps=2048 → 减少网络更新频率，提高吞吐量
  - batch_size=512 → 更大批处理稳定梯度
  - total_timesteps=500000 → 给足探索时间
  - 环境已降级城市兵力 40-50 → 10-15，并加入破城奖励 +2.0/座
"""
import time
import torch
import torch.nn as nn
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 导入 V3 完全体环境（已修改为课程学习模式）
from generals_gym_v3 import GeneralsEnvV3


# ==========================================
# 🌟 V3 CNN：7通道 + 12x12 地图专用提取器
# ==========================================
class GeneralsCNNV3(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 7

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def mask_fn_v3(env: gym.Env):
    return env.unwrapped.valid_action_mask(player_id=0)


CHECKPOINT_PATH = "generals_ppo_v3_base_curriculum"
CHECKPOINT_INTERVAL = 50000  # 每5万步保存一次检查点


def train_v3_curriculum():
    print("=" * 60)
    print("  🎓 课程学习 Phase 1: V3 新手村模式")
    print("=" * 60)
    print("  地图: 12x12 | 城市兵力: 10-15 (原40-50)")
    print("  奖励: 占地+0.05/格 + 破城+5.0/座 + 斩首±20.0")
    print("  训练步数: 500,000 | n_steps=2048 | batch_size=512")
    print("  检查点: 每50k步自动保存 (可从中断处恢复)")
    print("=" * 60)
    
    base_env = GeneralsEnvV3(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    obs, info = env.reset()
    print(f"\n📐 观测空间: {obs.shape} (通道: {obs.shape[0]}, {obs.shape[1]}x{obs.shape[2]})")
    print(f"🕹️ 动作空间: {env.action_space.n} (12x12x8+1 = {12*12*8+1})")
    env.render()

    policy_kwargs = dict(
        features_extractor_class=GeneralsCNNV3,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # 检测是否有检查点，支持恢复训练
    import os
    checkpoint_zip = CHECKPOINT_PATH + ".zip"
    if os.path.exists(checkpoint_zip):
        print(f"\n🔄 检测到已有检查点 {checkpoint_zip}，从中恢复训练...\n")
        model = MaskablePPO.load(checkpoint_zip, env=env)
        # 从已完成的步数继续
        saved_steps = getattr(model, '_total_timesteps', 0)
        remaining = max(500000 - saved_steps, 0)
        print(f"   已完成 {saved_steps} 步，还需训练 {remaining} 步\n")
    else:
        print("\n🚀 开始从零训练 V3 新手村模型 (50万步)...")
        print("   预计耗时: 1.5 - 2 小时 (CPU)\n")
        saved_steps = 0
        remaining = 500000
        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=1e-4,    # 🌟 从3e-4降到1e-4，学慢点更稳
            gamma=0.99,
            n_steps=2048,      # 🌟 减少更新频率，增加吞吐量
            batch_size=512,    # 🌟 更大批处理，稳定梯度
            ent_coef=0.02,     # 🌟 强制探索！禁止过早躺平
        )

    # 分段训练：每段 CHECKPOINT_INTERVAL 步保存一次
    total_completed = saved_steps
    while total_completed < 500000:
        segment = min(CHECKPOINT_INTERVAL, 500000 - total_completed)
        model.learn(total_timesteps=segment, reset_num_timesteps=False)
        total_completed += segment
        model._total_timesteps = total_completed
        model.save(CHECKPOINT_PATH)
        import time
        t = time.strftime("%H:%M:%S")
        print(f"\n[{t}] ✅ 检查点已保存 ({total_completed}/500000 步, {total_completed/500000*100:.0f}%)\n")

    model.save(CHECKPOINT_PATH)
    print(f"\n✅ V3 课程学习基础模型已保存为 {CHECKPOINT_PATH}.zip")


def evaluate_v3_curriculum(num_games=20):
    """评估 V3 新手村模型 vs 随机 Bot"""
    print(f"\n{'='*40}")
    print(f"  🏋️ 评估 V3 课程模型 vs 随机 Bot ({num_games}局)")
    print(f"{'='*40}")
    
    base_env = GeneralsEnvV3()
    env = ActionMasker(base_env, mask_fn_v3)
    
    model = MaskablePPO.load("generals_ppo_v3_base_curriculum")
    
    wins, losses, draws = 0, 0, 0
    start = time.time()
    
    for i in range(num_games):
        obs, info = env.reset()
        done, trunc = False, False
        while not (done or trunc):
            masks = mask_fn_v3(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
        if base_env.winner == 0:
            wins += 1
        elif base_env.winner == 1:
            losses += 1
        else:
            draws += 1
    
    elapsed = time.time() - start
    print(f"\n📊 结果 ({num_games}局, {elapsed:.1f}s):")
    print(f"  🔴 AI 胜利: {wins} 局 ({wins/num_games*100:.0f}%)")
    print(f"  🔵 随机 Bot: {losses} 局 ({losses/num_games*100:.0f}%)")
    print(f"  🤝 平局:     {draws} 局 ({draws/num_games*100:.0f}%)")
    
    if wins / num_games >= 0.8:
        print("\n💡 评价: ✅ 准备就绪！可以进入 Phase 2")
        print("   - 恢复城市兵力到 40-50")
        print("   - 或者开启 Self-Play 联赛训练")
    elif wins / num_games >= 0.5:
        print("\n💡 评价: ⏳ 有点感觉了，再练一会儿")
    else:
        print("\n💡 评价: 🔄 还需要更多步数或调整超参数")


if __name__ == "__main__":
    train_v3_curriculum()
    evaluate_v3_curriculum(num_games=20)
