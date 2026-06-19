import time
import torch
import torch.nn as nn
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 导入 V3 完全体环境
from generals_gym_v3 import GeneralsEnvV3

# ==========================================
# 🌟 V3 CNN：7通道 + 12x12 地图专用提取器
# ==========================================
class GeneralsCNNV3(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 7

        # 12x12 地图比 8x8 更大，可以加一层深度并提高特征维度
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 动态计算卷积后的维度
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# 提取掩码的回调函数
def mask_fn_v3(env: gym.Env):
    return env.unwrapped.valid_action_mask(player_id=0)

def train_cnn_v3():
    print("=== 初始化 V3 完全体环境 (12x12 + 城市 + 分兵) ===")
    base_env = GeneralsEnvV3(width=12, height=12, max_steps=600)
    env = ActionMasker(base_env, mask_fn_v3)

    obs, info = env.reset()
    print(f"观测空间: {obs.shape} (通道: {obs.shape[0]}, {obs.shape[1]}x{obs.shape[2]})")
    print(f"动作空间: {env.action_space.n} (12x12x8+1 = {12*12*8+1})")
    env.render()

    # 注入自定义 CNN
    policy_kwargs = dict(
        features_extractor_class=GeneralsCNNV3,
        features_extractor_kwargs=dict(features_dim=512),
    )

    print("=== 开始训练 V3 基础模型 (30万步) ===")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=256,
    )
    
    model.learn(total_timesteps=300000)
    model.save("generals_ppo_v3_base")
    print("=== V3 基础模型已保存为 generals_ppo_v3_base.zip ===")

def evaluate_v3_random():
    print("\n=== 评估 V3 AI vs 随机 Bot (10局) ===")
    base_env = GeneralsEnvV3()
    env = ActionMasker(base_env, mask_fn_v3)
    
    model = MaskablePPO.load("generals_ppo_v3_base")
    
    wins, losses, draws = 0, 0, 0
    start = time.time()
    
    for i in range(10):
        obs, info = env.reset()
        done, trunc = False, False
        while not (done or trunc):
            masks = mask_fn_v3(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
        if base_env.winner == 0: wins += 1
        elif base_env.winner == 1: losses += 1
        else: draws += 1
    
    print(f"  10局: W:{wins} L:{losses} D:{draws}  ({time.time()-start:.1f}s)")

if __name__ == "__main__":
    train_cnn_v3()
    evaluate_v3_random()
