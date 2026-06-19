import time
import torch
import torch.nn as nn
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 导入你的环境
from generals_gym import GeneralsRLGymEnv

# ==========================================
# 🌟 核心：自定义 8x8 地图专属的 CNN 提取器 🌟
# ==========================================
class GeneralsCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 输入通道数为 6 (己方, 敌方, 中立, 山脉, 己方塔, 敌方塔)
        n_input_channels = observation_space.shape[0]

        # 设计轻量级 CNN，不使用 MaxPooling，以保留 8x8 的所有空间信息
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 动态计算卷积后的维度
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        # 映射到特征维度 (默认 256)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# 提取掩码的回调函数
def mask_fn(env: gym.Env):
    return env.unwrapped.valid_action_mask(player_id=0)

def train_cnn_ai():
    print("=== 初始化带 CNN 的训练环境 ===")
    base_env = GeneralsRLGymEnv(width=8, height=8, max_steps=400)
    env = ActionMasker(base_env, mask_fn)

    # 告诉 PPO 使用我们自定义的 CNN 提取器
    policy_kwargs = dict(
        features_extractor_class=GeneralsCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    print("=== 开始训练 CNN 模型 (这回我们要练 30 万步！) ===")
    model = MaskablePPO(
        "MlpPolicy", # 依然用 MlpPolicy，但底层特征提取会被替换为我们的 CNN
        env, 
        policy_kwargs=policy_kwargs, # 注入灵魂
        verbose=1, 
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=256, # 加入批处理大小，稳定训练
    )
    
    # 增加到 30 万步，给 CNN 足够的时间收敛（大约需要 3-10 分钟）
    model.learn(total_timesteps=300000)
    
    model.save("generals_ppo_cnn_v2")
    print("=== CNN 模型已保存为 generals_ppo_cnn_v2.zip ===")

def evaluate_cnn_ai():
    print("\n=== 开始观看 CNN-AI 表演 ===")
    base_env = GeneralsRLGymEnv(width=8, height=8, max_steps=400)
    env = ActionMasker(base_env, mask_fn)
    
    model = MaskablePPO.load("generals_ppo_cnn_v2")
    
    obs, info = env.reset()
    env.render()
    
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        action_masks = mask_fn(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        time.sleep(0.1)

    if base_env.winner == 0:
        print("\n🏆 CNN 人工智能 (红方) 以绝对优势斩首获胜！")
    elif base_env.winner == 1:
        print("\n💀 对手 (蓝方) 赢了！")
    else:
        print("\n🤝 还是平局！但注意看它是否有战术走位。")

if __name__ == "__main__":
    train_cnn_ai()
    evaluate_cnn_ai()
