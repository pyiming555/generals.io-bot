import time
import gymnasium as gym
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# 导入你刚刚写的环境
from generals_gym import GeneralsRLGymEnv

def mask_fn(env: gym.Env):
    """
    这是一个回调函数，告诉 SB3 去哪里获取 Action Mask。
    我们需要将底层环境的掩码提取出来给 PPO 算法。
    """
    # 如果环境被其他 Wrapper 包装过，需要用 env.unwrapped 访问底层方法
    return env.unwrapped.valid_action_mask(player_id=0)

def train_ai():
    print("=== 初始化环境 ===")
    base_env = GeneralsRLGymEnv(width=8, height=8, max_steps=400)
    # 用 ActionMasker 包装环境
    env = ActionMasker(base_env, mask_fn)

    print("=== 开始训练模型 ===")
    # MlpPolicy 会自动将 6x8x8 的图像展平为 384 维向量，适合这种微型网格
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        gamma=0.99, # 折扣因子，0.99 表示 AI 会考虑较长远的未来
        n_steps=1024, # 每次收集 1024 步的数据更新一次网络
    )
    
    # 训练 10 万步（在普通电脑 CPU 上大约只需 1~3 分钟）
    model.learn(total_timesteps=100000)
    
    # 保存模型
    model.save("generals_ppo_v1")
    print("=== 模型已保存为 generals_ppo_v1.zip ===")

def evaluate_ai():
    print("\n=== 开始观看 AI 表演 ===")
    base_env = GeneralsRLGymEnv(width=8, height=8)
    env = ActionMasker(base_env, mask_fn)
    
    # 加载刚刚训练好的模型
    model = MaskablePPO.load("generals_ppo_v1")
    
    obs, info = env.reset()
    env.render()
    
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        # ⚠️ 注意：预测时必须把 action_masks 传给模型！
        # 否则模型仍可能输出非法动作
        action_masks = mask_fn(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每次行动后渲染画面，停顿 0.1 秒方便肉眼观察
        env.render()
        time.sleep(0.1)

    if base_env.winner == 0:
        print("\n🏆 训练后的人工智能 (红方) 获得了胜利！")
    elif base_env.winner == 1:
        print("\n💀 对手 (蓝方) 赢了！可能训练步数还不够。")
    else:
        print("\n🤝 平局！(到达最大步数)")

if __name__ == "__main__":
    # 1. 先执行训练（如果你已经有模型了，可以注释掉这行）
    train_ai()
    
    # 2. 训练完后，让 AI 和随机机器人在环境里打一局看看效果
    evaluate_ai()
