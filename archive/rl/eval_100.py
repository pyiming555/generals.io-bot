import time
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from tqdm import tqdm # 用于显示进度条

# 导入环境和 CNN 架构
from generals_gym import GeneralsRLGymEnv
from train_cnn import GeneralsCNN, mask_fn

def evaluate_100_games():
    print("=== 加载环境与 CNN 模型 ===")
    base_env = GeneralsRLGymEnv(width=8, height=8, max_steps=400)
    env = ActionMasker(base_env, mask_fn)
    
    # 加载刚刚训练的 30 万步 CNN 模型
    model = MaskablePPO.load("generals_ppo_cnn_v2")
    
    wins = 0
    losses = 0
    draws = 0
    
    print("=== 开始 100 局盲测 ===")
    start_time = time.time()
    
    # 跑 100 局
    for i in tqdm(range(100)):
        obs, info = env.reset()
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action_masks = mask_fn(env)
            # deterministic=True 表示使用网络输出的最大概率动作（最强形态）
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
        if base_env.winner == 0:
            wins += 1
        elif base_env.winner == 1:
            losses += 1
        else:
            draws += 1

    print("\n" + "="*30)
    print("📊 100 局测试最终结果 📊")
    print("="*30)
    print(f"🔴 AI 胜利 (斩首): {wins} 局")
    print(f"🔵 随机对手胜利  : {losses} 局")
    print(f"🤝 平局 (400步)  : {draws} 局")
    print(f"⏱️ 总耗时: {time.time() - start_time:.2f} 秒")
    print("="*30)
    
    if wins > 80:
        print("💡 评价：AI 已经绝对碾压了随机对手，可以开启 Self-Play 了！")
    elif wins > 50:
        print("💡 评价：AI 略占优势，但偶尔会失误被偷家。")
    else:
        print("💡 评价：AI 还需要更多步数的训练，或者需要调整奖励函数以增加防御意识。")

if __name__ == "__main__":
    evaluate_100_games()
