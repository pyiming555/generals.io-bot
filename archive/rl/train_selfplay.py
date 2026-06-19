import os
import glob
import random
from typing import Callable
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# 导入新环境和我们之前写的 CNN
from generals_gym_selfplay import GeneralsSelfPlayEnv
from train_cnn import GeneralsCNN, mask_fn

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性学习率衰减：随着训练进度 progress_remaining 从 1.0 降到 0.0，
    学习率也会线性降低。
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class LeagueTrainingCallback(BaseCallback):
    """
    联赛训练机制（Self-Play）：
    每隔一定步数，将当前模型保存至 pool 文件夹中。
    然后让环境随机加载一个旧版本的自己作为接下来的对手。
    """
    def __init__(self, pool_dir="league_pool", update_freq=20480, verbose=1):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.update_freq = update_freq
        os.makedirs(self.pool_dir, exist_ok=True)
        
    def _on_step(self):
        # 只有在达到指定步数时才更新对手
        if self.num_timesteps % self.update_freq == 0:
            # 1. 把当前的自己保存到池子里
            model_path = os.path.join(self.pool_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(model_path)
            
            # 2. 从池子里找所有历史模型
            models = glob.glob(os.path.join(self.pool_dir, "*.zip"))
            
            # 为了防止灾难性遗忘，我们 80% 的概率跟过去随机的自己打，20%概率跟最新的自己打
            if random.random() < 0.2:
                chosen_opponent = model_path
            else:
                chosen_opponent = random.choice(models)
            
            if self.verbose > 0:
                print(f"\n[League Training] 步数 {self.num_timesteps}: 蓝方对手切换为 -> {os.path.basename(chosen_opponent)}")
            
            # 3. 将选中的对手装载进环境中
            self.training_env.env_method("load_opponent", chosen_opponent)
            
        return True

def start_self_play():
    print("=== 初始化 Self-Play 环境 ===")
    base_env = GeneralsSelfPlayEnv(width=8, height=8, max_steps=400)
    env = ActionMasker(base_env, mask_fn)
    
    # 建立一个文件夹用来装历代 AI
    pool_dir = "league_pool"
    os.makedirs(pool_dir, exist_ok=True)
    
    # 💡 核心：继承过去 50 万步的记忆！
    # 使用上一个 Self-Play 训练的模型作为起点
    starting_model = "generals_ppo_selfplay_final"
    
    print(f"=== 加载大师级基座模型 {starting_model}，注入 LR 衰减 ===")
    model = MaskablePPO.load(starting_model, env=env, custom_objects={
        "learning_rate": linear_schedule(3e-4)  # 注入衰减学习率
    })
    
    # 设置 Callback
    callback = LeagueTrainingCallback(pool_dir=pool_dir, update_freq=20480)

    print("=== 🚀 开始终极左右互搏！(100万步 + 混合奖励 + LR衰减) ===")
    model.learn(total_timesteps=1000000, callback=callback)
    
    model.save("generals_ppo_master_final")
    print("=== 大师级训练完成，模型已保存为 generals_ppo_master_final.zip ===")

if __name__ == "__main__":
    start_self_play()
