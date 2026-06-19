"""
train_fcn.py — 监督学习训练 FCN

从脚本生成的数据中学习策略。
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from fcn_model import LightweightFCN, compute_loss


class ScriptDataset(Dataset):
    """脚本数据数据集"""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states = torch.from_numpy(data['states']).float()       # (N, 7, H, W)
        self.policies = torch.from_numpy(data['policies']).float()   # (N, H*W)
        self.actions = torch.from_numpy(data['actions']).long()      # (N,)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.actions[idx]


def train(resume_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # 路径
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _models_dir = os.path.join(_script_dir, '..', '..', 'models')
    _data_dir = os.path.join(_script_dir, '..', 'data_generation', 'training_data')
    os.makedirs(_models_dir, exist_ok=True)
    
    # 加载数据
    dataset = ScriptDataset(os.path.join(_data_dir, 'script_dataset.npz'))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    print(f'Dataset: {len(dataset)} samples, {len(loader)} batches')
    
    # 模型
    model = LightweightFCN(in_channels=7, base_filters=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    n_epochs = 20
    start_epoch = 0
    best_loss = float('inf')
    
    # === 恢复模式 ===
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f'↻ 从 {resume_from} 恢复 (epoch {start_epoch}, best_loss={best_loss:.4f})')
    
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Starting from epoch {start_epoch+1}/{n_epochs}')
    
    # 训练循环
    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for states, policies, actions in loader:
            states = states.to(device)
            policies = policies.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            policy_logits, action_logits = model(states)
            loss = compute_loss(policy_logits, action_logits, policies, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f}, lr={lr:.6f}')
        
        # 保存最佳模型 + 完整检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(_models_dir, 'checkpoint_full.pt'))
            # 同时保存旧格式的纯权重（用于推理）
            torch.save(model.state_dict(), os.path.join(_models_dir, 'fcn_script_model.pt'))
            print(f'  → 模型已保存 (best loss: {best_loss:.4f})')
    
    print(f'\n✅ 训练完成! 最佳模型: fcn_script_model.pt (loss={best_loss:.4f})')


if __name__ == '__main__':
    resume_path = sys.argv[1] if len(sys.argv) > 1 else None
    if resume_path and not os.path.isabs(resume_path):
        # 如果是相对路径，相对于 models/ 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resume_path = os.path.join(script_dir, '..', '..', 'models', resume_path)
    train(resume_from=resume_path)
