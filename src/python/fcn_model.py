"""
fcn_model.py — 轻量全卷积网络

结构：
  5 层 Conv2D (3x3) + BN + ReLU → Policy Head + Action Head
  无全连接层 → 支持任意尺寸地图

输入:  (B, C, H, W)  C=7 特征通道
输出:
  policy_head: (B, 1, H, W) softmax 关注格概率图
  action_head: (B, 4, H, W) 每个格子的动作类型 [扩张=0, 集结=1, 进攻=2, 跳过=3]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两层 Conv + BN + ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class LightweightFCN(nn.Module):
    """
    轻量级全卷积网络
    
    参数:
      in_channels: 输入特征通道数 (默认7)
      base_filters: 基础卷积核数 (默认64)
    """
    
    def __init__(self, in_channels=7, base_filters=64):
        super().__init__()
        
        # 12x12 地图：只用 2 层池化 (12→6→3)
        self.n_layers = 2
        
        self.in_conv = DoubleConv(in_channels, base_filters)
        
        # 下采样
        ch = base_filters
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(self.n_layers):
            out_ch = min(ch * 2, 512)
            self.encoders.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(ch, out_ch),
            ))
            ch = out_ch
        
        # 上采样 (对称)
        for i in range(self.n_layers):
            down_ch = ch
            out_ch = ch // 2
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(down_ch, out_ch, 2, stride=2),
                DoubleConv(out_ch + out_ch, out_ch),
            ))
            ch = out_ch
        
        # Policy Head: 关注格概率图
        self.policy_head = nn.Sequential(
            nn.Conv2d(base_filters, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        # Action Head: 每格的动作类型
        self.action_head = nn.Sequential(
            nn.Conv2d(base_filters, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )

    def forward(self, x):
        skips = []
        x = self.in_conv(x)
        skips.append(x)
        
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        for i, decoder in enumerate(self.decoders):
            x = decoder[0](x)
            skip = skips[-(i+2)] if i < len(skips)-1 else None
            if skip is not None and skip.shape[2:] == x.shape[2:]:
                x = torch.cat([x, skip], dim=1)
            x = decoder[1](x)
        
        policy = self.policy_head(x)
        actions = self.action_head(x)
        
        return policy, actions


def compute_loss(policy_logits, actions_logits, policy_target, action_target):
    """
    计算训练损失
    
    参数:
      policy_logits: (B, 1, H, W)
      actions_logits: (B, 4, H, W)
      policy_target: (B, H*W) one-hot 目标关注格
      action_target: (B,) 目标动作类型 (0-3)
    
    返回:
      loss: 标量
    """
    B, _, H, W = policy_logits.shape
    
    # Policy loss: 关注格交叉熵
    policy_flat = policy_logits.view(B, -1)  # (B, H*W)
    policy_target_idx = policy_target.argmax(dim=1)  # (B,) — 关注格索引
    policy_loss = F.cross_entropy(policy_flat, policy_target_idx)
    
    # Action loss: 只在关注格上计算
    action_loss = 0.0
    for b in range(B):
        idx = policy_target_idx[b].item()
        focus_action = actions_logits[b, :, idx // W, idx % W]  # (4,)
        action_loss += F.cross_entropy(focus_action[None], action_target[b:b+1])
    action_loss = action_loss / B
    
    # 🌟 熵正则化: 鼓励策略分布多样化，防止崩溃到单格预测
    policy_prob = F.softmax(policy_flat, dim=1)
    entropy = -(policy_prob * torch.log(policy_prob + 1e-10)).sum(dim=1).mean()
    entropy_bonus = 0.01 * entropy  # 小权重熵奖励
    
    return policy_loss + action_loss - entropy_bonus


if __name__ == '__main__':
    model = LightweightFCN(in_channels=7, base_filters=64)
    x = torch.randn(2, 7, 12, 12)
    policy, actions = model(x)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Policy: {policy.shape}')
    print(f'Actions:{actions.shape}')
    print('✅ Model OK')
