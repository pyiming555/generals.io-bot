"""Export v3 model to TorchScript for C++ LibTorch inference"""
import torch, os, sys
sys.path.insert(0, '.')

from rl_pipeline import PolicyValueNet

net = PolicyValueNet()
net.load_state_dict(torch.load('rl_models/policy_value_v3.pt', map_location='cpu'))
net.eval()

example = torch.randn(1, 7, 12, 12)
traced = torch.jit.trace(net, example)
ts_path = 'rl_models/policy_value_v3.ptl'
traced.save(ts_path)
print(f"✅ v3 TorchScript: {ts_path} ({os.path.getsize(ts_path)/1024:.0f}KB)")

# Verify
ts = torch.jit.load(ts_path)
with torch.no_grad():
    p1, v1 = net(example)
    p2, v2 = ts(example)
print(f"  原始: v={v1.item():.6f}, p_sum={p1.sum():.4f}")
print(f"  TorchScript: v={v2.item():.6f}, p_sum={p2.sum():.4f}")
print(f"  一致: {'✅' if abs(v1.item()-v2.item())<1e-5 else '❌'}")
