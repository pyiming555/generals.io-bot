"""Generate V4 training curves"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('v4_tiebreaker.log') as f:
    lines = f.readlines()

data = []
current = {}

for line in lines:
    for key, pat in [
        ('ep_rew_mean', r'\|\s+ep_rew_mean\s+\|\s+([-\d.]+)\s+\|'),
        ('total_timesteps', r'\|\s+total_timesteps\s+\|\s+(\d+)\s+\|'),
        ('entropy_loss', r'\|\s+entropy_loss\s+\|\s+([-\d.]+)\s+\|'),
        ('value_loss', r'\|\s+value_loss\s+\|\s+([\d.]+)\s+\|'),
        ('clip_fraction', r'\|\s+clip_fraction\s+\|\s+([\d.]+)\s+\|'),
    ]:
        m = re.search(pat, line)
        if m: current[key] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))

    stripped = line.strip()
    if stripped and all(c in '- |' for c in stripped) and 'total_timesteps' in current:
        data.append(dict(current))
        current = {}

steps = [d['total_timesteps'] for d in data]
rews = [d['ep_rew_mean'] for d in data]
ents = [d.get('entropy_loss', 0) for d in data]
clips = [d.get('clip_fraction', 0) for d in data]
vfs = [d.get('value_loss', 0) for d in data]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('V4 Tiebreaker Self-Play — Training Dashboard (500K steps)',
             fontsize=16, fontweight='bold', color='white', y=0.98)

def setup_ax(ax, title):
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='gray')
    ax.grid(alpha=0.15, color='#333')
    ax.set_xlim(min(steps), max(steps))

# 1) Reward with highlighted phases
ax = axes[0, 0]; setup_ax(ax, 'Episode Reward (ep_rew_mean)')
ax.plot(steps, rews, color='#e94560', linewidth=1.5, alpha=0.8)
ax.scatter(steps[::3], rews[::3], color='#e94560', s=12, alpha=0.3, zorder=5)
ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
ax.fill_between(steps, rews, alpha=0.08, color='#e94560')
ax.axhline(y=np.mean(rews[20:]), color='#f5a623', linestyle=':', alpha=0.6, label=f'Mean (post-warmup): {np.mean(rews[20:]):.2f}')
ax.set_ylabel('Reward', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')
ax.legend(fontsize=9)

# 2) Entropy
ax = axes[0, 1]; setup_ax(ax, 'Entropy (entropy_loss)')
ax.plot(steps, ents, color='#7bc043', linewidth=1.5, alpha=0.8)
ax.scatter(steps[::3], ents[::3], color='#7bc043', s=12, alpha=0.3, zorder=5)
ax.axhline(y=-4.5, color='#e94560', linestyle=':', alpha=0.4)
ax.fill_between(steps, ents, alpha=0.08, color='#7bc043')
ax.set_ylabel('Entropy', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')

# 3) Clip fraction
ax = axes[1, 0]; setup_ax(ax, 'Clip Fraction (policy convergence)')
ax.plot(steps, clips, color='#f5a623', linewidth=1.5, alpha=0.8)
ax.scatter(steps[::3], clips[::3], color='#f5a623', s=12, alpha=0.3, zorder=5)
ax.axhline(y=0.3, color='#e94560', linestyle=':', alpha=0.4, label='Danger (0.3)')
ax.axhline(y=0.1, color='#7bc043', linestyle=':', alpha=0.4, label='Safe (0.1)')
ax.set_ylabel('clip_fraction', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')
ax.set_ylim(0, 0.35)
ax.legend(fontsize=9)

# 4) Value Loss
ax = axes[1, 1]; setup_ax(ax, 'Value Loss')
ax.plot(steps, vfs, color='#533483', linewidth=1.5, alpha=0.8)
ax.scatter(steps[::3], vfs[::3], color='#533483', s=12, alpha=0.3, zorder=5)
ax.axhline(y=1.0, color='white', linestyle=':', alpha=0.4)
ax.fill_between(steps, vfs, alpha=0.08, color='#533483')
ax.set_ylabel('Value Loss', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = "v4_training_curves.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f'✅ Chart saved: {chart_path}')
