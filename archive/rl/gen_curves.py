"""
Generate V3 training curves from log data
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read log
with open("v3_selfplay_resume.log") as f:
    lines = f.readlines()

data = []
current = {}

for line in lines:
    m = re.search(r'\|\s+ep_rew_mean\s+\|\s+([-\d.]+)\s+\|', line)
    if m: current['ep_rew_mean'] = float(m.group(1))
    m = re.search(r'\|\s+ep_len_mean\s+\|\s+([\d.]+)\s+\|', line)
    if m: current['ep_len_mean'] = float(m.group(1))
    m = re.search(r'\|\s+total_timesteps\s+\|\s+(\d+)\s+\|', line)
    if m: current['total_timesteps'] = int(m.group(1))
    m = re.search(r'\|\s+entropy_loss\s+\|\s+([-\d.]+)\s+\|', line)
    if m: current['entropy_loss'] = float(m.group(1))
    m = re.search(r'\|\s+value_loss\s+\|\s+([\d.]+)\s+\|', line)
    if m: current['value_loss'] = float(m.group(1))
    m = re.search(r'\|\s+approx_kl\s+\|\s+([\d.]+)\s+\|', line)
    if m: current['approx_kl'] = float(m.group(1))
    m = re.search(r'\|\s+clip_fraction\s+\|\s+([\d.]+)\s+\|', line)
    if m: current['clip_fraction'] = float(m.group(1))
    m = re.search(r'\|\s+policy_gradient_loss\s+\|\s+([-\d.]+)\s+\|', line)
    if m: current['policy_gradient_loss'] = float(m.group(1))
    m = re.search(r'\|\s+explained_variance\s+\|\s+([-\d.]+)\s+\|', line)
    if m: current['explained_variance'] = float(m.group(1))
    m = re.search(r'\|\s+fps\s+\|\s+(\d+)\s+\|', line)
    if m: current['fps'] = int(m.group(1))

    stripped = line.strip()
    if stripped and all(c in '- |' for c in stripped):
        if 'total_timesteps' in current:
            data.append(dict(current))
            current = {}

steps = [d['total_timesteps'] for d in data]
rews = [d.get('ep_rew_mean', 0) for d in data]
entropies = [d.get('entropy_loss', 0) for d in data]
kls = [d.get('approx_kl', 0) for d in data]
clips = [d.get('clip_fraction', 0) for d in data]
vfs = [d.get('value_loss', 0) for d in data]
fps_list = [d.get('fps', 0) for d in data]
pgls = [d.get('policy_gradient_loss', 0) for d in data]

# Create chart
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('V3 Self-Play Phase 2 — Training Dashboard (778,240 → 1,001,472 steps)',
             fontsize=16, fontweight='bold', color='white', y=0.98)

c1, c2, c3, c4, c5, c6 = '#e94560', '#0f3460', '#16213e', '#533483', '#f5a623', '#7bc043'

def setup_ax(ax, title, color):
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='gray')
    ax.grid(alpha=0.15, color='#333')
    ax.set_xlim(min(steps), max(steps))

# 1) Reward
ax = axes[0, 0]; setup_ax(ax, 'Episode Reward (ep_rew_mean)', c1)
ax.plot(steps, rews, color=c1, linewidth=1.5, alpha=0.8)
ax.scatter(steps[::5], rews[::5], color=c1, s=20, alpha=0.4, zorder=5)
ax.axhline(y=np.mean(rews), color=c5, linestyle='--', alpha=0.6, label=f'Mean: {np.mean(rews):.2f}')
ax.fill_between(steps, rews, alpha=0.08, color=c1)
ax.set_ylabel('Reward', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')
ax.legend(fontsize=9, loc='lower right')

# 2) Entropy
ax = axes[0, 1]; setup_ax(ax, 'Entropy (entropy_loss)', c6)
ax.plot(steps, entropies, color=c6, linewidth=1.5, alpha=0.8)
ax.scatter(steps[::5], entropies[::5], color=c6, s=20, alpha=0.4, zorder=5)
ax.axhline(y=-4.5, color=c1, linestyle=':', alpha=0.5, label='Warning (-4.5)')
ax.fill_between(steps, entropies, alpha=0.08, color=c6)
ax.set_ylabel('Entropy', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')
ax.legend(fontsize=9)

# 3) KL + Clip
ax = axes[1, 0]; setup_ax(ax, 'Policy Stability (KL + Clip)', c1)
ax.plot(steps, kls, color='#45b7d1', linewidth=1.5, alpha=0.8, label='approx_kl')
ax.set_ylabel('approx_kl', color='#45b7d1')
ax2 = ax.twinx()
ax2.plot(steps, clips, color=c5, linewidth=1.5, alpha=0.8, label='clip_fraction')
ax2.axhline(y=0.3, color=c1, linestyle=':', alpha=0.5, label='Danger (0.3)')
ax2.set_ylabel('clip_fraction', color=c5)
ax2.tick_params(colors='gray')
ax2.spines['right'].set_color('#333')
l1, la1 = ax.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, fontsize=9, loc='upper right')

# 4) Value Loss
ax = axes[1, 1]; setup_ax(ax, 'Value Loss', c4)
ax.plot(steps, vfs, color=c4, linewidth=1.5, alpha=0.8)
ax.scatter(steps[::5], vfs[::5], color=c4, s=20, alpha=0.4, zorder=5)
ax.fill_between(steps, vfs, alpha=0.08, color=c4)
ax.axhline(y=np.mean(vfs), color=c6, linestyle=':', alpha=0.5, label=f'Mean: {np.mean(vfs):.3f}')
ax.set_ylabel('Value Loss', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')
ax.legend(fontsize=9)

# 5) FPS
ax = axes[2, 0]; setup_ax(ax, 'Training Speed (FPS)', c2)
ax.plot(steps, fps_list, color=c2, linewidth=1.5, alpha=0.8)
ax.fill_between(steps, fps_list, alpha=0.08, color=c2)
ax.axhline(y=np.mean(fps_list), color='white', linestyle='--', alpha=0.4, label=f'Mean: {np.mean(fps_list):.0f}')
ax.set_ylabel('FPS', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')
ax.legend(fontsize=9)

# 6) Policy Gradient Loss
ax = axes[2, 1]; setup_ax(ax, 'Policy Gradient Loss', '#45b7d1')
ax.plot(steps, pgls, color='#45b7d1', linewidth=1.5, alpha=0.8)
ax.scatter(steps[::5], pgls[::5], color='#45b7d1', s=20, alpha=0.4, zorder=5)
ax.axhline(y=np.mean(pgls), color='white', linestyle='--', alpha=0.4, label=f'Mean: {np.mean(pgls):.4f}')
ax.set_ylabel('Policy Loss', color='gray')
ax.set_xlabel('Total Timesteps', color='gray')
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = "v3_training_curves.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"✅ Chart saved: {chart_path}")

# Stats
print(f"\n=== Training Data ===")
print(f"Iterations parsed: {len(data)}")
print(f"Steps range: {data[0]['total_timesteps']:,} → {data[-1]['total_timesteps']:,}")
print(f"Reward range: {min(rews):.2f} → {max(rews):.2f}, mean={np.mean(rews):.2f}")
print(f"Entropy range: {min(entropies):.2f} → {max(entropies):.2f}, mean={np.mean(entropies):.2f}")
print(f"Clip fraction mean: {np.mean(clips):.3f}")
print(f"FPS mean: {np.mean(fps_list):.0f}")

# Simple numpy polyfit for slope
if len(steps) > 1:
    coeffs = np.polyfit(steps, rews, 1)
    slope = coeffs[0]
    # R² calculation
    y_pred = np.polyval(coeffs, steps)
    ss_res = np.sum((np.array(rews) - y_pred) ** 2)
    ss_tot = np.sum((np.array(rews) - np.mean(rews)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    print(f"Reward slope: {slope:.6f}, R²={r2:.4f}")
    print("↗️  Slight upward trend" if slope > 0 else "→  Plateaued")
