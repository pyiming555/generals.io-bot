import re

with open('v4_tiebreaker.log') as f:
    lines = f.readlines()

data = []
current = {}

for line in lines:
    m = re.search(r'\|\s+ep_rew_mean\s+\|\s+([-\d.]+)\s+\|', line)
    if m: current['ep_rew_mean'] = float(m.group(1))
    m = re.search(r'\|\s+total_timesteps\s+\|\s+(\d+)\s+\|', line)
    if m: current['total_timesteps'] = int(m.group(1))
    m = re.search(r'\|\s+entropy_loss\s+\|\s+([-\d.]+)\s+\|', line)
    if m: current['entropy_loss'] = float(m.group(1))
    m = re.search(r'\|\s+value_loss\s+\|\s+([\d.]+)\s+\|', line)
    if m: current['value_loss'] = float(m.group(1))
    m = re.search(r'\|\s+clip_fraction\s+\|\s+([\d.]+)\s+\|', line)
    if m: current['clip_fraction'] = float(m.group(1))

    stripped = line.strip()
    if stripped and all(c in '- |' for c in stripped) and 'total_timesteps' in current:
        data.append(dict(current))
        current = {}

print(f'Total iterations: {len(data)}')
if data:
    print(f'Steps: {data[0]["total_timesteps"]:,} -> {data[-1]["total_timesteps"]:,}')
    print()
    print('=== Reward Trend ===')
    for i, d in enumerate(data):
        if i % 15 == 0 or i == len(data)-1 or abs(d['ep_rew_mean']) < 3:
            print(f'  step={d["total_timesteps"]:>7d} | rew={d["ep_rew_mean"]:>+6.2f} | ent={d.get("entropy_loss",0):.2f} | clip={d.get("clip_fraction",0):.4f} | vf={d.get("value_loss",0):.3f}')

    max_rew = max(data, key=lambda d: d['ep_rew_mean'])
    print(f'\n=== Key Stats ===')
    print(f'Start reward: {data[0]["ep_rew_mean"]:+.2f}')
    print(f'Max reward:   {max_rew["ep_rew_mean"]:+.2f} at step {max_rew["total_timesteps"]:,}')
    print(f'Final reward: {data[-1]["ep_rew_mean"]:+.2f}')
    print(f'Final entropy: {data[-1].get("entropy_loss",0):.2f}')
    print(f'Final clip: {data[-1].get("clip_fraction",0):.4f}')
    print(f'Final value_loss: {data[-1].get("value_loss",0):.3f}')
