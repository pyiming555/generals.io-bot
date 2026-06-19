"""
launch_v4.py — 启动 V4 绝杀平局自对弈训练

使用 subprocess.Popen(start_new_session=True) 确保不会被 SIGTERM 杀死。
"""
import subprocess
import os
import time

log_path = os.path.abspath("v4_tiebreaker.log")
script_path = os.path.abspath("train_v4_tiebreaker.py")
workdir = os.path.abspath(".")

with open(log_path, "w") as f:
    f.write(f"=== V4 绝杀平局训练启动 {__import__('datetime').datetime.now()} ===\n")
    f.write(f"Python: /usr/bin/python3\n")
    f.write(f"Script: {script_path}\n")
    f.write(f"种子模型: generals_ppo_v3_master.zip\n")
    f.write(f"目标步数: 500,000\n")
    f.write(f"环境: GeneralsEnvV4TieBreaker (绝杀平局 + 纯净奖励)\n\n")

proc = subprocess.Popen(
    ["/usr/bin/python3", "-u", script_path],
    stdout=open(log_path, "a"),
    stderr=subprocess.STDOUT,
    cwd=workdir,
    start_new_session=True,
)

print(f"PID: {proc.pid}")
print(f"Log: {log_path}")

time.sleep(15)

poll = proc.poll()
if poll is None:
    r = subprocess.run(["tail", "-10", log_path], capture_output=True, text=True)
    print(f"\n✅ 进程 {proc.pid} 正在运行")
    print("--- 最新日志 ---")
    print(r.stdout)
elif poll == 0:
    print(f"⚠️ 已退出 (code 0)")
    r = subprocess.run(["tail", "-15", log_path], capture_output=True, text=True)
    print(r.stdout)
else:
    print(f"❌ 退出 code {poll}")
    r = subprocess.run(["tail", "-20", log_path], capture_output=True, text=True)
    print(r.stdout)
