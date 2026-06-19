#!/bin/bash
# generals.io V3 curriculum training script (runs from shell, bypasses cron_mode)
cd /home/pyiming/project/generals.io
python3 train_v3_base.py 2>&1
echo "EXIT_CODE=$?"
