#!/bin/bash
# 启动 NN+MCTS vs 纯MCTS 锦标赛（后台可靠运行）
export PATH="/home/pyiming/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
cd /media/pyiming/C22AA0E82AA0DB25/project/generals.io/src/python
/usr/bin/python3 -u evaluate_nn_vs_pure.py --games 25 --model policy_value_v2 > nn_vs_pure_25games.log 2>&1
echo "DONE: $(date)" >> nn_vs_pure_25games.log
