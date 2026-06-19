#!/bin/bash
cd /home/pyiming/project/generals.io
python3 -c '
import sb3_contrib
import gymnasium
import torch
print("dependencies OK")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("sb3_contrib:", sb3_contrib.__version__)
' 2>&1
