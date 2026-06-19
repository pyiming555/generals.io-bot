import ctypes, time, os, sys, numpy as np

_PROJECT = "/media/pyiming/C22AA0E82AA0DB25/project/generals.io/src/python"
os.chdir(_PROJECT)

lib = ctypes.cdll.LoadLibrary(os.path.join(_PROJECT, '..', 'cpp', 'libgenerals_nn.so'))

lib.generals_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.generals_create.restype = ctypes.c_void_p
lib.generals_destroy.argtypes = [ctypes.c_void_p]; lib.generals_destroy.restype = None
lib.generals_get_winner.argtypes = [ctypes.c_void_p]; lib.generals_get_winner.restype = ctypes.c_int
lib.generals_step_dual.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.generals_step_dual.restype = ctypes.c_int
lib.belief_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.belief_create.restype = ctypes.c_void_p
lib.belief_destroy.argtypes = [ctypes.c_void_p]; lib.belief_destroy.restype = None
lib.belief_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.belief_observe.restype = None
lib.mcts_create.argtypes = [ctypes.c_uint]; lib.mcts_create.restype = ctypes.c_void_p
lib.mcts_destroy.argtypes = [ctypes.c_void_p]; lib.mcts_destroy.restype = None
lib.mcts_search_nn_auto.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.mcts_search_nn_auto.restype = ctypes.c_int
lib.nn_create.argtypes = [ctypes.c_char_p]; lib.nn_create.restype = ctypes.c_void_p
lib.nn_destroy.argtypes = [ctypes.c_void_p]; lib.nn_destroy.restype = None

W, H, MAX_S, DET = 12, 12, 300, 4
model_path = os.path.join(_PROJECT, 'rl_models', 'policy_value_v3.ptl')
nn = lib.nn_create(model_path.encode())
print(f"✓ 模型加载: {model_path}")

def run(n_a, n_b, ng, label):
    wa, wb, dr = 0, 0, 0
    ta, tb, ca, cb = 0.0, 0.0, 0, 0
    for sw in [0, 1]:
        for g in range(ng):
            seed = 42 + g * 13 + sw * 777
            gs = lib.generals_create(W, H, MAX_S, seed)
            b0 = lib.belief_create(W, H, 0); b1 = lib.belief_create(W, H, 1)
            m0 = lib.mcts_create(seed); m1 = lib.mcts_create(seed + 999)
            st, win = 0, -1
            while st < MAX_S and win == -1:
                if sw == 0:  # A红 B蓝
                    lib.belief_observe(b0, gs, st)
                    t0 = time.perf_counter(); a0 = lib.mcts_search_nn_auto(m0, b0, 0, DET, n_a, nn); t1 = time.perf_counter()
                    ta += (t1-t0); ca += 1
                    lib.belief_observe(b1, gs, st)
                    t0 = time.perf_counter(); a1 = lib.mcts_search_nn_auto(m1, b1, 1, DET, n_b, nn); t1 = time.perf_counter()
                    tb += (t1-t0); cb += 1
                else:  # B红 A蓝
                    lib.belief_observe(b0, gs, st)
                    t0 = time.perf_counter(); a0 = lib.mcts_search_nn_auto(m0, b0, 0, DET, n_b, nn); t1 = time.perf_counter()
                    tb += (t1-t0); cb += 1
                    lib.belief_observe(b1, gs, st)
                    t0 = time.perf_counter(); a1 = lib.mcts_search_nn_auto(m1, b1, 1, DET, n_a, nn); t1 = time.perf_counter()
                    ta += (t1-t0); ca += 1
                win = lib.generals_step_dual(gs, a0, a1)
                st += 1
            if (sw == 0 and win == 0) or (sw == 1 and win == 1): wa += 1
            elif (sw == 0 and win == 1) or (sw == 1 and win == 0): wb += 1
            elif win == 2: dr += 1
            lib.belief_destroy(b0); lib.belief_destroy(b1); lib.mcts_destroy(m0); lib.mcts_destroy(m1); lib.generals_destroy(gs)
    total = ng * 2
    print(f"\n  {label} ({total}局)")
    print(f"  n={n_a}: {wa}胜 ({wa/total*100:.0f}%)  {ta/ca*1000:.2f}ms/步 (共{ca}次)")
    print(f"  n={n_b}: {wb}胜 ({wb/total*100:.0f}%)  {tb/cb*1000:.2f}ms/步 (共{cb}次)")

run(260, 200, 25, "NN+MCTS(n=260) vs NN+MCTS(n=200)")
lib.nn_destroy(nn)
print("\n✅ 完成!")
