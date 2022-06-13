from math import inf
from multiprocessing.connection import answer_challenge
import jittor as jt
import numpy as np
import get_data_from_numpy

jt.set_global_seed(42)
print(jt.normal(0,1,(3,3)).reshape(-1))

# target = jt.Var([ 0.3367,  0.1288,  0.2345,  0.2303, -1.1229, -0.1863,  2.2082, -0.6380,
#          0.4617]).reshape((3,3))

# seed = 0
# res = seed
# loss_fn = jt.nn.L1Loss()
# loss = jt.Var(inf)

# with jt.no_grad():
#     for seed in range(0,100000007):
#         jt.set_global_seed(seed)
#         output = jt.normal(0,1,(3,3))
#         loss_ = loss_fn(output, target) * 10000
#         if loss_ < loss:
#             loss = loss_
#             res = seed
#             print(f"current seed: {seed}, min_loss: {loss:.4}")
# print(f"closest seed = {res}")