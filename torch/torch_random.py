import torch
import get_data_from_numpy

torch.manual_seed(42)
print(torch.normal(0,1,(3,3)).reshape(-1))