import sys
sys.path.append(r"E:\code\PytorchAndJittor")

import jittor as jt
import jittor.nn as nn
import jittor.init as init
from get_data_from_numpy import *


def init_networks(networks):
    def init_weights(m, gain=0.02):
        classname = m.__class__.__name__
        if isinstance(m, jt.nn.BatchNorm2d):
            if hasattr(m, 'weight') and isinstance(m.weight, jt.Var):
                data = get_numpy_weight("normal", m.weight.shape[2], m.weight.shape[3])
                weight = jt.Var(data[0].astype(np.float32))
                m.weight.data = weight.expand_as(m.weight.data)
            if hasattr(m, 'bias') and isinstance(m.weight, jt.Var):
                init.constant_(m.bias, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            data = get_numpy_weight("xavier_normal", m.weight.shape[2], m.weight.shape[3])
            weight = jt.Var(data[0].astype(np.float32))
            m.weight.data = weight.expand_as(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
    for net in networks:
        net.apply(init_weights)
        

class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features,out_features,3,1,1)
        self.bn = nn.BatchNorm2d(num_features=out_features, affine=False)
        self.relu = nn.ReLU()
        
    def execute(self, x):
        dx = self.conv(x)
        dx = self.bn(dx)
        dx = self.relu(dx)
        return dx
   
    
if __name__ == "__main__":
    # fix seed
    np_fix_seed()
    jt.set_global_seed(0)
    # args define
    in_features = 3
    out_features = 6
    net = Net(in_features, out_features)
    init_networks([net])
    net.train()
    jt.use_cuda = 1
    jt.flags.lazy_execution = 0
    for i in range(5):
        print(net.bn.running_mean, net.bn.running_var, net.bn.is_train)
        # input
        x = jt.Var(get_data(1,in_features,3,3))
        # output
        y = net(x)
        print(x.max(), x.min())
        print(y.max(), y.min())
        print("*"*20)
    