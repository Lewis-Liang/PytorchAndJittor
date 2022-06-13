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
        self.conv = nn.Conv2d(in_features,out_features,3,1,1,bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_features, affine=False, is_train=True)
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
    
    loss_fn = jt.nn.MSELoss()
    optim = jt.optim.SGD(net.parameters(), lr=0.1)
    
    for i in range(3):
        # input
        x = jt.Var(get_data(1,in_features,3,3))
        # output
        # 计图的MSELoss在计算时，不会检查output和target的维度是否匹配，而torch中会检查和警告
        # 例如此处out_features后面的两个参数应该为3,3
        # y = jt.Var(get_data(1,out_features,1,1))
        y = jt.Var(get_data(1,out_features,3,3))
        y_  = net(x)
        loss = loss_fn(y_, y)
        # before backward
        print(net.conv.weight[0,0].detach().data.ravel())
        optim.zero_grad()
        optim.backward(loss)
        optim.step()
        # after backward
        print(net.conv.weight[0,0].detach().data.ravel())
        print("one iter end")
        
        