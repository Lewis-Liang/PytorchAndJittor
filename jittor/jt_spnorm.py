import jittor as jt
import jittor.nn as nn
import jittor.init as init

from jt_spectral_norm import spectral_norm
from jt_convbn import init_networks
from get_data_from_numpy import *


class Net(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super().__init__()
        self.spnorm = spectral_norm
        self.conv1 = nn.Conv2d(in_features,mid_features,3,1,1)
        self.conv2 = self.spnorm(nn.Conv2d(mid_features,mid_features,3,1,1))
        self.conv3 = self.spnorm(nn.Conv2d(mid_features,out_features,3,1,1))
        self.bn = nn.BatchNorm2d(num_features=out_features, affine=False,is_train=True)
        self.relu = nn.ReLU()
        
    def execute(self, x):
        dx = self.conv1(x)
        dx = self.conv2(dx)
        dx = self.conv3(dx)
        dx = self.bn(dx)
        dx = self.relu(dx)
        return dx
   
    
if __name__ == "__main__":
    # fix seed
    np_fix_seed(0)
    jt.set_global_seed(0)
    jt.use_cuda = 1
    # args define
    in_features = 3
    mid_features = 6
    out_features = 9
    # model
    net = Net(in_features, mid_features, out_features)
    for n, c in net.named_modules():
        # c是具体哪个层或模块对应的nn.Module对象，可以直接用c.weight？（待验证）
        # 如果net内包含sequential或ModuleList时，结果会是啥？
        print(f"{n}:\n{c}")
    # train mode
    net.train()
    # optim
    # lr=0.001太小了，基本看不到效果，所以调整为0.1
    optim = jt.optim.SGD(net.parameters(), lr=0.1)
    # loss
    loss_fn = jt.nn.MSELoss()
    init_networks([net])
    # for- and back- ward
    for i in range(5):
        # input
        x = jt.Var(get_data(1,in_features,3,3))
        # output
        y = jt.Var(get_data(1,out_features,3,3))
        y_  = net(x)
        loss = loss_fn(y_, y)

        # before backward
        print(f"{'-'*20}iter:{i}{'-'*20}")
        print(f"{'*'*20}before backward{'*'*20}")
        
        for n,p in net.named_parameters():
            # 只关心conv的权重、梯度变化
            # 只关心conv权重的梯度变化
            if n.find("conv") == -1 or n.find("bias") != -1:
                continue
            print(f"{n}:\n{p.detach().data[0,0]}")
            if p.grad  is None:
                print(f"{n}.grad:\nNone")
            else:
                print(f"{n}.grad:\n{p.grad.detach().data[0,0]}")
        
        # after backward and step
        optim.zero_grad()
        optim.backward(loss)
        optim.step()
        print(f"{'*'*20}after backward and step{'*'*20}")
        for n,p in net.named_parameters():
            # 只关心conv权重的梯度变化
            if n.find("conv") == -1 or n.find("bias") != -1:
                continue
            print(f"{n}:\n{p.detach().data[0,0]}")
            if p.grad  is None:
                print(f"{n}.grad:\nNone")
            else:
                print(f"{n}.grad:\n{p.grad.detach().data[0,0]}")
        print("one iter end")
    
    