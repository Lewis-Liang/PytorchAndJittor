import sys
sys.path.append(r"E:\code\PytorchAndJittor")

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.init as init
from get_data_from_numpy import *


def init_networks(networks):
    def init_weights(m, gain=0.02):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                data = get_numpy_weight("normal", m.weight.shape[2], m.weight.shape[3])
                weight = torch.cuda.FloatTensor(data[0].astype(np.float32))
                # 此处要使用clone()
                m.weight.data = weight.expand_as(m.weight.data).clone()
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            data = get_numpy_weight("xavier_normal", m.weight.shape[2], m.weight.shape[3])
            weight = torch.cuda.FloatTensor(data[0].astype(np.float32))
            # 此处要使用clone()
            m.weight.data = weight.expand_as(m.weight.data).clone()
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
    for net in networks:
        net.apply(init_weights)
        

class Net(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super().__init__()
        self.spnorm = spectral_norm
        self.conv1 = nn.Conv2d(in_features,mid_features,3,1,1)
        self.conv2 = self.spnorm(nn.Conv2d(mid_features,mid_features,3,1,1))
        self.conv3 = self.spnorm(nn.Conv2d(mid_features,out_features,3,1,1))
        self.bn = nn.BatchNorm2d(num_features=out_features, affine=False,track_running_stats=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        dx = self.conv1(x)
        dx = self.conv2(dx)
        dx = self.conv3(dx)
        dx = self.bn(dx)
        dx = self.relu(dx)
        return dx
   
    
if __name__ == "__main__":
    # fix seed
    np_fix_seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # args define
    in_features = 3
    mid_features = 6
    out_features = 9
    # model
    net = Net(in_features, mid_features, out_features)
    for n, c in net.named_children():
        # c是具体哪个层或模块对应的nn.Module对象，可以直接用c.weight？（待验证）
        # 如果net内包含sequential或ModuleList时，结果会是啥？
        print(f"{n}:\n{c}")
    net.cuda()
    # train mode
    net.train()
    # optim
    # lr=0.001太小了，基本看不到效果，所以调整为0.1
    optim = torch.optim.SGD(net.parameters(), lr=0.1)
    # loss
    loss_fn = torch.nn.MSELoss()
    init_networks([net])
    # for- and back- ward
    for i in range(5):
        # input
        x = torch.from_numpy(get_data(1,in_features,3,3)).cuda()
        # output
        y = torch.from_numpy(get_data(1,out_features,3,3)).cuda()
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
            print(f"{n}:\n{p.detach().cpu()[0,0]}")
            if p.grad  is None:
                print(f"{n}.grad:\nNone")
            else:
                print(f"{n}.grad:\n{p.grad.detach().cpu()[0,0]}")
        
        # after backward and step
        net.zero_grad()
        loss.backward()
        optim.step()
        print(f"{'*'*20}after backward and step{'*'*20}")
        for n,p in net.named_parameters():
            # 只关心conv权重的梯度变化
            if n.find("conv") == -1 or n.find("bias") != -1:
                continue
            print(f"{n}:\n{p.detach().cpu()[0,0]}")
            if p.grad  is None:
                print(f"{n}.grad:\nNone")
            else:
                print(f"{n}.grad:\n{p.grad.detach().cpu()[0,0]}")
        print("one iter end")
    
    