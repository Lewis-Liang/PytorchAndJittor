import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.encoder.append(nn.Conv2d(3, 64, 3, 1, 1))
        self.encoder.append(nn.Conv2d(64, 128, 3, 1, 1))
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(torch.relu(x))
        x = torch.softmax(x, dim=1)
        return x
    
if __name__ == '__main__':
    model = TestNet()
    model.module = model
    try:
        # model.module = model会把module注册到model._modules中，后续遍历时会进入死循环
        for key, module in model._modules.items():
            mod_str = repr(module)
    except Exception as e:
        print(e.__class__.__name__, ":", e)
    