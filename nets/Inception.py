import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import nn
import math

class Inception(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, r:int=4, bias: bool = True,
                 device=None, dtype=None, Types=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Inception, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.path1 = nn.Sequential(
            nn.AvgPool2d(3,1,padding=1),
            nn.Conv2d(in_features, out_features//r, kernel_size=1),
            nn.BatchNorm2d(out_features//r),
            nn.ReLU()
            )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_features, out_features//r, kernel_size=1),
            nn.BatchNorm2d(out_features//r),
            nn.ReLU()
            )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_features//r,padding=1, kernel_size=3),
            nn.BatchNorm2d(out_features//r),
            nn.ReLU()
            )
        self.path4 = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, out_features//r, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_features//r),
            nn.ReLU()
            )






    def forward(self, inputs: Tensor) -> Tensor:
        y1 = self.path1(inputs)
        y2 = self.path2(inputs)
        y3 = self.path3(inputs)
        y4 = self.path4(inputs)

        return torch.cat([y1,y2,y3,y4],dim=1)
    