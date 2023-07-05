



import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch import nn
from . import Residual



class Squeeze(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, r: int, bias: bool = True,
                 device=None, dtype=None, Types=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Squeeze, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_features, out_features//r),
            nn.ReLU(inplace=True),
            nn.Linear(out_features//r, out_features),
            nn.Sigmoid()
            )
        self.first = nn.Sequential(
            Residual.ResidualB(in_features,out_features),
            )
        
        self.second = nn.Sequential(
            nn.Conv2d(in_features,out_features,kernel_size=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.first(inputs)
        y1 = torch.unsqueeze(torch.unsqueeze(self.feature_extractor(x),2),3)
        y4 = self.second(inputs)
        y2 = y1*x
        y3 = y2+y4
        return y3
