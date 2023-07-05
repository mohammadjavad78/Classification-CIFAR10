from torch import Tensor
from torch.nn.modules.module import Module
from torch import nn


class ResnextB(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, g:int, b1:int, b2:int, bias: bool = True,
                 device=None, dtype=None, Types=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ResnextB, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.feature_extractor1 = nn.Sequential(
                nn.Conv2d(in_features, b1, kernel_size=1),
                nn.BatchNorm2d(b1),
                nn.ReLU(inplace=True),
                )
        self.feature_extractor3 = nn.Sequential(
                nn.Conv2d(b2, out_features, kernel_size=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
                )
        self.feature_extractor2=nn.Sequential(
                nn.Conv2d(b1, b2, kernel_size=3, padding=1,groups=g),
                nn.BatchNorm2d(b2),
                nn.ReLU(inplace=True),
                )
        
        
        self.feature_extractor4=nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
                )




    def forward(self, inputs: Tensor) -> Tensor:
        y1 = self.feature_extractor1(inputs)
        y2=self.feature_extractor2(y1)
        y4=self.feature_extractor3(y2)
        y6=self.feature_extractor4(inputs)
        y5= y4+y6
        return y5
    
