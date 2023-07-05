from torch import Tensor
from torch.nn.modules.module import Module
from torch import nn

class ResidualA(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, Types=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ResidualA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_features),
            )


        self.rel = nn.Sequential(
            nn.ReLU(inplace=True),
        )




    def forward(self, inputs: Tensor) -> Tensor:
        y = self.feature_extractor(inputs)
        y2= y+inputs
        y3 = self.rel(y2)

        return y3
    

class ResidualB(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, Types=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ResidualB, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.feature_extractor1 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            )
        
        self.feature_extractor2 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1),
        )

        self.rel = nn.Sequential(
            nn.ReLU(inplace=True),
        )



    def forward(self, inputs: Tensor) -> Tensor:
        y1 = self.feature_extractor1(inputs)
        y2 = self.feature_extractor2(inputs)
        y3= y2+y1
        y4 = self.rel(y3)

        return y4
    
