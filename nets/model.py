from nets import Residual, Inception, Resnext,Squeeze

import torch.nn as nn





class ModelCNN(nn.Module):
    def __init__(self,model_name):
        super().__init__()

        if(model_name=="Residual"):
            self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Residual.ResidualA(128,128),
            Residual.ResidualB(128,256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Residual.ResidualA(512,512),
            nn.AdaptiveAvgPool2d(1),
            ) 
        elif(model_name=="Inception"):
            self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Inception.Inception(128,128),
            Inception.Inception(128,256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Inception.Inception(512,512),
            nn.AdaptiveAvgPool2d(1),
            )
        
        elif(model_name=="Resnext"):
            self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Resnext.ResnextB(128,128,64,64,64),
            Resnext.ResnextB(128,256,64,64,64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Resnext.ResnextB(512,512,64,64,64),
            nn.AdaptiveAvgPool2d(1),
            )
        

        elif(model_name=="Squeeze"):
            self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Squeeze.Squeeze(128,128,1),
            Squeeze.Squeeze(128,256,1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Squeeze.Squeeze(512,512,1),
            nn.AdaptiveAvgPool2d(1),
            )
        

        else:
            self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Residual.ResidualA(128,128),
            Inception.Inception(128,256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Squeeze.Squeeze(512,512,1),
            nn.AdaptiveAvgPool2d(1),
            )        

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0],512)
        x = self.classifier(x)
        return x