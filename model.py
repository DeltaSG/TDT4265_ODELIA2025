import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.bs1 = nn.InstanceNorm3d(in_channels,affine=True)
        self.bs2 = nn.InstanceNorm3d(out_channels,affine=True)
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=stride,bias=False,padding = 1)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.InstanceNorm3d(in_channels,affine=True),
                nn.ReLU(),
                nn.Conv3d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self,x):

        out = self.bs1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bs2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out
    
class ResNet18(nn.Module):
    def __init__(self,num_classes = 3):
        super().__init__()
        self.conv1 = nn.Conv3d(5,64,kernel_size=3,stride=2,padding=1,bias = False)
        self.bn1 = nn.InstanceNorm3d(64,affine=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        self.block1 = Block(64,64,1)
        self.block2 = Block(64,64,1)
        self.block3 = Block(64,128,2)
        self.block4 = Block(128,128,1)
        self.block5 = Block(128,256,2)
        self.block6 = Block(256,256,1)
        self.block7 = Block(256,512,2)
        self.block8 = Block(512,512,1)
        self.avrpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)

        out = self.avrpool(out)
        out = out.view(out.size(0),-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out