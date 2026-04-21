import torch
import torch.nn as nn

# Implementation of ResNet18 below:

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

# Implementation of DenseNet121 below:

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.bn1 = nn.InstanceNorm3d(in_channels, affine=True)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(4 * growth_rate, affine=True)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat([x, out], dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate=32):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels

        for _ in range(num_layers):
            self.layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate

        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.InstanceNorm3d(in_channels, affine=True)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, num_classes=3, growth_rate=32):
        super().__init__()
        self.growth = growth_rate
        self.conv1 = nn.Conv3d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.block1 = DenseBlock(64, 6, growth_rate)
        self.trans1 = Transition(self.block1.out_channels, self.block1.out_channels // 2)
        self.block2 = DenseBlock(self.block1.out_channels // 2, 12, growth_rate)
        self.trans2 = Transition(self.block2.out_channels, self.block2.out_channels // 2)
        self.block3 = DenseBlock(self.block2.out_channels // 2, 24, growth_rate)
        self.trans3 = Transition(self.block3.out_channels, self.block3.out_channels // 2)
        self.block4 = DenseBlock(self.block3.out_channels // 2, 16, growth_rate)
        final_channels = self.block4.out_channels
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(final_channels, num_classes)
        self.relu = nn.ReLU()
        self.bn1 = nn.InstanceNorm3d(64,affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.trans3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x