import torch
import torch.nn as nn


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=512, out_features=10)

    def forward(self, input):
        """
        VGG-11 for CIFAR-10
        @innpv size (32, 3, 32, 32)
        """
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu(output)
        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu(output)
        output = self.conv6(output)
        output = self.bn6(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv7(output)
        output = self.bn7(output)
        output = self.relu(output)
        output = self.conv8(output)
        output = self.bn8(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = output.view(-1, 512)
        output = self.linear(output)

        return output
