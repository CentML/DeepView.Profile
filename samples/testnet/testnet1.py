import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(num_features=1024)

        self.linear = nn.Linear(in_features=4096, out_features=1000)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, input):
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
        output = self.max_pool(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output
