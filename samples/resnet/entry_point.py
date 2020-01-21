import torch
import torch.nn as nn

import resnet


class ResNetWithLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.testnet = resnet.resnet18()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        out = self.testnet(input)
        return self.loss(out, target)


def skyline_model_provider():
    return ResNetWithLoss().cuda()


def skyline_input_provider(batch_size=32):
    return (
        torch.randn((batch_size, 3, 224, 224)).cuda(),
        torch.randint(low=0, high=1000, size=(batch_size,)).cuda(),
    )


def skyline_iteration_provider(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out.backward()
        optimizer.step()
    return iteration
