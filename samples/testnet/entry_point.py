import torch
import torch.nn as nn

import testnet1


class TestNetWithLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.testnet = testnet1.TestNet()

    def forward(self, input):
        return self.testnet(input).sum()


def skyline_model_provider():
    return TestNetWithLoss().cuda()


def skyline_input_provider(batch_size=32):
    return (torch.randn((batch_size, 3, 128, 128)).cuda(),)


def skyline_iteration_provider(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out.backward()
        optimizer.step()
    return iteration
