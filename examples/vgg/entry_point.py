import torch

import vgg


def deepview_model_provider():
    return vgg.vgg11().cuda()


def deepview_input_provider(batch_size=16):
    return (
        torch.randn((batch_size, 3, 224, 224)).cuda(),
        torch.randint(low=0, high=1000, size=(batch_size,)).cuda(),
    )


def deepview_iteration_provider(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out.backward()
        optimizer.step()
    return iteration
