import torch
from torch.autograd.function import Function
from torch.nn import Module
from torch.nn import ModuleList
from torch.random import fork_rng
from torch.utils.checkpoint import get_device_states
from torch.utils.checkpoint import set_device_states


class DeterministicLayer(Module):
    def __init__(self, layer):
        super(DeterministicLayer, self).__init__()
        self.layer = layer
        self.cpu_state = None
        self.gpu_devices = None
        self.gpu_states = None

    def forward(self, x, backward=False):
        if self.training:
            self.cpu_state = torch.get_rng_state()
            self.gpu_devices, self.gpu_states = get_device_states(x)
        if backward:
            torch.set_rng_state(self.cpu_state)
            with fork_rng(devices=self.gpu_devices, enabled=True):
                set_device_states(self.gpu_devices, self.gpu_states)
                return self.layer(x)
        return self.layer(x)


class ReversibleLayer(Module):
    def __init__(self, layer_1, layer_2):
        super(ReversibleLayer, self).__init__()
        self.layer_1 = DeterministicLayer(layer_1)
        self.layer_2 = DeterministicLayer(layer_2)

    def forward(self, x):
        x1, x2 = (t.squeeze() for t in x.chunk(2))
        with torch.no_grad():
            y1 = x1 + self.layer_1(x2)
            y2 = x2 + self.layer_2(y1)
        return torch.stack([y1, y2])

    def backward(self, y, dy):
        y1, y2 = (t.squeeze() for t in y.chunk(2))
        dy1, dy2 = (t.squeeze() for t in dy.chunk(2))
        y1.requires_grad = True
        y2.requires_grad = True
        with torch.enable_grad():
            y2_no_res = self.layer_2(y1, backward=True)
            y2_no_res.backward(dy2, retain_graph=True)
        with torch.no_grad():
            x2 = y2 - y2_no_res
            dx1 = dy1 + y1.grad
            y1.grad = None
        x2.requires_grad = True
        with torch.enable_grad():
            y1_no_res = self.layer_1(x2, backward=True)
            y1_no_res.backward(dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - y1_no_res
            dx2 = dy2 + x2.grad
            x2.grad = None
        x = torch.stack([x1, x2])
        dx = torch.stack([dx1, dx2])
        del y, y1, y2, dy, dy1, dy2, y1_no_res, y2_no_res
        return x, dx


class ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, stack):
        for layer in stack:
            x = layer(x)
        ctx.y = x.detach()
        ctx.stack = stack
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        for layer in ctx.stack[::-1]:
            y, dy = layer.backward(y, dy)
        return dy, None


class ReversibleStack(Module):
    def __init__(self, stack):
        super(ReversibleStack, self).__init__()
        self.layers = ModuleList([ReversibleLayer(layer_1, layer_2) for layer_1, layer_2 in stack])

    def forward(self, x):
        out = ReversibleFunction.apply(x.expand(2, *x.shape), self.layers)
        return out.mean(dim=0)
