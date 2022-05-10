import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from line_profiler_pycharm import profile
from tqdm.auto import tqdm

from tensor import Tensor

from nn import Module, LinearLayer, LeakyReLu, SGD, mse_loss

ds = torchvision.datasets.MNIST('./mnist', train=True, download=True,
                                transform=
                                T.Compose([
                                    T.ToTensor()
                                ]))

dl = torch.utils.data.DataLoader(ds,
                                 batch_size=2,
                                 shuffle=True)


class Net(Module):
    def __init__(self):
        super().__init__()

        self.l1 = LinearLayer(784, 32)
        self.a1 = LeakyReLu()
        self.l2 = LinearLayer(32, 32)
        self.a2 = LeakyReLu()
        self.l3 = LinearLayer(32, 10)
        self.a3 = LeakyReLu()

        self.register_module(self.l1)
        self.register_module(self.a1)
        self.register_module(self.l2)
        self.register_module(self.a2)
        self.register_module(self.l3)
        self.register_module(self.a3)

    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.a3(self.l3(x))
        return x


@profile
def main():
    net = Net()
    optim = SGD(net, learning_rate=0.01)

    losses = []
    i = 0
    tbar = tqdm(dl)
    for x, y in tbar:
        x = Tensor(x.flatten(start_dim=1).tolist(), require_grad=False)
        y = Tensor(F.one_hot(y, num_classes=10).tolist(), require_grad=False)
        y_ = net(x)
        s = y_.sum().detach(disable_grad=True)
        # y_ = y_ / s
        loss = mse_loss(y_, y)
        losses.append(loss.item())
        tbar.set_description(f'loss={loss}')
        loss.backward()
        optim.step()

        i += 1
        if i == 3:
            break


if __name__ == '__main__':
    main()
