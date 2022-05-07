from util import rand_tensor


class Module:
    def __init__(self):
        self.params = []
        self.submodules = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update_hooks(self):
        pass

    def register_module(self, m):
        self.submodules.append(m)

    def register_param(self, name):
        self.params.append(name)

    def forward(self, x):
        return x


class LinearLayer(Module):
    def __init__(self, input_f, output_f):
        super().__init__()

        self.weights = rand_tensor((input_f, output_f), require_grad=True)
        self.bias = rand_tensor((output_f,), require_grad=True)

        self.register_param('weights')
        self.register_param('bias')

    def forward(self, x):
        x = x @ self.weights
        x = x.apply(lambda d: d + self.bias, dim=-2)
        return x


class LeakyReLu(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def apply(self, v):
        return v if v.value > 0 else v * self.alpha

    def forward(self, x):
        return x.apply(self.apply)


class Optimizer:
    def __init__(self, module):
        self.main_module = module

    def optimize(self, p):
        pass

    def search(self, module):
        for param_name in module.params:
            p = getattr(module, param_name)
            setattr(module, param_name, self.optimize(p))

        for m in module.submodules:
            self.search(m)

    def step(self):
        self.search(self.main_module)


class SGD(Optimizer):
    def __init__(self, module, learning_rate=0.01):
        super().__init__(module)
        self.learning_rate = learning_rate

    def optimize(self, p):
        return (p - p.grad_tensor() * self.learning_rate).detach()


def mse_loss(predictions, target):
    e = (predictions - target)
    e2 = e * e
    return e2.sum() * (1 / e2.numel())


def softmax(p, dim=-1):
    e = p.exp()
    div = e.apply(lambda t: t.sum(), dim=dim + (-1 if dim < 0 else +1))
    return e / div
