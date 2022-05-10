class BackwardContext:
    def __init__(self):
        pass

    @staticmethod
    def execute(base_fn, g):
        order = []
        base_fn.topsort(order)
        base_fn.add_grad(g)

        for fn in order[::-1]:
            fn.propagate()


class BaseBackwardFn:
    def __init__(self):
        self.g = 0.
        self.visited = False

    def topsort(self, lis):
        self.visited = True
        lis.append(self)

    def add_grad(self, g):
        self.g += g

    def propagate(self):
        pass

    def zero_grad(self):
        self.g = 0.0


class AccumulateGrad(BaseBackwardFn):
    def __init__(self):
        super().__init__()


class BaseBinaryBackwardFn(BaseBackwardFn):
    def __init__(self, v, u):
        super().__init__()
        self.v = v
        self.u = u

    def topsort(self, lis):
        self.visited = True
        if self.v.require_grad and not self.v.backward_fn.visited:
            self.v.backward_fn.topsort(lis)
        if self.u.require_grad and not self.u.backward_fn.visited:
            self.u.backward_fn.topsort(lis)
        lis.append(self)

    def zero_grad(self):
        super().zero_grad()
        self.v.zero_grad()
        self.u.zero_grad()


class MultiplyBackwardFn(BaseBinaryBackwardFn):
    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.u.value * self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(self.v.value * self.g)


class AddBackwardFn(BaseBinaryBackwardFn):
    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(self.g)


class SubtractBackwardFn(BaseBinaryBackwardFn):
    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(-self.g)


class DivideBackwardFn(BaseBinaryBackwardFn):
    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(1 / self.u.value * self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(- self.v.value / (self.u.value * self.u.value) * self.g)


class BaseUnaryBackwardFn(BaseBackwardFn):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def topsort(self, lis):
        self.visited = True
        if self.v.require_grad and not self.v.backward_fn.visited:
            self.v.backward_fn.topsort(lis)
        lis.append(self)

    def zero_grad(self):
        self.v.zero_grad()


class ExpBackwardFn(BaseUnaryBackwardFn):
    def __init__(self, v, v_exp):
        super().__init__(v)
        self.v_exp = v_exp

    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.v_exp * self.g)
