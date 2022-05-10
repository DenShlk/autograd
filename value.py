import math

from autograd import AccumulateGrad, BackwardContext, ExpBackwardFn, MultiplyBackwardFn, AddBackwardFn, \
    SubtractBackwardFn, DivideBackwardFn


class Value:
    def __init__(self, value, require_grad=True, backward_fn=None):
        self.value = float(value)
        self.require_grad = require_grad

        if self.require_grad and backward_fn is None:
            self.backward_fn = AccumulateGrad()
        else:
            self.backward_fn = backward_fn

    def item(self):
        return self.value

    def zero_grad(self):
        if self.backward_fn:
            self.backward_fn.zero_grad()
        if self.require_grad:
            self.backward_fn = AccumulateGrad()
        else:
            self.backward_fn = None

    def backward(self):
        ctx = BackwardContext()
        ctx.execute(self.backward_fn, 1)

    def grad(self):
        return self.backward_fn.g

    def detach(self, disable_grad=False):
        return Value(self.value, require_grad=False if disable_grad else self.require_grad)

    def exp(self):
        return exp(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Value(" + str(self.value) + ")"

    def __abs__(self):
        return self * Value(1 if self.value > 0 else -1, require_grad=False)

    def __neg__(self):
        return self * Value(-1, require_grad=False)

    def __mul__(self, other):
        if isinstance(other, Value):
            return multiply(self, other)
        elif isinstance(other, float):
            return self * Value(other, require_grad=False)
        elif isinstance(other, int):
            return self * Value(other, require_grad=False)
        else:
            return other * self

    def __rmul__(self, other):
        if isinstance(other, float):
            return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Value):
            return add(self, other)
        elif isinstance(other, float):
            return self + Value(other, require_grad=False)
        elif isinstance(other, int):
            return self + Value(other, require_grad=False)
        else:
            return other + self

    def __radd__(self, other):
        if isinstance(other, float):
            return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Value):
            return subtract(self, other)
        elif isinstance(other, float):
            return self - Value(other, require_grad=False)
        elif isinstance(other, int):
            return self - Value(other, require_grad=False)
        else:
            return (-other) + self

    def __rsub__(self, other):
        if isinstance(other, float):
            return Value(other, require_grad=False) - self

    def __truediv__(self, other):
        if isinstance(other, Value):
            return divide(self, other)
        elif isinstance(other, float):
            return self / Value(other, require_grad=False)
        elif isinstance(other, int):
            return self / Value(other, require_grad=False)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        if isinstance(other, float):
            return Value(other, require_grad=False) / self


def exp(v):
    e = math.exp(v.value)
    return Value(e, require_grad=v.require_grad, backward_fn=ExpBackwardFn(v, e))


def multiply(v, u):
    return Value(v.value * u.value, require_grad=v.require_grad or u.require_grad, backward_fn=MultiplyBackwardFn(v, u))


def add(v, u):
    return Value(v.value + u.value, require_grad=v.require_grad or u.require_grad, backward_fn=AddBackwardFn(v, u))


def subtract(v, u):
    return Value(v.value - u.value, require_grad=v.require_grad or u.require_grad, backward_fn=SubtractBackwardFn(v, u))


def divide(v, u):
    return Value(v.value / u.value, require_grad=v.require_grad or u.require_grad, backward_fn=DivideBackwardFn(v, u))
