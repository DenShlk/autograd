import math

from util import get_shape


class BackwardContext:

    def __init__(self):
        pass

    @staticmethod
    def execute(base_fn, g):
        order = []
        base_fn.toposort(order)
        base_fn.add_grad(g)

        for fn in order[::-1]:
            fn.propagate()


class BaseBackwardFn:
    def __init__(self):
        self.g = 0.
        self.visited = False

    def toposort(self, lis):
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


class MultiplyBackwardFn(BaseBackwardFn):
    def __init__(self, v, u):
        super().__init__()
        self.v = v
        self.u = u

    def toposort(self, lis):
        self.visited = True
        if self.v.require_grad and not self.v.backward_fn.visited:
            self.v.backward_fn.toposort(lis)
        if self.u.require_grad and not self.u.backward_fn.visited:
            self.u.backward_fn.toposort(lis)
        lis.append(self)

    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.u.value * self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(self.v.value * self.g)

    def zero_grad(self):
        super().zero_grad()
        self.v.zero_grad()
        self.u.zero_grad()


class AddBackwardFn(BaseBackwardFn):
    def __init__(self, v, u):
        super().__init__()
        self.v = v
        self.u = u

    def toposort(self, lis):
        self.visited = True
        if self.v.require_grad and not self.v.backward_fn.visited:
            self.v.backward_fn.toposort(lis)
        if self.u.require_grad and not self.u.backward_fn.visited:
            self.u.backward_fn.toposort(lis)
        lis.append(self)

    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(self.g)

    def zero_grad(self):
        self.v.zero_grad()
        self.u.zero_grad()


class SubtractBackwardFn(BaseBackwardFn):
    def __init__(self, v, u):
        super().__init__()
        self.v = v
        self.u = u

    def toposort(self, lis):
        self.visited = True
        if self.v.require_grad and not self.v.backward_fn.visited:
            self.v.backward_fn.toposort(lis)
        if self.u.require_grad and not self.u.backward_fn.visited:
            self.u.backward_fn.toposort(lis)
        lis.append(self)

    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(-self.g)

    def zero_grad(self):
        self.v.zero_grad()
        self.u.zero_grad()


class DivideBackwardFn(BaseBackwardFn):
    def __init__(self, v, u):
        super().__init__()
        self.v = v
        self.u = u

    def toposort(self, lis):
        self.visited = True
        if self.v.require_grad and not self.v.backward_fn.visited:
            self.v.backward_fn.toposort(lis)
        if self.u.require_grad and not self.u.backward_fn.visited:
            self.u.backward_fn.toposort(lis)
        lis.append(self)

    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(1 / self.u.value * self.g)
        if self.u.require_grad:
            self.u.backward_fn.add_grad(- self.v.value / (self.u.value * self.u.value) * self.g)

    def zero_grad(self):
        self.v.zero_grad()
        self.u.zero_grad()


class ExpBackwardFn(BaseBackwardFn):
    def __init__(self, v, v_exp):
        super().__init__()
        self.v = v
        self.v_exp = v_exp

    def toposort(self, lis):
        self.visited = True
        if self.v.require_grad and not self.v.backward_fn.visited:
            self.v.backward_fn.toposort(lis)
        lis.append(self)

    def propagate(self):
        if self.v.require_grad:
            self.v.backward_fn.add_grad(self.v_exp * self.g)

    def zero_grad(self):
        self.v.zero_grad()


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


class Tensor:
    def __init__(self, data, require_grad=True):
        self.shape = get_shape(data)
        if len(self.shape) > 1:
            self.data = [d if isinstance(d, Tensor) else Tensor(d, require_grad=require_grad) for d in data]
        else:
            self.data = [d if isinstance(d, Value) else Value(d, require_grad=require_grad) for d in data]

    def backward(self):
        assert list(self.shape) == [1] * len(self.shape)

        self.data[0].backward()

    def _apply(self, f, target_shape_length):
        if len(self.shape) == target_shape_length:
            return Tensor([
                f(v) for v in self.data
            ])

        return Tensor([
            d._apply(f, target_shape_length) for d in self.data
        ])

    def item(self):
        assert self.numel() == 1
        return self.data[0].value if isinstance(self.data[0], Value) else self.data[0].item()

    def tolist(self):
        if len(self.shape) == 1:
            return [d.value for d in self.data]
        return [d.tolist() for d in self.data]

    def sum(self):
        if len(self.shape) == 1:
            s = Value(0.0)
            for v in self.data:
                s = s + v
            return s

        s = self.data[0].sum()
        for i in range(1, self.shape[0]):
            s = s + self.data[i].sum()

        return s

    def numel(self):
        if len(self.shape) == 1:
            return self.shape[0]
        return self.shape[0] * self.data[0].numel()

    def apply(self, f, dim=-1):
        if dim >= 0:
            target = len(self.shape) - dim
        else:
            target = -dim

        if len(self.shape) + 1 == target:
            return f(self.data)

        return self._apply(f, target)

    def zero_grad(self):
        for v in self.data:
            v.zero_grad()

    def detach(self):
        return Tensor([v.detach() for v in self.data])

    def grad_tensor(self):
        if len(self.shape) == 1:
            return Tensor([
                v.backward_fn.g for v in self.data
            ], require_grad=False)

        return Tensor(
            [d.grad_tensor() for d in self.data]
        )

    def exp(self):
        if len(self.shape) == 1:
            return Tensor(
                [exp(v) for v in self.data]
            )
        return Tensor([
            t.exp() for t in self.data
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return str(self.data)

    def __abs__(self):
        return Tensor(
            [abs(d) for d in self.data]
        )

    def __neg__(self):
        return self * (-1.0)

    def __add__(self, other):
        if isinstance(other, Tensor):
            assert len(self.data) == len(other.data)

            return Tensor(
                [v + u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Tensor(
                [v + other for v in self.data]
            )
        elif isinstance(other, float):
            return self + Value(other, require_grad=False)
        elif isinstance(other, int):
            return self + Value(other, require_grad=False)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, Tensor):
            assert len(self.data) == len(other.data)

            return Tensor(
                [v - u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Tensor(
                [v - other for v in self.data]
            )
        elif isinstance(other, float):
            return self - Value(other, require_grad=False)
        elif isinstance(other, int):
            return self - Value(other, require_grad=False)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Tensor):
            assert len(self.data) == len(other.data)

            return Tensor(
                [v * u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Tensor(
                [v * other for v in self.data]
            )
        elif isinstance(other, float):
            return self * Value(other, require_grad=False)
        elif isinstance(other, int):
            return self * Value(other, require_grad=False)
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            if len(self.shape) == 2 and len(other.shape) == 2:  # matrices
                assert self.shape[1] == other.shape[0], f"Shapes: {self.shape}, {other.shape}"

                vectors = []
                for i in range(self.shape[0]):
                    values = [Value(0) for _ in range(other.shape[1])]
                    for j in range(other.shape[1]):
                        for k in range(self.shape[1]):
                            values[j] = values[j] + self.data[i][k] * other.data[k][j]
                    vectors.append(Tensor(values))
                return Tensor(vectors)
            else:
                raise NotImplementedError(f"Shapes: {self.shape}, {other.shape}")
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            assert len(self.data) == len(other.data)

            return Tensor(
                [v / u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Tensor(
                [v / other for v in self.data]
            )
        elif isinstance(other, float):
            return self / Value(other, require_grad=False)
        elif isinstance(other, int):
            return self / Value(other, require_grad=False)
        else:
            raise NotImplementedError


class Vector:
    def __init__(self, values):
        self.data = [v if isinstance(v, Value) else Value(v) for v in values]

    def __repr__(self):
        return str(self.data)

    def detach(self):
        return [v.detach() for v in self.data]

    def __add__(self, other):
        if isinstance(other, Vector):
            assert len(self.data) == len(other.data)

            return Vector(
                [v + u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v + other for v in self.data]
            )

    def __sub__(self, other):
        if isinstance(other, Vector):
            assert len(self.data) == len(other.data)

            return Vector(
                [v - u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v - other for v in self.data]
            )

    def __mul__(self, other):
        if isinstance(other, Vector):
            assert len(self.data) == len(other.data)

            return Vector(
                [v * u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v * other for v in self.data]
            )

    def __truediv__(self, other):
        if isinstance(other, Vector):
            assert len(self.data) == len(other.data)

            return Vector(
                [v / u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v / other for v in self.data]
            )


class Matrix:
    def __init__(self, vectors):
        self.data = [v if isinstance(v, Vector) else Vector(v) for v in vectors]

    def __repr__(self):
        return str(self.data)

    def detach(self):
        return [v.detach() for v in self.data]

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert len(self.data) == len(other.data)

            return Matrix(
                [v + u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v + other for v in self.data]
            )

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert len(self.data) == len(other.data)

            return Matrix(
                [v - u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v - other for v in self.data]
            )

    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert len(self.data) == len(other.data)

            return Matrix(
                [v * u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v * other for v in self.data]
            )

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            assert len(self.data) == len(other.data)

            return Matrix(
                [v / u for v, u in zip(self.data, other.data)]
            )
        elif isinstance(other, Value):
            return Vector(
                [v / other for v in self.data]
            )
