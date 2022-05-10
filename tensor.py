from util import get_shape
from value import Value, exp


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