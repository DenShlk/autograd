import random


def get_shape(data):
    try:
        d = data[0]
        return (len(data),) + get_shape(d)
    except Exception as e:
        return tuple()


def rand_tensor(shape, require_grad=True):
    from tensor import Tensor

    if len(shape) == 1:
        return Tensor([
            random.random() * 2 - 1 for _ in range(shape[0])
        ], require_grad=require_grad)
    return Tensor([
        rand_tensor(shape[1:], require_grad) for _ in range(shape[0])
    ])
