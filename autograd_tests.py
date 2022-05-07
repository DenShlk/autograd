from autograd import *
import random
import torch

ITERATIONS_BUDGET = 10000

random.seed(4269)


def random_float():
    return random.random() * 2 - 1


def get_diff(my_val, torch_val):
    if isinstance(my_val, Value):
        return get_diff(my_val.item(), torch_val)
    if isinstance(torch_val, torch.Tensor):
        return get_diff(my_val, torch_val.item())

    return abs(my_val - torch_val)


def get_relative_diff(my_val, torch_val):
    if isinstance(my_val, Value):
        return get_relative_diff(my_val.item(), torch_val)
    if isinstance(torch_val, torch.Tensor):
        return get_relative_diff(my_val, torch_val.item())

    return abs(abs(my_val / torch_val) - 1)


def assert_equal(my_val, torch_val):
    assert get_diff(my_val, torch_val) == 0.


def assert_almost_equal(my_val, torch_val, eps=1e-4):
    assert get_diff(my_val, torch_val) < eps or get_relative_diff(my_val, torch_val) < eps


def base_value_binary_op_test(op, a, b, eps=1e-4):
    truth = op(torch.Tensor([a]), torch.Tensor([b]))
    assert_almost_equal(op(Value(a), Value(b)), truth, eps=eps)
    assert_almost_equal(op(Value(a), b), truth, eps=eps)
    assert_almost_equal(op(a, Value(b)), truth, eps=eps)


def value_binary_op_normalized_test(op):
    for _ in range(ITERATIONS_BUDGET):
        a = random_float()
        b = random_float()
        base_value_binary_op_test(op, a, b)


def value_binary_op_different_scales_test(op):
    iters = ITERATIONS_BUDGET // 5
    power = 5
    eps = 1e-4

    for _ in range(iters):
        a = random_float() * (10 ** random.randint(-power, power))
        b = random_float() * (10 ** -power)
        base_value_binary_op_test(op, a, b, eps=eps)
        base_value_binary_op_test(op, b, a, eps=eps)

    for _ in range(iters):
        a = random_float() * (10 ** power)
        b = random_float() * (10 ** random.randint(-power, power))
        base_value_binary_op_test(op, a, b, eps=eps)
        base_value_binary_op_test(op, b, a, eps=eps)

    for _ in range(iters):
        a = random_float() * (10 ** random.randint(-power, power))
        b = random_float() * (10 ** random.randint(-power, power))
        base_value_binary_op_test(op, a, b, eps=eps)


def test_value_add():
    def op(a, b):
        return a + b

    value_binary_op_normalized_test(op)
    value_binary_op_different_scales_test(op)


def test_value_sub():
    def op(a, b):
        return a - b

    value_binary_op_normalized_test(op)
    value_binary_op_different_scales_test(op)


def test_value_mul():
    def op(a, b):
        return a * b

    value_binary_op_normalized_test(op)
    value_binary_op_different_scales_test(op)


def test_value_div():
    def op(a, b):
        return a / b

    value_binary_op_normalized_test(op)
    value_binary_op_different_scales_test(op)


def gen_random_expression(var_names, single_op_names, binary_op_names, op_count):
    if op_count == 0:
        return random.choice(var_names)

    if random.randint(0, len(single_op_names) + len(binary_op_names)) < len(single_op_names):
        # single op
        return random.choice(single_op_names) + '(' + \
               gen_random_expression(var_names, single_op_names, binary_op_names, op_count - 1) + ')'
    # binary op
    left_op_count = op_count - 1 - random.randint(0, op_count - 1)
    right_op_count = op_count - 1 - left_op_count
    left = gen_random_expression(var_names, single_op_names, binary_op_names, left_op_count)
    right = gen_random_expression(var_names, single_op_names, binary_op_names, right_op_count)
    return f'({left} {random.choice(binary_op_names)} {right})'


def execute_expression(expr, var_names, var_values, mode, finalizer=lambda res, vals: res, require_grad=False):
    values = [0] * len(var_names)
    for i, name, value in zip(range(len(var_names)), var_names, var_values):
        if mode == 'torch':
            exec(f'{name} = torch.tensor([{value}], requires_grad={require_grad})')
        elif mode == 'autograd':
            exec(f'{name} = Value({value}, require_grad={require_grad})')
        values[i] = eval(name)

    if mode == 'torch':
        expr = expr.replace('exp', 'torch.exp')
        torch.autograd.set_detect_anomaly(True)
    try:
        result = eval(expr)
    except ZeroDivisionError:
        return 'zero_div', 0
    except OverflowError:
        return 'overflow', 0
    if (mode == 'torch' and abs(result) > 1e11) or (mode == 'autograd' and abs(result.value) > 1e11):
        return 'inf', 0

    return finalizer(result, values), 1


def test_value_random_expressions():
    budget = ITERATIONS_BUDGET
    while budget > 0:
        ops = random.randint(0, int(budget ** .5))
        budget -= ops
        vars_count = random.randint(1, 26)
        var_names = [f'var_{i}' for i in range(vars_count)]

        expr = gen_random_expression(var_names, ['exp'], ['+', '-', '*', '/'], ops)
        values = [random_float() for _ in range(vars_count)]
        my_res, my_ok = execute_expression(expr, var_names, values,
                                           'autograd')
        torch_res, torch_ok = execute_expression(expr, var_names, values,
                                                 'torch')

        if not torch_ok:
            assert not my_ok

        #assert my_ok == torch_ok or my_res == 'zero_div'
        # sorry, i gave up, torch uses inf values, and the end result may turn out to be finite, so it breaks
        if my_ok:
            assert_almost_equal(my_res, torch_res, eps=1e-3)
        else:
            # compare exceptions
            # assert my_res == torch_res
            # actually division by zero produces exception or infinity, so it is hard to check are they equal or not
            pass


def test_gradients_for_random_expressions():
    budget = ITERATIONS_BUDGET // 2
    while budget > 0:
        ops = random.randint(0, int(budget ** .5))
        budget -= ops
        vars_count = random.randint(1, 26)
        var_names = [f'var_{i}' for i in range(vars_count)]

        expr = gen_random_expression(var_names, ['exp'], ['+', '-', '*', '/'], ops)
        values = [random_float() for _ in range(vars_count)]

        def collect_grad_values_autograd(res, vals):
            res.backward()
            return [v.grad() for v in vals]

        def collect_grad_values_torch(res, vals):
            res.backward()
            return [v.grad for v in vals]

        my_res, my_ok = execute_expression(expr, var_names, values,
                                           'autograd', finalizer=collect_grad_values_autograd, require_grad=True)
        torch_res, torch_ok = execute_expression(expr, var_names, values,
                                                 'torch', finalizer=collect_grad_values_torch, require_grad=True)

        if my_ok and torch_ok:
            for x, y in zip(my_res, torch_res):
                if y is None:
                    assert x == 0.0
                else:
                    assert_almost_equal(x, y)
