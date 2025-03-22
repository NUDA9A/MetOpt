import numpy as np


def f1(args):
    return args[0] ** 2 + args[1] ** 2


def f1_1(args):
    return (args[0] + 2) ** 2 + args[1] ** 2


def f1_2(args):
    return args[0] ** 2 + (args[1] - 3) ** 2


def f1_3(args):
    return (args[0] - 2) ** 2 + (args[1] + 1) ** 2


B = 1.5


def f3(args):
    return args[0] ** 2 + B * args[0] * args[1] + args[1] ** 2


def multimodal_f(args):
    return 20 + args[0] ** 2 + args[1] ** 2 - 10 * np.cos(2*np.pi * args[0]) - 10 * np.cos(2 * np.pi * args[1])


N = 0.2
M = 3


def noisy_multimodal_f(args):
    np.random.seed(42)
    x, y = args[0], args[1]
    value = 0
    for m in range(1, M + 1):
        value += np.sin(m * x) * np.cos(m * y)
    noise = np.random.normal(0, N)
    return value + noise
