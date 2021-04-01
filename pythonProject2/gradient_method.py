# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.1, step_num=10):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x)

        grad = numerical_gradient(f, x)
        x += lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x

init_x = 1

lr = 0.1
step_num = 50
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-10, 10], [0,0], '--b')
plt.plot( [0,0], [-10, 10], '--b')
plt.plot(x_history[:], x_history[:], 'o')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
