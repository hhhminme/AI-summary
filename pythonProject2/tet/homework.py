import numpy as np
import matplotlib.pylab as plt


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    tmp_val = x
    # f(x+h) 계산
    x = float(tmp_val) + h
    fxh1 = f(x)

    # f(x-h) 계산
    x = tmp_val - h
    fxh2 = f(x)

    grad = (fxh1 - fxh2) / (2 * h)
    x = tmp_val
    # 값 복원
    return grad

def numerical_gradient(f, X):
    return _numerical_gradient_no_batch(f, X)

def gradient_descent(f, init_x, lr=0.1, step_num=10):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x)

        grad = numerical_gradient(f, x)
        x += lr * grad

    return x, np.array(x_history)


def function(x):
    return x

init_x = 1
lr = 0.1
step_num = 50
x, x_history = gradient_descent(function, init_x, lr=lr, step_num=step_num)

plt.plot( [-10, 10], [0,0], '--b')
plt.plot( [0,0], [-10, 10], '--b')
plt.plot(x_history[:], x_history[:], 'o')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
