import numpy as np

W = np.random.rand(1, 1)
b = np.random.rand(1)
print("W = ", W, ",W.shape = ", W.shape, ", b =", b, ",b.shape", b.shape)

x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_func(x, t):
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log(1 - y + delta))


# 작은 값을 더해주지 않으면 y가 0이나 1일 경우 아주 작은 값을 더해주어야 한다.
# 수치해석에서 로그를 구현하는 일반적인 부분

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)  # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad


def error_val(x, t):
    delta = 1e-7  # log 무한대 발생 방지

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # cross-entropy
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) * delta))


def predict(x):
    z = np.dot(x, W) + b
    y = sigmoid(z)

    if y > 0.5:
        result = 1  # true
    else:
        result = 0  # false
    return y, result


learning_rate = 1e-2  # 발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어 실행
f = lambda x: loss_func(x_data, t_data)  # f(x) = loss_func(x_data, t_data)

print("Initial error value = ", error_val(x_data, t_data), "Initial W =", W, "\n", ",b =", b)

for step in range(10001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if (step % 400 == 0):
        print("step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, ",b = ", b)

(real_val, logical_val) = predict(3)
print(real_val, logical_val)
(real_val, logical_val) = predict(17)
print(real_val, logical_val)