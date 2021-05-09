import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1.0])
t_data = np.array([5.0])
# 학습데이터 행렬로 변경

# raw_data = [[1,2],[2,3],[3,4],[4,5],[5,6]]
# 입력값과 정답값이 함께 주어진다면 list comprihension을 통해 분리시켜야한다. 생각해보기

W = np.array([2.0])
# 0과 1사이의 랜덤 값
b = np.array([1.0])
print("W = ", W, "W.shape = ", W.shape, "b = ", b, "b.shape = ", b.shape)


def loss_func(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y) ** 2)) / (len(x))


def numerical_derivative(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


# 손실 함수 값 계산 함수
# 입력변수 x, t : numpy type
def error_val(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y) ** 2)) / (len(x))


# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x : numpy type
def predict(x):
    y = np.dot(x, W) + b
    return y


learning_rate = 0.1
# 발산하는 경우 1e-3 ~ 1e-6 등으로 바꾸어 실행

f = lambda x: loss_func(x_data, t_data)
# f(x) = loss_func(x_data, t_data)

print("Initial error value = ", error_val(x_data, t_data), "Initail W =", W, "\n", ",b =", b)

for step in range(8001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if error_val(x_data,t_data) <= 1e-10:
        print(" BREAK step = ", step, "error_value = ", error_val(x_data, t_data), "W = ", W, " b = ", b)
        break

    if step % 100== 0:
        print("step = ", step, "error_value = ", error_val(x_data, t_data), "W = ", W, " b = ", b)
        plt.scatter(step, error_val(x_data, t_data))

print("Predict value : ",predict(1.0))
plt.show()