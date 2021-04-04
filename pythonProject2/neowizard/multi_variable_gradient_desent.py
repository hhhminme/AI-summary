import numpy as np
loaded_data = np.loadtxt('./data-01-test-score.csv',delimiter = ',', dtype = np.float32)

x_data = loaded_data[ :, 0:-1] # 모든 행에 대하여 0열부터 3열
t_data = loaded_data[ :,[-1]] # 모든 행에 대하여 4열

W = np.random.rand(3,1) #3X1 행렬
#0과 1사이의 랜덤 값
b = np.random.rand(1)
print("W = ",W,"W.shape = ", W.shape, "b = ",b,"b.shape = ", b.shape)

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


learning_rate = 1e-5
# 발산하는 경우 1e-3 ~ 1e-6 등으로 바꾸어 실행

f = lambda x: loss_func(x_data, t_data)
# f(x) = loss_func(x_data, t_data)

print("Initial error value = ", error_val(x_data, t_data), "Initail W =", W, "\n", ",b =", b)

for step in range(8001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if step % 400 == 0:
        print("step = ", step, "error_value = ", error_val(x_data, t_data), "W = ", W, " b = ", b)

test_data = np.array([100,98,81])

print(predict(test_data))