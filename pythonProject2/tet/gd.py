#업데이트할 W : Learning Rate * ((Y예측 - Y실제) * X)평균
#업데이트할 b : Learning Rate * ((Y예측 - Y실제) * 1)평균

import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100)
#임의의 랜덤한 x값 100개
Y = 0.2 * X + 0.5
#임의의 선형수식 w = 0.2 b = 0.5 정도로해보자

plt.figure(figsize=(8,6))
plt.scatter(X,Y)
plt.show()

#시각화 해보기 위해 함수 하나 디파인
# 예측값과 y 실제값을 넣어주면 실제값을 뿌려주고 예측값을 같이 뿌려줘서 비교해볼 예정이다.
def plot_prediction(pred, y):
    plt.figure(figsize=(8,6))
    plt.scatter(X,Y)
    plt.scatter(X,pred)
    plt.show()

##Gradient Descent 구현
#랜덤하게 W,b를 생성 초기값은 -1과 1사이의 값을 생성하게 할거임
#업데이트할 W : Learning Rate * ((Y예측 - Y실제) * X)평균
#업데이트할 b : Learning Rate * ((Y예측 - Y실제) * 1)평균
W = np.random.uniform(-1,1)
b = np.random.uniform(-1.1)

learning_rate = 0.7

#에포크를 100번정도 돌려보면서 학습 시켜볼 예정이다.
for epoch in range(200):
    Y_Pred = W * X + b #예측값

    error = np.abs(Y_Pred - Y).mean()
    if error < 0.001: #에러가 어느정도 도달하면 학습을 멈추겠다.
        break

    #gradient descent 계산
    w_grad = learning_rate * ((Y_Pred - Y)*X).mean()
    b_grad = learning_rate * ((Y_Pred - Y)).mean()
    #움직여야할 방향을 정한 것이다.

    #W,b 값 갱신
    W = W - w_grad
    b = b - b_grad

    #에폭을 100번돌때마다 과부하가 심하기 때문에 5의 배수마다 확인하도록 하겠다.
    if epoch % 10 == 0:
        Y_Pred = W * X + b
        plot_prediction(Y_Pred,Y) #실체값과 예측값이 얼마나 근사하는지를 확인해볼 것이다.
