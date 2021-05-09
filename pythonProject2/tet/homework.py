import numpy as np
import matplotlib.pyplot as plt

X = 1
#초기 X 인풋값
Y = 5
#임의의 Y타겟값.
#임의의 선형 수식은 Y = X로 구현
plt.scatter(X,X)
##Gradient Descent 구현
#업데이트할 W : Learning Rate * ((Y예측 - Y실제) * X)
#업데이트할 b : Learning Rate * ((Y예측 - Y실제) * 1)
#초기 w값과 b값 지정
W = 2
b = 1

#학습률은 0.1과 0.2로진행
learning_rate = 0.1
learning_rate2 = 0.1

#에포크를 200번정도 돌려보면서 학습 시켜볼 예정이다.
for epoch in range(100):
    Y_Pred = W * X + b #예측값

    error = 0.5*(Y_Pred - Y)**2
    if error <= 1e-50: #예측값이 타겟값보다 커질경우 학습을 종료한다.
        print("BREAK epoch------{0}".format(epoch))
        print('y_pred : {0} , y : {1}'.format(Y_Pred, Y))
        print('w_grad : {0} , W : {1}'.format(w_grad, W))
        print('b_grad : {0} , b : {1}'.format(b_grad, b))
        print('error : {0}'.format(error))
        plt.scatter(Y_Pred, Y_Pred)
        break
    #gradient descent 계산
    w_grad = learning_rate2 * ((Y_Pred - Y)*X)
    b_grad = learning_rate2 * ((Y_Pred - Y))
    #움직여야할 방향을 정한 것이다.

    #W,b 값 갱신
    W = W - w_grad
    b = b - b_grad

    # 10번째마다 학습현황을 파악
    if epoch % 10 == 0:
        print("epoch------{0}".format(epoch))
        print('y_pred : {0} , y : {1}'.format(Y_Pred,Y))
        print('w_grad : {0} , W : {1}'.format(w_grad,W))
        print('b_grad : {0} , b : {1}'.format(b_grad,b))
        print('error : {0}'.format(error))
        plt.scatter(Y_Pred,Y_Pred)
        # plt.scatter(error,epoch)

plt.title('gradient descent homework, Learning_rate = 0.2')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()