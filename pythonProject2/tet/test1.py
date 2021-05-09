import numpy as np
import matplotlib.pyplot as plt

dataNum = 1000
trainData1 = np.random.randn(dataNum,2)
#randn 은 정규분포라고 불리는 값을 return 하는 함수이다. 대체로 -1~1 사이의 값을 리턴
# 1000행 2열
trainData2 = np.random.randn(dataNum,2) + 5
trainData = np.zeros((dataNum*2,2))
# 데이터를 만들고
trainData[0:dataNum,:] = trainData1
trainData[dataNum:dataNum*2,:] =trainData2
trainout = np.zeros((dataNum*2,1))
trainout[dataNum:dataNum*2,:] = np.zeros((dataNum,1)) + 1

print(trainout)

figure = plt.figure()
ax = plt.gca()

ax.plot(trainData1[:, 0],trainData1[:, 1],"*")
ax.plot(trainData2[:, 0],trainData2[:, 1],"*")
# 데이터 1이라는 어레이를 생성했잖아요 1000 by 2라는 어레이 일거 아니에요.
# 콜론은 전체를 의미하는 것이다 앞에껀 첫번째 인덱스에 접근하는 것이고,
# 1 은 두번째 인덱스에 접근하는 것이다.
# 행은 전부 다 가져오는 것이고 여은 첫번째만 가져온다는 뜻.
#  [ 0 1
#    2 3
#    4 5 ...
step = 100
input = 2
out = 1
w = np.random.rand(input,out) #2 X 1 (입력값이 두개라 그런듯 w1,w2)
b = np.random.randn(1)
a = np.arange(-3,6,0.1) # -3부터 6까지 0.1 step 만큼 증가하는 array를 만들어 주겠다.
#a 는 활성 함수값
# x1과 x2에 대한 관계식으로 나타나게 된다. 그리고 이걸 x1과 x2를 구분하기 위한 학습시킨다.
# 결졍 경계, 초평면 이라고 부른다.
# 뭐 그래도 되고 영원히 그렇게 할 수 없으니깐 가중치를 갱신하는 방법에 대해 배웠잖아요. 그래서 일단은 위에 경사하강법말고 델타규칙.
# 이 델타규칙을 이용해가지고 없데이트를 할거에요. 그러면 학습을 어떻게 할것이냐.
plt.plot(-w[0,0]*a-b/w[1,0])
eta = 0.5
#eta 는 학습률을 의미한다.

for j in range(1,step,1):
    for i in range(0, dataNum*2,1):
        x = trainData[i]
        ry = trainout[i]
        if(np.dot(x,w) + b) > 0:
            y = 1
        else:
            y = 0
        e = ry - y

        dw = eta * e * np.transpose([np.array(x)])
        db = eta * e * 1

        w = w + dw
        b = b + db

plt.plot(a,(-w[0,0]*a-b)/w[1,0],'r')
plt.show()
