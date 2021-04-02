#행렬에 열과 행 추가
import numpy as np

A = np.array([[10,20,30],[40,50,60]])
print(A.shape)

#A matrix에 행 추가할 행렬, 1행 3열로 reshape
#행을 추가하기 때문에 우선 열을 3열로 만들어야함
row_add = np.array([70,80,90]).reshape(1,3)

#A matrix에 열 추가할 행렬 2행 1열로 생성해야함
#열을 추가하기 때문에 일단 행을 2행으로 만들어야함
col_add = np.array([1000,2000]).reshape(2,1)
print(col_add)

#numpu.concatenate 에서 axis = 0행(row) 기준
# A행렬에 row_add 추가
B = np.concatenate((A,row_add),axis=0)
print(B)

#axis = 1열 기준
C = np.concatenate((A,col_add),axis = 1)
print(C)