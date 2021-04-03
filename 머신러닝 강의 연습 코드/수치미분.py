import numpy as np
def numerical_gradient(f, x):
    delta_x = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    print("debug 1. initial input variable = ", x)
    print("debug 2. initial grad =", grad)
    print("-------------")

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        print("debug 3. idx =", idx, ", x[idx] =", x[idx])
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + delta_x
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - delta_x
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * delta_x)

        print("debug 4. grad[idx] =", grad[idx])
        print("debug 5. grad = ", grad)
        print("-------------")

        x[idx] = tmp_val  # 값 복원
        it.iternext()
    return grad


def func1(input_obj):
    x = input_obj[0]
    y = input_obj[1]

    return (2 * x + 3 * x * y + np.power(y, 3))

def func2(input_obj):
    w = input_obj[0,0]
    x = input_obj[0,1]
    y = input_obj[1,0]
    z = input_obj[1,1]

    return (w*x + x*y*z + 3*w + z*np.power(y,2))



input = np.array([1.0, 2.0])
numerical_gradient(func1, input)
print('---------------------------------------------')
input2 = np.array([[1.0,2.0],[3.0,4.0]])
numerical_gradient(func2,input2)