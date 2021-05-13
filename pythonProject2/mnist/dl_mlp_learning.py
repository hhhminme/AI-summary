import pickle
import os
from mnist.dl_mlp_class import NeuralNetMLP
from mnist.dl_load_digits import X_train, y_train

mlp = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50,
                   I2=0.1, I1=0.0, epochs=1000, eta=0.001, alpha=0.001,
                   decrease_const=0.00001, shuffle=True, minibatches=50, random_state=1)

mlp.fit(X_train, y_train, print_progress=True)

with open(os.path.join('C:\\Users\\허민\\Desktop\\2021-1\\인공지능\\pythonProject2\\mnist\\data\\dataset',
                       'mlp_digits.pkl'),'wb') as f:
    pickle.dump(mlp, f, protocol=4)

print('머신러닝 데이터 저장 완료')

