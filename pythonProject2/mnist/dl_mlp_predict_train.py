import matplotlib.pyplot as plt
from dl_mlp_learning import mlp
from mnist.dl_load_digits import X_train, y_train
import numpy as np


y_train_pred = mlp.predict(X_train)
success = np.sum(y_train == y_train_pred, axis= 0)
total = X_train.shape[0]
accuracy = success/total

print('예측성공/총개수: [%d]/[%d]' %(success,total))
print('딥러닝 정확도: %.2f%%' %(accuracy*100))