import matplotlib.pyplot as plt
from mnist.dl_load_digits import X_train, y_train, X_test, y_test
import numpy as np

from mnist.dl_load_mlpobj import mlp

y_test_pred = mlp.predict(X_test)
success = np.sum(y_test == y_test_pred, axis= 0)
total = X_test.shape[0]
accuracy = success/total

print('예측성공/총개수: [%d]/[%d]' %(success, total))
print('딥러닝 정확도: %.2f%%' %(accuracy*100))

miscl_img = X_test[y_test != y_test_pred][:25]
correct_digit = y_test[y_test != y_test_pred][:25]
miscl_digit = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.ravel()
for i in range(25):
    img = miscl_img[i].reshape(28,28)
    ax[i].imshow(img, cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) T : %d P : %d' %(i + 1, correct_digit[i], miscl_digit[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()