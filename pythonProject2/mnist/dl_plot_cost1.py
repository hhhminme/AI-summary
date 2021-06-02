import matplotlib.pyplot as plt
from dl_mlp_learning import mlp
import numpy as np

from mnist.dl_load_digits import X_test, y_test
from mnist.dl_mlp_predict_test import y_test_pred

batches = np.array_split(range(len(mlp.cost_)),1000)
cost_ary = np.array(mlp.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)),cost_avgs, color='red')
plt.ylim([0,2000])
plt.tight_layout()
plt.ylabel('J(w)')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

