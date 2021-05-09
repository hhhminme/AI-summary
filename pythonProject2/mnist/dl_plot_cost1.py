import matplotlib.pyplot as plt
from dl_mlp_learning import mlp
import numpy as np

batches = np.array_split(range(len(mlp.cost_)),1000)
cost_ary = np.array(mlp.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(mlp.cost_)),mlp.cost_)
plt.ylim([0,2000])
plt.tight_layout()
plt.ylabel('J(w)')
plt.xlabel('Epochs')
plt.show()