import matplotlib.pyplot as plt
from dl_mlp_learning import mlp
import numpy as np

plt.plot(range(len(mlp.cost_)),mlp.cost_)
plt.ylim([0,2000])
plt.tight_layout()
plt.ylabel('J(w)')
plt.xlabel('Epochs*50')
plt.tight_layout()
plt.show()
