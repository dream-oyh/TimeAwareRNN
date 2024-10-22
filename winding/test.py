import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("winding\data\winding_missing_prob_0.00.dat")
dt = np.expand_dims(data[:, 8], axis=1)  # (Nsamples, 1)
plt.plot(dt)
plt.show()