import numpy as np
import pandas as pd

data = pd.read_csv("winding\data\odom-19-02-2024-run6.csv", index_col=0).to_numpy()
t = np.expand_dims(data[:, 0], axis=1)  # (Nsamples, 1)

X_1 = data[:, 1:11]  # (Nsamples, 10)
X_2 = data[:, 26:34]  # (Nsamples, 8)
X = np.hstack((X_1, X_2))  # (Nsamples, 18) x,y,z,qx,qy,qz,qw,bu,bv,bw,pwm1-8
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = data[:, 11:14]  # (Nsamples, 3)
k_in = X.shape[1]
k_out = Y.shape[1]
sample_rate = 0.1
dt = sample_rate * np.ones(
    (X.shape[0], 1)
)  # (Nsamples, 1) # In out version, assume sample rate is 0.1

N = X.shape[0]  # number of samples in total