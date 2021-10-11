# This file is to generate the matrix for generating colored noise.
import numpy as np
from numpy import linalg as la
import struct


eta = 0.5
N = 576
cov = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        cov[i, j] = eta**(abs(i - j))
        cov[j, i] = cov[i, j]

# v are the eigenvalues, and P the eigenvectors
v, P = la.eig(cov)
V = np.diag(v**(0.5))
transfer_mat = P @ V @ la.inv(P)

with open(f"noise/cov_1_2_corr_para{eta:.2f}.dat", 'wb') as fout:
    transfer_mat.astype(np.float32).T.tofile(fout)
