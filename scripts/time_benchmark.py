import os
import time
import numpy as np
import pandas as pd
import pyxu.operator as pxop
from src import *


Ns = [64, 128, 256, 512, 1024]
nreps = 20
eps = 1e-4

# Problem specifications
lambda1f = 8e-2
lambda2f = .5
L = .3
psnrdb = 20

if __name__ == "__main__":


    labels = ['N', 'coupled', 'total time', 'solving time']
    res = []
    for i, N in enumerate(Ns):
        print(f"\nSize {N} ({i+1}/{len(Ns)})")
        # Operator
        op = NuFFT(N, L, 0.5, on_grid=True)

        # Setting of the regularization parameter lambda 2
        lambda2 = lambda2f * N ** 2  # Using that svd(A) = N
        lap = pxop.Laplacian((N, N), mode="wrap")
        lap.lipschitz = 8.  # lap.estimate_lipschitz(method='svd')
        lambda2 /= lap.lipschitz ** 2

        # For lambda 1
        vec = np.array([op.dim_in, 1e-10, *[op.dim_in / 2] * (op.dim_out - 2)])
        B_vec = (1 / vec) * FFT_L_gram_vec(op)  # Depends on the samples of the DFT
        Ml2 = pxop.DiagonalOp(lambda2 * B_vec / (vec + lambda2 * B_vec))

        rep_t = []
        rep_s = []
        for _ in range(nreps):
            # Signal
            sparse_signal = compute_sparse(N, (-6, 6), 0.005)
            smooth_signal = compute_smooth(N, 2, (2e-2, 2e-1), int(0.5 * N))
            signal = sparse_signal + smooth_signal
            x0 = signal.reshape(-1)

            # Measurements
            y0 = op(x0)
            y = compute_y(y0, psnrdb)

            lambda1_max = np.abs(op.phi.adjoint(Ml2.adjoint(y))).max()
            lambda1 = lambda1f * lambda1_max

            # Solve
            start = time.time()
            _, solving_t_coupled = solve(y, op, lambda1, lambda2, coupled=True, laplacian=True, eps=eps)
            total_t_coupled = time.time() - start

            start = time.time()
            _, solving_t_decoupled = solve(y, op, lambda1, lambda2, coupled=False, laplacian=True, eps=eps)
            total_t_decoupled = time.time() - start

            res.append([N, True, total_t_coupled, solving_t_coupled])
            res.append([N, False, total_t_decoupled, solving_t_decoupled])

        # write many times to avoid losing data
        df = pd.DataFrame(res, columns=labels)
        df.to_csv(os.path.join(os.getcwd(), '..', 'exps', 'time_bench.csv'), index=False)



