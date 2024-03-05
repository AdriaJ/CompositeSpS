import matplotlib.pyplot as plt
import numpy as np

from src import *

## Parameters

# Signal size
N = 128
# rate of measurements
L = 0.3

# sampling parameters
theta = 0.5

# Sparse signal parameters
sparse_range = (-6, 6)
density = 0.005

# Smooth signal parameters
smooth_amplitude = 2
sigmas_range = (2e-2, 2e-1)
nb_gaussian = int(0.5 * N)

# Noise
psnr = 20

# Seed
seed = 21

# Reconstruction parameters
lambda1f = 5e-2
lambda2f = 1.
eps = 1e-4

# Use Laplacian ?
laplacian = True

history = True

if __name__ == "__main__":
    ## Problem setup
    # Simulate source signal
    sparse_signal = compute_sparse(N, sparse_range, density, seed)
    smooth_signal = compute_smooth(N, smooth_amplitude, sigmas_range, nb_gaussian, seed)
    signal = sparse_signal + smooth_signal
    x0 = signal.reshape(-1)

    # Operator
    op = NuFFT(N, L, theta, on_grid=True, seed=seed)

    # Measurements
    y0 = op(x0)
    y = compute_y(y0, psnr)

    ## retro compute psnr
    import pyxu.util.complex as pxuc
    noise = pxuc.view_as_complex(y - y0)
    20 * np.log10(np.abs(pxuc.view_as_complex(y0)).max()/np.linalg.norm(noise, 2))
    #
    # print(np.linalg.norm(y - y0))

    # Regularization
    lambda2 = lambda2f * N ** 2  # Using that svd(A) = N
    lap = Laplacian((N, N), mode="wrap")
    lap.lipschitz = 8  # lap.estimate_lipschitz(method='svd')
    lambda2 /= lap.lipschitz ** 2
    # lambda1 = lambda1f * np.abs(op.phi.adjoint(y)).max()
    # print(lambda1/lambda2)

    # Lambda_r = lambda1\lambda2
    # Compute lambda_r max
    vec = np.array([op.dim_in, 1e-10, *[op.dim_in / 2] * (op.dim_out - 2)])
    B_vec = (1 / vec) * FFT_L_gram_vec(op)  # Depends on the samples of the DFT
    Ml2 = DiagonalOp(lambda2 * B_vec / (vec + lambda2 * B_vec))
    lambda1_max = np.abs(op.phi.adjoint(Ml2.adjoint(y))).max()

    lambda1 = lambda1f * lambda1_max

    # print(lambda1_max)

    # compute and show dirty image
    dirty_image = op.phi.adjoint(y)
    dirty_image = dirty_image.reshape((N, N))
    plt.figure()
    divnorm = colors.CenteredNorm(vcenter=0.0, halfrange=dirty_image.max())
    plt.imshow(dirty_image, cmap="seismic", norm=divnorm, interpolation="none")
    plt.title(f"Dirty image with PSNR = {psnr:d} dB")
    plt.colorbar()
    plt.show()

    # -----------------------------------------
    ## Reconstruction

    # Coupled
    coupled = True
    start_coupled = time.time()
    (x1_coupled, x2_coupled), t_coupled, hist_coupled = solve(y, op, lambda1, lambda2, coupled, laplacian, eps=eps, history=history)
    total_t_coupled = time.time() - start_coupled

    # Decoupled
    coupled = False
    start_decoupled = time.time()
    (x1_decoupled, x2_decoupled), t_decoupled, hist_decoupled = solve(y, op, lambda1, lambda2, coupled, laplacian, eps=eps, history=history)
    total_t_decoupled = time.time() - start_decoupled

    # -----------------------------------------
    ## Analysis
    # laplacian_op = Laplacian((N, N), mode="wrap")

    approaches = ["Coupled", "Decoupled"]
    sparse_rcstr_coupled, smooth_rcstr_coupled, signal_rcstr_coupled = map(
        lambda x: x.reshape((N, N)), (x1_coupled, x2_coupled, x1_coupled + x2_coupled)
    )
    sparse_rcstr_decoupled, smooth_rcstr_decoupled, signal_rcstr_decoupled = map(
        lambda x: x.reshape((N, N)), (x1_decoupled, x2_decoupled, x1_decoupled + x2_decoupled)
    )
    data_sets = [
        (sparse_rcstr_coupled, smooth_rcstr_coupled, t_coupled, total_t_coupled),
        (sparse_rcstr_decoupled, smooth_rcstr_decoupled, t_decoupled, total_t_decoupled),
    ]

    for approach, (sparse_rcstr, smooth_rcstr, t, total_t) in zip(approaches, data_sets):
        print(f"{approach} Approach")

        data_fidelity, L2, L1, = objective_func(
            op,
            lap,
            y,
            sparse_rcstr,
            smooth_rcstr,
            lambda1,
            lambda2,
        )

        # Value of the objective functions (Sanity check)
        print(
            f"  Cost:  {data_fidelity + L2 + L1:.2e}\n" +
            f"    - Data fidelity: {data_fidelity:.2e}\n" +
            f"    - L2: {L2:.2e} \n" +
            f"    - L1: {L1:.2e}"
        )

        # Reconstruction time
        print(f"  Time [s]: \n" +
              f"    - With preprocessing : {total_t :.2f}\n" +
              f"    - Without preprocessing: {t:.2f} \n")

    # Metrics
    x1_source = sparse_signal.reshape(-1)

    print("Sparse signal analysis")

    for x1, t in zip([x1_coupled, x1_decoupled, x1_source], ['Coupled', 'Decoupled', 'Source']):
        print(t)
        print(f"\tSparsity: {np.count_nonzero(x1)}")
        print(f"\tAlmost sparsity: {(np.abs(x1) > 1e-1).sum()}")
        print(f"\tMin value: {x1.min()}")
        print(f"\tMax value: {x1.max()}\n")

    print("L2 norm:")
    print(f"\tRelative difference between the reconstructions: "
          f"{np.linalg.norm(x1_decoupled - x1_coupled, 2) / np.linalg.norm(x1_source, 2):.3f}")
    print(f"\tRelative difference between coupled and source: "
          f"{np.linalg.norm(x1_source - x1_coupled, 2) / np.linalg.norm(x1_source, 2):.3f}")
    print(f"\tRelative difference between decoupled and source: "
          f"{np.linalg.norm(x1_decoupled - x1_source, 2) / np.linalg.norm(x1_source, 2):.3f}")

    print("L1 norm:")
    print(f"\tRelative difference between the reconstructions: "
          f"{np.linalg.norm(x1_decoupled - x1_coupled, 1) / np.linalg.norm(x1_source, 1):.3f}")
    print(f"\tRelative difference between coupled and source: "
          f"{np.linalg.norm(x1_source - x1_coupled, 1) / np.linalg.norm(x1_source, 1):.3f}")
    print(f"\tRelative difference between decoupled and source: "
          f"{np.linalg.norm(x1_decoupled - x1_source, 1) / np.linalg.norm(x1_source, 1):.3f}")

    x2_source = smooth_signal.reshape(-1)

    print("Smooth signal analysis")

    for x2, t in zip([x2_coupled, x2_decoupled, x2_source], ['Coupled', 'Decoupled', 'Source']):
        print(t)
        print(f"\tMin value: {x2.min()}")
        print(f"\tMax value: {x2.max()}\n")

    print("L2 norm:")
    print(f"\tRelative difference between the reconstructions: "
          f"{np.linalg.norm(x2_decoupled - x2_coupled, 2) / np.linalg.norm(x2_source, 2):.3f}")
    print(f"\tRelative difference between coupled and source: "
          f"{np.linalg.norm(x2_source - x2_coupled, 2) / np.linalg.norm(x2_source, 2):.3f}")
    print(f"\tRelative difference between decoupled and source: "
          f"{np.linalg.norm(x2_decoupled - x2_source, 2) / np.linalg.norm(x2_source, 2):.3f}")

    print("Metrics in the center of the image (l2):")
    window = slice(N//4, N - N//4)
    def crop_mse(reco, source, window):
        reco_im = reco.reshape((N, N))[window, window]
        source_im = source.reshape((N, N))[window, window]
        return np.linalg.norm((reco_im-source_im).reshape(-1), 2) / np.linalg.norm(source_im.reshape(-1), 2)
    print(f"\tRelative difference between coupled and source: "
          f"{crop_mse(x2_coupled, x2_source, window):.3f}")
    print(f"\tRelative difference between decoupled and source: "
          f"{crop_mse(x2_decoupled, x2_source, window):.3f}")

    print("L1 norm:")
    print(f"\tRelative difference between the reconstructions: "
          f"{np.linalg.norm(x2_decoupled - x2_coupled, 1) / np.linalg.norm(x2_source, 1):.3f}")
    print(f"\tRelative difference between coupled and source: "
          f"{np.linalg.norm(x2_source - x2_coupled, 1) / np.linalg.norm(x2_source, 1):.3f}")
    print(f"\tRelative difference between decoupled and source: "
          f"{np.linalg.norm(x2_decoupled - x2_source, 1) / np.linalg.norm(x2_source, 1):.3f}")

    f = compare3([sparse_rcstr_coupled, sparse_rcstr_decoupled, sparse_signal],
                 names=["Coupled", "Decoupled", "Source"],
                 title="Sparse component, " + r"($\lambda_1=$" + f"{lambda1f:.3f}" + r" & $\lambda_2=$" + f"{lambda2f:.3f})")
    f.show()
    f = compare3([smooth_rcstr_coupled, smooth_rcstr_decoupled, smooth_signal],
                 names=["Coupled", "Decoupled", "Source"],
                 title="Smooth component, " + r"($\lambda_1=$" + f"{lambda1f:.3f}" + r" & $\lambda_2=$" + f"{lambda2f:.3f})")
    f.show()
    f = compare3([signal_rcstr_coupled, signal_rcstr_decoupled, signal],
                 names=["Coupled", "Decoupled", "Source"],
                 title="Total signal, " + r"($\lambda_1=$" + f"{lambda1f:.3f}" + r" & $\lambda_2=$" + f"{lambda2f:.3f})")
    f.show()

    # import pyxu.operator as pxop
    #
    # N = 10
    # op = pxop.Laplacian((N, N), mode="wrap")
    # op.svdvals(1, which="SM")
    #
    # laplacian_op.estimate_lipschitz()

    # op.phi.estimate_lipschitz(method='trace')
    # lambda_max = np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
    # lambda_2 = lambda2 * lambda_max
    #
    # lap = Laplacian((N, N), mode="wrap")
    # lap.estimate_lipschitz(method='trace')
    # lap.estimate_lipschitz(method='svd')

    ####################
    ## Simple LASSO solution
    do_lasso = False
    if do_lasso:
        print("Solve with basic LASSO problem")
        import pyxu.operator as pxop
        import pyxu.opt.solver as pxs
        import pyxu.opt.stop as pxstop

        lambda_lasso_f = .2

        lambda_lasso_max = np.abs(op.phi.adjoint(y)).max()
        lambda_lasso = lambda_lasso_f * lambda_lasso_max

        lasso_data_fid = 0.5 * pxop.SquaredL2Norm(dim=op.dim_out).asloss(y) * op.phi
        lasso_regul = lambda_lasso * pxop.L1Norm(op.dim_in)
        sc = pxstop.MaxIter(n=100) & pxstop.RelError(eps=eps)
        pgd = pxs.PGD(lasso_data_fid, lasso_regul, verbosity=500)

        start = time.time()
        pgd.fit(x0=np.zeros(op.dim_in), stop_crit=sc)
        lasso_time = time.time() - start
        x_lasso = pgd.solution()

        print("Time [s]: ", lasso_time)

        f = compare3([x_lasso.reshape((N, N)), sparse_rcstr_decoupled, sparse_signal],
                     names=["Lasso", "Decoupled", "Source"],
                     title="Sparse component, " + r"($\lambda_1=$" + f"{lambda1f:.3f}" + r" & $\lambda_2=$" + f"{lambda2f:.3f})")
        f.show()

    ####################
    ## Show the relative improvement stopping criterion

    plt.figure()
    plt.scatter(hist_coupled['iteration'][1:], hist_coupled['RelError[x]'][1:], label="Coupled solve")
    plt.scatter(hist_decoupled['iteration'][1:], hist_decoupled['RelError[x]'][1:], label="Decoupled solve")
    plt.yscale('log')
    plt.legend()
    plt.title("Relative improvement of the reconstruction")
    plt.show()
