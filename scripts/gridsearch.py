from src import *
import os
import pickle
import pandas as pd

## Parameters
N = 128
L = 0.3
theta = 0.5  # sampling parameters
psnrdb = 20
seed = 10

# Sparse signal parameters
sparse_range = (-6, 6)
density = 0.005
# Smooth signal parameters
smooth_amplitude = 2
sigmas_range = (2e-2, 2e-1)
nb_gaussian = int(0.5 * N)

# Reconstruction parameters
eps = 1e-4

lambda1f = [0.01, 0.05, 0.1, 0.2,]
lambda2f = [0.1, 0.5, 1., 2.]

# Use Laplacian ?
laplacian = True
do_coupled = False

if __name__ == "__main__":
    save_path = f"../exps/psnrdb{psnrdb:d}_N{N:d}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # simulate problem
    sparse_signal = compute_sparse(N, sparse_range, density, seed)
    smooth_signal = compute_smooth(N, smooth_amplitude, sigmas_range, nb_gaussian, seed)
    signal = sparse_signal + smooth_signal
    x0 = signal.reshape(-1)

    # Operator
    op = NuFFT(N, L, theta, on_grid=True, seed=seed)

    # Measurements
    y0 = op(x0)
    y = compute_y(y0, psnrdb)

    vec = np.array([op.dim_in, 1e-10, *[op.dim_in / 2] * (op.dim_out - 2)])
    B_vec = (1 / vec) * FFT_L_gram_vec(op)  # Depends on the samples of the DFT

    lap = Laplacian((N, N), mode="wrap")
    lap.lipschitz = lap.estimate_lipschitz(method='svd')

    data_fidelity = lambda x1, x2: .5 * np.sum((op(x1 + x2) - y) ** 2)
    l1_norm = lambda x1: np.sum(np.abs(x1))
    l2_gen_norm = lambda x2: .5 * np.sum(lap(x2) ** 2)
    rel_l1_err = lambda x, x0: np.linalg.norm(x - x0, ord=1) / np.linalg.norm(x0, ord=1)
    rel_l2_err = lambda x, x0: np.linalg.norm(x - x0) / np.linalg.norm(x0)
    def rel_err_center(x, x0, order=2):
        window = slice(N // 4, N - N // 4)
        x_im_cropped = x.reshape((N, N))[window, window]
        x0_im_cropped = x0.reshape((N, N))[window, window]
        return (np.linalg.norm(x_im_cropped.reshape(-1) - x0_im_cropped.reshape(-1), ord=order) /
                np.linalg.norm(x0_im_cropped.reshape(-1), ord=order))

    with open(os.path.join(save_path, "smooth_comp.pkl"), "wb") as f:
        pickle.dump(smooth_signal, f)
    with open(os.path.join(save_path, "sparse_comp.pkl"), "wb") as f:
        pickle.dump(sparse_signal, f)
    with open(os.path.join(save_path, "source.pkl"), "wb") as f:
        pickle.dump(signal, f)
    with open(os.path.join(save_path, "dirty_im.pkl"), "wb") as f:
        pickle.dump(op.phi.adjoint(y), f)

    labels = ['seed', 'N', 'psnr-db',
              'lambda1', 'lambda2',
              'coupled',
              'total-time', 'opt-time',
              'objective-value', 'data-fid', 'regul-1', 'regul-2',
              'sparse-err-l1', 'sparse-err-l2', 'smooth-err-l1', 'smooth-err-l2',
              'smooth-err-l1-center', 'smooth-err-l2-center']
    final_res = []

    for l2f in lambda2f:
        path_l2 = os.path.join(save_path, f"lambda2_{l2f:f}")
        if not os.path.exists(path_l2):
            os.makedirs(path_l2)

        lambda2 = l2f * N ** 2  # Using that svd(A) = N
        lambda2 /= lap.lipschitz ** 2
        Ml2 = DiagonalOp(lambda2 * B_vec / (vec + lambda2 * B_vec))
        lambda1_max = np.abs(op.phi.adjoint(Ml2.adjoint(y))).max()

        for l1f in lambda1f:
            path_l1 = os.path.join(path_l2, f"lambda1_{l1f:6f}")
            if not os.path.exists(path_l1):
                os.makedirs(path_l1)

            lambda1 = l1f * lambda1_max

            if do_coupled:
                coupled = True
                start_coupled = time.time()
                (x1_coupled, x2_coupled), t_coupled = solve(y, op, lambda1, lambda2, coupled, laplacian, eps=eps)
                total_t_coupled = time.time() - start_coupled

            # Decoupled
            coupled = False
            start_decoupled = time.time()
            (x1_decoupled, x2_decoupled), t_decoupled = solve(y, op, lambda1, lambda2, coupled, laplacian, eps=eps)
            total_t_decoupled = time.time() - start_decoupled

            ## Analysis

            # Save metrics
            if do_coupled:
                sol_coupled = [seed, N, psnrdb, l1f, l2f, True, total_t_coupled, t_coupled]
                func_values = [data_fidelity(x1_coupled, x2_coupled), lambda1 * l1_norm(x1_coupled),
                               lambda2 * l2_gen_norm(x2_coupled)]
                sol_coupled.append(sum(func_values))
                sol_coupled += func_values
                sol_coupled += [rel_l1_err(x1_coupled, sparse_signal.reshape(-1)), rel_l2_err(x1_coupled, sparse_signal.reshape(-1)),
                                rel_l1_err(x2_coupled, smooth_signal.reshape(-1)), rel_l2_err(x2_coupled, smooth_signal.reshape(-1)),
                                rel_err_center(x2_coupled, smooth_signal.reshape(-1), order=1), rel_err_center(x2_coupled, smooth_signal.reshape(-1), order=2)]
                final_res.append(sol_coupled)


            sol_decoupled = [seed, N, psnrdb, l1f, l2f, False, total_t_decoupled, t_decoupled]
            func_values = [data_fidelity(x1_decoupled, x2_decoupled), lambda1 * l1_norm(x1_decoupled),
                           lambda2 * l2_gen_norm(x2_decoupled)]
            sol_decoupled.append(sum(func_values))
            sol_decoupled += func_values
            sol_decoupled += [rel_l1_err(x1_decoupled, sparse_signal.reshape(-1)), rel_l2_err(x1_decoupled, sparse_signal.reshape(-1)),
                              rel_l1_err(x2_decoupled, smooth_signal.reshape(-1)), rel_l2_err(x2_decoupled, smooth_signal.reshape(-1)),
                              rel_err_center(x2_decoupled, smooth_signal.reshape(-1), order=1), rel_err_center(x2_decoupled, smooth_signal.reshape(-1), order=2)]

            final_res.append(sol_decoupled)

            # Save the images
            if do_coupled:
                sparse_rcstr_coupled, smooth_rcstr_coupled, signal_rcstr_coupled = map(
                    lambda x: x.reshape((N, N)), (x1_coupled, x2_coupled, x1_coupled + x2_coupled)
                )
                sparse_rcstr_decoupled, smooth_rcstr_decoupled, signal_rcstr_decoupled = map(
                    lambda x: x.reshape((N, N)), (x1_decoupled, x2_decoupled, x1_decoupled + x2_decoupled)
                )

                f = compare3([sparse_rcstr_coupled, sparse_rcstr_decoupled, sparse_signal],
                             names=["Coupled", "Decoupled", "Source"],
                             title="Sparse component, " + r"($\lambda_1=$" + str(
                                 lambda1) + r" & $\lambda_2=$" + f"{lambda2})")
                f.savefig(os.path.join(path_l1, "sparse_comp.png"))
                plt.close(f)
                f = compare3([smooth_rcstr_coupled, smooth_rcstr_decoupled, smooth_signal],
                             names=["Coupled", "Decoupled", "Source"],
                             title="Smooth component, " + r"($\lambda_1=$" + str(
                                 lambda1) + r" & $\lambda_2=$" + f"{lambda2})")
                f.savefig(os.path.join(path_l1, "smooth_comp.png"))
                plt.close(f)
                f = compare3([signal_rcstr_coupled, signal_rcstr_decoupled, signal],
                             names=["Coupled", "Decoupled", "Source"],
                             title="Total signal, " + r"($\lambda_1=$" + str(lambda1) + r" & $\lambda_2=$" + f"{lambda2})")
                f.savefig(os.path.join(path_l1, "signal.png"))
                plt.close(f)

            # for method, sol in zip(["coupled", "decoupled"], [[x1_coupled, x2_coupled], [x1_decoupled, x2_decoupled]]):
            #     for comp, x in zip(['sparse', 'smooth'], sol):
            #         with open(os.path.join(path_l1, method + '_' + comp + '_comp.pkl'), "wb") as f:
            #             pickle.dump(x, f)

            if do_coupled:
                for comp, x in zip(['sparse', 'smooth'], [x1_coupled, x2_coupled]):
                    with open(os.path.join(path_l1, 'coupled' + '_' + comp + '_comp.pkl'), "wb") as f:
                        pickle.dump(x, f)
            for comp, x in zip(['sparse', 'smooth'], [x1_decoupled, x2_decoupled]):
                with open(os.path.join(path_l1, 'decoupled' + '_' + comp + '_comp.pkl'), "wb") as f:
                    pickle.dump(x, f)

    df = pd.DataFrame(final_res, columns=labels)
    df.to_csv(os.path.join(save_path, "results.csv"), index=False)


