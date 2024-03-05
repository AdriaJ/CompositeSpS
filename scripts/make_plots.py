from src import *

## Parameters

# Signal size
N = 128
# rate of measurements
L = 0.3

# Reconstruction parameters
lambda1f = 8e-2
lambda2f = .5
eps = 1e-4

# Noise
psnrdb = 20

# Seed
seed = 21


def compare2(images, names: list = None, title: str = None):
    f = plt.figure(figsize=(12, 5))
    axes = f.subplots(1, 2, sharex=True, sharey=True)
    divnorm = colors.CenteredNorm(
        vcenter=0.0,
        halfrange=max([np.abs(im).max() for im in images]),
    )
    for j in range(2):
        im = axes[j].imshow(
            images[j],
            cmap="seismic",
            norm=divnorm,
            interpolation='none'
        )
        axes[j].set_yticks([])
        axes[j].set_xticks([])
        divider = make_axes_locatable(axes[j])
        if names:
            axes[j].set_title(names[j], fontsize=16)
        plt.subplots_adjust(wspace=0, left=0, right=1)
        if j == 1:
            cax = divider.append_axes(position="right", size="5%", pad=0.5)
            cbar = f.colorbar(im, cax=cax, location='right')
            cbar.ax.tick_params(labelsize=20)
            # print(cbar.ax.get_tick_params())
        elif j == 0:
            cax = divider.append_axes(position="left", size="5%", pad=0.5)
            cax.axis("off")
    if title:
        plt.suptitle(title, fontsize=16)

    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=0.94, hspace=0.14, wspace=0.11)

    return f

if __name__ == "__main__":
    # sampling parameters
    theta = 0.5
    # Sparse signal parameters
    sparse_range = (-6, 6)
    density = 0.005
    # Smooth signal parameters
    smooth_amplitude = 2
    sigmas_range = (2e-2, 2e-1)
    nb_gaussian = int(0.5 * N)
    # Use Laplacian ?
    laplacian = True

    savepath = f"../exps/figures/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

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
    y = compute_y(y0, psnrdb)

    # Regularization
    lambda2 = lambda2f * N ** 2  # Using that svd(A) = N
    lap = Laplacian((N, N), mode="wrap")
    lap.lipschitz = 8
    lambda2 /= lap.lipschitz ** 2

    vec = np.array([op.dim_in, 1e-10, *[op.dim_in / 2] * (op.dim_out - 2)])
    B_vec = (1 / vec) * FFT_L_gram_vec(op)  # Depends on the samples of the DFT
    Ml2 = DiagonalOp(lambda2 * B_vec / (vec + lambda2 * B_vec))
    lambda1_max = np.abs(op.phi.adjoint(Ml2.adjoint(y))).max()

    lambda1 = lambda1f * lambda1_max

    ## Reconstruction
    coupled = False
    start_decoupled = time.time()
    (x1_decoupled, x2_decoupled), t_decoupled = solve(y, op, lambda1, lambda2, coupled, laplacian, eps=eps)
    total_t_decoupled = time.time() - start_decoupled

    ## Show metrics
    # Reconstruction time
    print(f"Decoupled reconstruction time: {total_t_decoupled:.2f} s")
    # Error
    print("Sparse component:")
    print(f"\tRelative error (l1): {rel_l1_err(x1_decoupled, sparse_signal.reshape(-1)):.3f}")
    print(f"\tRelative error (l2): {rel_l2_err(x1_decoupled, sparse_signal.reshape(-1)):.3f}")
    print("Smooth component:")
    print(f"\tRelative error (l1): {rel_l1_err(x2_decoupled, smooth_signal.reshape(-1)):.3f}")
    print(f"\tRelative error (l2): {rel_l2_err(x2_decoupled, smooth_signal.reshape(-1)):.3f}")

    # Jaccard index
    tp = np.sum((x1_decoupled != 0) & (sparse_signal.reshape(-1) != 0))
    fp = np.sum((x1_decoupled != 0) & (sparse_signal.reshape(-1) == 0))
    fn = np.sum((x1_decoupled == 0) & (sparse_signal.reshape(-1) != 0))
    jaccard = tp / (tp + fp + fn)
    print(f"Jaccard index: {jaccard:.2f}")
    # print sparsity of the source and the reconstuction
    print(f"Sparsity of the source: {np.sum(sparse_signal != 0):d}")
    print(f"Sparsity of the reconstruction: {np.sum(x1_decoupled != 0):d}")
    print(tp, fp, fn)

    names = ["Source", "Reconstruction"]
    title = f"Factor for $\lambda_1$: {lambda1f:.2f} - Factor for $\lambda_2$: {lambda2f:.2f}"


    f = compare2([sparse_signal, x1_decoupled.reshape((N, N))])
    f.savefig(os.path.join(savepath, 'sparse_comp.pdf'), dpi=300)
    f = compare2([smooth_signal, x2_decoupled.reshape((N, N))])
    f.savefig(os.path.join(savepath, 'smooth_comp.pdf'), dpi=300)
    f = compare2([signal, (x1_decoupled + x2_decoupled).reshape((N, N))])
    f.savefig(os.path.join(savepath, 'total_reconstruction.pdf'), dpi=300)

    # images = [sparse_signal, x1_decoupled.reshape((N, N))]
    # f = plt.figure(figsize=(12, 5))
    # axes = f.subplots(1, 2, sharex=True, sharey=True)
    # divnorm = colors.CenteredNorm(
    #     vcenter=0.0,
    #     halfrange=max([np.abs(im).max() for im in images]),
    # )
    # for j in range(2):
    #     im = axes[j].imshow(
    #         images[j],
    #         cmap="seismic",
    #         norm=divnorm,
    #         interpolation='none'
    #     )
    #     axes[j].set_yticks([])
    #     axes[j].set_xticks([])
    #     divider = make_axes_locatable(axes[j])
    #     if names:
    #         axes[j].set_title(names[j], fontsize=16)
    #     plt.subplots_adjust(wspace=0, left=0, right=1)
    #     if j == 0:
    #         cax = divider.append_axes(position="left", size="5%", pad=0.5)
    #         cbar = f.colorbar(im, cax=cax, location='left')
    #         cbar.ax.tick_params(labelsize=16)
    #         # print(cbar.ax.get_tick_params())
    #     elif j == 1:
    #         cax = divider.append_axes(position="right", size="5%", pad=0.5)
    #         cax.axis("off")
    # if title:
    #     plt.suptitle(title, fontsize=16)
    #
    # plt.subplots_adjust(top=1.0, bottom=0.0, left=0.06, right=1., hspace=0.14, wspace=0.11)
    #
    # plt.show()