import numpy as np
import time
from pyxu.abc import LinOp, QuadraticFunc
from pyxu.operator import (
    SquaredL2Norm,
    L1Norm,
    hstack,
    NullFunc,
    IdentityOp,
    DiagonalOp,
    Laplacian,
)
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError
from src.operators import NuFFT


def solve(
    y: np.ndarray,
    op: NuFFT,
    lambda1: float,
    lambda2: float,
    coupled: bool,
    laplacian: bool,
    eps: float = 1e-4,
    history: bool = False,
):
    """
    Solve the optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        coupled (bool): If True, solve a coupled optimization problem; otherwise, solve a decoupled one.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.
        eps (float, optional): Tolerance for the stopping criterion.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
    """
    # lambda_max = np.linalg.norm(op.phi.adjoint(y), ord=np.inf)
    # lambda2 *= lambda_max
    # lambda1 *= lambda_max

    if coupled:
        return coupled_solve(y, op, lambda1, lambda2, laplacian, eps=eps, history=history)
    else:
        return decoupled_solve(y, op, lambda1, lambda2, laplacian, eps=eps, history=history)


def coupled_solve(
    y: np.ndarray,
    op: NuFFT,
    lambda1: float,
    lambda2: float,
    laplacian: bool,
    eps: float = 1e-4,
    history: bool = False,
) -> tuple:
    """
    Solve the coupled optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.
        eps (float, optional): Tolerance for the stopping criterion.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
        float: The time taken to solve the optimization problem.
    """

    print("Coupled")

    l22_loss = (1 / 2) * SquaredL2Norm(dim=op.dim_out).asloss(y)
    F = l22_loss * hstack([op.phi, op.phi])

    if lambda2 != 0.0:
        if laplacian:
            l2operator = Laplacian((op.N, op.N), mode="wrap")
            L = lambda2 / 2 * SquaredL2Norm(l2operator.shape[0]) * l2operator
        else:
            L = lambda2 / 2 * SquaredL2Norm(op.dim_in)

        F = F + hstack([NullFunc(op.dim_in), L])
    # F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")
    F.diff_lipschitz = 2 * op.phi.lipschitz ** 2

    if lambda1 == 0.0:
        G = NullFunc(2 * op.dim_in)
    else:
        G = hstack([lambda1 * L1Norm(op.dim_in), NullFunc(op.dim_in)])
    pgd = PGD(f=F, g=G, verbosity=500)
    sc = MaxIter(n=10) & RelError(eps=eps)
    start = time.time()
    pgd.fit(x0=np.zeros(2 * op.dim_in), stop_crit=sc)
    x = pgd.solution()
    x1 = x[: op.dim_in]
    x2 = x[op.dim_in:]
    time_solve = time.time() - start
    if history:
        _, hist = pgd.stats()
        return (x1, x2), time_solve, hist
    return (x1, x2), time_solve


def decoupled_solve(
    y: np.ndarray,
    op: NuFFT,
    lambda1: float,
    lambda2: float,
    laplacian: bool,
    eps: float = 1e-4,
    history: bool = False,
) -> tuple:
    """
    Solve the decoupled optimization problem.

    Args:
        y (np.ndarray): Input data.
        op (NuFFT): A NuFFT object.
        lambda1 (float): Regularization parameter for the L1 norm.
        lambda2 (float): Regularization parameter for the L2 norm.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.
        eps (float, optional): Tolerance for the stopping criterion.

    Returns:
        tuple: A tuple containing two NumPy arrays - x1 and x2, which are solutions to the optimization problem.
        float: The time taken to solve the optimization problem.
    """

    print("Decoupled")

    Q_Linop, compute_x2 = Op_x2(op, lambda2, laplacian)

    l22_loss = QuadraticFunc((1, op.dim_out), Q=Q_Linop).asloss(y)
    F = l22_loss * op.phi
    # F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")
    F.diff_lipschitz = .5 * op.phi.lipschitz ** 2 * Q_Linop.lipschitz

    if lambda1 == 0.0:
        G = NullFunc(op.dim_in)
    else:
        G = lambda1 * L1Norm(op.dim_in)
    pgd = PGD(f=F, g=G, verbosity=500)
    sc = MaxIter(n=10) & RelError(eps=eps)
    start = time.time()
    pgd.fit(x0=np.zeros(op.dim_in), stop_crit=sc)
    x1 = pgd.solution()
    x2 = compute_x2(x1, y)
    time_solve = time.time() - start
    if history:
        _, hist = pgd.stats()
        return (x1, x2), time_solve, hist
    return (x1, x2), time_solve


def Op_x2(op, lambda2, laplacian):
    """
    Compute the linear operator Q_Linop and a function compute_x2 to compute x2.

    Args:
        op: NuFFT object.
        lambda2 (float): Regularization parameter for the L2 norm.
        laplacian (bool): If True, the Laplacian is included in the L2 regularization.

    Returns:
        tuple: A tuple containing Q_Linop (linear operator) and compute_x2 (function to compute x2).
    """

    # Co-Gram operator = Identity ?
    # diagonal value of the Co-Gram operator when the forward operator is the NUFFT implemented with Pyxu.
    # Corresponds to N^2 Identity for complex numbers
    vec = np.array([op.dim_in, 1e-10, *[op.dim_in / 2] * (op.dim_out - 2)])

    # Does it really work ??? Should be better to not put a test here
    # diag_op = DiagonalOp(vec)
    # random_y = np.random.rand(op.dim_out)
    # cogram_id = np.allclose(op.phi.cogram().apply(random_y), diag_op.apply(random_y))

    cogram_id = isinstance(op, NuFFT) and op.on_grid

    if cogram_id:  # Co-Gram operator = Identity
        print("Co-Gram Identity")
        if laplacian:
            B_vec = (1 / vec) * FFT_L_gram_vec(op)  # F, which is diagonal
            Q_Linop = DiagonalOp(lambda2 * B_vec / (vec + lambda2 * B_vec))
        else:
            vec = 1 / (
                np.array([op.dim_in, 1e-12, *[op.dim_in / 2] * (op.dim_out - 2)])
                + lambda2 * np.ones(op.dim_out)
            )
            Q_Linop = DiagonalOp(lambda2 * vec)

    else:  # Co-Gram operator ≠ Identity
        #  Force explicit computation of the operator so that it does not get calculated twice
        if laplacian:
            l2operator = Laplacian((op.N, op.N), mode="wrap")
            B = op.phi.cogram().dagger(damp=0) * op.phi * l2operator.gram() * op.phi.T
            Q_Linop = lambda2 * LinOp.from_array(
                (B * (op.phi.cogram() + lambda2 * B).dagger(damp=0)).asarray()
            )
        else:
            Q_Linop = LinOp.from_array( lambda2 *
                (op.phi.cogram() + lambda2 * IdentityOp(op.dim_out))
                .dagger(damp=0)
                .asarray()
            )

    def compute_x2(x1, y):
        if cogram_id:  # Co-Gram operator = Identity
            if laplacian:
                x2 = (-op.phi.T * DiagonalOp(1 / (vec + lambda2 * B_vec))).apply(
                    op(x1) - y
                )
            else:
                x2 = -op.phi.pinv(op.phi.apply(x1) - y, damp=lambda2)  # Could use a diagonal operator instead
        else:  # Co-Gram operator ≠ Identity
            if laplacian:
                x2 = (-op.phi.T * (op.phi.cogram() + lambda2 * B).dagger(damp=0)).apply(
                    op(x1) - y
                )
            else:
                x2 = -op.phi.pinv(op.phi.apply(x1) - y, damp=lambda2)

        return x2

    return Q_Linop, compute_x2


def FFT_L_gram_vec(op):
    """
    Compute the vectorized result of the linear operator corresponding to the Laplacian kernel sampled in the frequency domain.

    Args:
        op: NuFFT object.

    Returns:
        np.ndarray: Vector to create the diagonal linear operator.
    """
    Lkernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    Lpad = np.pad(Lkernel, ((0, op.N - 3), (0, op.N - 3)))
    Lroll = np.roll(Lpad, -1, axis=(0, 1))
    Lfft_real = np.flip(np.fft.fftshift(np.fft.rfft2(Lroll), axes=0), axis=0)

    samples = (1 / (2 * np.pi / op.N) * op.samples).astype(int)
    samples[:, 1] = samples[:, 1] + op.N // 2 - op.even
    Lvec = (op.dim_in / 2 *
            np.repeat(np.real(((Lfft_real[samples[:, 1], samples[:, 0]]) ** 2).ravel()), 2)
    )
    return Lvec

if __name__=="__main__":
    # Test on the value of the diff_lipschitz constant of the data fidelity in the decoupled approach
    import time
    N = 128
    L = .2
    theta = 0.5

    op = NuFFT(N, L, theta, on_grid=True, do_lipschitz=True)
    print(f"Lipschitz constant: {op.phi.lipschitz}")

    y = np.random.randn(op.dim_out)
    y[1] = 0.

    lambda2 = .1 * N ** 2  # Using that svd(A) = N
    start = time.time()
    lap = Laplacian((N, N), mode="wrap")
    lap_time = time.time() - start
    print(f"Instantiation time of the Laplacian: {lap_time}")
    start = time.time()
    lap.lipschitz = 8  #lap.estimate_lipschitz(method='svd')  # Is this long ?? yes, and equal to 8 always in 2D
    lap_lips_time = time.time() - start
    print(f"Lipschitz constant of the Laplacian: {lap.lipschitz}")
    print(f"Time to estimate the Lipschitz constant of the Laplacian: {lap_lips_time}")
    lambda2 /= lap.lipschitz ** 2

    # Decoupled case
    start = time.time()
    Q_Linop, compute_x2 = Op_x2(op, lambda2, laplacian=True)
    Q_Linop_time = time.time() - start
    print(f"Q_linop time: {Q_Linop_time}")
    print("Q_linop Lipschitz:")
    print(Q_Linop.lipschitz)

    l22_loss = QuadraticFunc((1, op.dim_out), Q=Q_Linop).asloss(y)
    F = l22_loss * op.phi
    print("Diff Lipschitz constant")
    print(f"\tDefault value at instantiation: {F.diff_lipschitz:.3e}")
    print(f"\tProduct of the lipschitz: {op.phi.lipschitz**2 * Q_Linop.lipschitz:.3e}")
    start = time.time()
    lips = F.estimate_diff_lipschitz(method='svd')
    F_lips_time = time.time() - start
    print(f"\tComputed value: {lips:.3e}")
    print(f"\t2 times computed value: {2 * lips:.3e}")
    # Computed value is .5 times the product of the lipschitz --> Why ???
    print(f"\tTime to estimate diff lipschitz: {F_lips_time}")

    # Coupled case
    print("Coupled scenario:")
    l22_loss = (1 / 2) * SquaredL2Norm(dim=op.dim_out).asloss(y)
    F = l22_loss * hstack([op.phi, op.phi])
    L = lambda2 / 2 * SquaredL2Norm(lap.shape[0]) * lap
    # F = F + hstack([NullFunc(op.dim_in), L])
    start = time.time()
    df_lips = F.estimate_diff_lipschitz(method="svd")
    diff_lips_time = time.time() - start
    print(f"\tAuto lip constant of the stack: {hstack([op.phi, op.phi]).lipschitz:.3e}")
    print(f"\tComputed lip constant of the stack: {hstack([op.phi, op.phi]).estimate_lipschitz(method='svd'):.3e}")  # np.sqrt(2) * N

    print(f"\tAuto calculated value at instantiation: {F.diff_lipschitz:.3e}")  # already good
    print(f"\tProduct of the lipschitz: {op.phi.lipschitz**2 * 2:.3e}")  # 2 * N**2

    print(f"\tDiff Lipschitz constant of the coupled approach (computed): {df_lips:.3e}")
    print(f"\tTime to estimate diff lipschitz of the coupled approach: {diff_lips_time:.3f}")



