import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import matplotlib.colors as colors

from src.utils import plot_signal

N = 128
psnrdb = 20

name = f"psnrdb{psnrdb:d}_N{N:d}"  #  f"noiseless{N:d}"

def sort_lists(X, Y):
    """
    Sorts the lists X and Y according to the order of X.
    """
    order = sorted(zip(Y, X), key=lambda pair: pair[0])
    x_sorted = [x for _, x in order]
    y_sorted = [y for y, _ in order]
    return x_sorted, y_sorted


def load_images(l2_folders, l1_folders, rootdir, N, filename):
    assert filename in ['smooth', 'sparse']

    images = []
    for l2_folder in l2_folders:
        im_l2 = []
        for l1_folder in l1_folders:
            path = os.path.join(rootdir, l2_folder, l1_folder)
            with open(os.path.join(path, "decoupled_" + filename + "_comp.pkl"), "rb") as f:
                im = pickle.load(f)
            im_l2.append(im.reshape((N, N)))
        images.append(im_l2)

    return images

def load_source(rootdir, N, filename):
    assert filename in ['smooth', 'sparse']
    with open(os.path.join(rootdir, filename + "_comp.pkl"), "rb") as f:
        im = pickle.load(f)
    return im.reshape((N, N))

def load_dirty(rootdir, N):
    with open(os.path.join(rootdir, "dirty_im.pkl"), "rb") as f:
        im = pickle.load(f)
    return im.reshape((N, N))

def fig_array_im(images, source, l1s, l2s):
    vext = max(*[np.abs(iml1).max() for imsl2 in images for iml1 in imsl2], np.abs(source).max())
    divnorm = colors.CenteredNorm(vcenter=0.0, halfrange=vext)

    fig, axes = plt.subplots(len(l1s), len(l2s), figsize=(12, 12), sharey=True, sharex=True)
    for i in range(len(l2s)):
        for j in range(len(l1s)):
            ax = axes[i, j]
            axes[i, j].set_yticks([])
            axes[i, j].set_xticks([])
            ax.imshow(images[i][j], interpolation='none', cmap='seismic', norm=divnorm)
            if i == 0:
                ax.set_title(f"l1f={l1s[j]:.3f}")
            if j == 0:
                ax.set_ylabel(f"l2f={l2s[i]:.3f}")
    fig.subplots_adjust(wspace=.05, hspace=.05, bottom=.05, left=.05, right=.95, top=.95)
    return fig


if __name__ == "__main__":
    rootdir = os.path.join("/home/jarret/PycharmProjects/CompositeSpS",
                           "exps",
                           name,)

    l2s = []
    l2_folders = []
    for name in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir, name)):
            l2f = float(name[8:])
            l2s.append(l2f)
            l2_folders.append(name)

    l1s = []
    l1_folders = []
    for name in os.listdir(os.path.join(rootdir, l2_folders[0])):
        if os.path.isdir(os.path.join(rootdir, l2_folders[0], name)):
            l1f = float(name[8:])
            l1s.append(l1f)
            l1_folders.append(name)

    l2s, l2_folders = sort_lists(l2s, l2_folders)
    l1s, l1_folders = sort_lists(l1s, l1_folders)

    # print(l2s)
    # print(l1s)
    # print(l2_folders)
    # print(l1_folders)

    ## Plotting convention

    #  # ----- l1 ----->
    #  |
    #  |
    #  l2
    #  |
    #  |
    #  V

    ## plot smooth
    smooth_images = load_images(l2_folders, l1_folders, rootdir, N, "smooth")
    smooth_source = load_source(rootdir, N, "smooth")
    fig = fig_array_im(smooth_images, smooth_source, l1s, l2s)
    fig.show()

    ## plot sparse
    sparse_images = load_images(l2_folders, l1_folders, rootdir, N, "sparse")
    sparse_source = load_source(rootdir, N, "sparse")
    fig = fig_array_im(sparse_images, sparse_source, l1s, l2s)
    fig.show()

    fig = plot_signal(sparse_source, smooth_source)
    fig.show()

    plt.figure()
    dirty = load_dirty(rootdir, N)
    divnorm = colors.CenteredNorm(vcenter=0.0, halfrange=dirty.max())
    plt.imshow(dirty, cmap="seismic", norm=divnorm, interpolation="none")
    plt.title(f"Dirty image")
    plt.colorbar()
    plt.show()

###  Observations  ###
# For N=128, noiseless:
#       We want lambda1 to be small: little constraint on sparsity
#       We want lambda2 to be large: enforce smoothness

    ## Change of the sparse components with respect to lambda 2, for lambda 1 fixed
    # s = np.array(sparse_images)
    # for j in range(s.shape[1]):
    #     print(f"Lambda1 factor: {l1s[j]:.3f}")
    #     for i in range(s.shape[0]):
    #         print(f"\tLambda2 factor: {l2s[i]:.3f}")
    #         print(f"\tSparsity: {np.count_nonzero(s[i, j])}")
    #         print(f"\tMin value: {s[i, j].min()}")
    #         print(f"\tMax value: {s[i, j].max()}")
    #         print("\n")
