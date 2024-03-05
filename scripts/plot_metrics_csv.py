import matplotlib.pyplot as plt
import os
import pandas as pd


N = 128
psnrdb = 20

name = f"psnrdb{psnrdb:d}_N{N:d}"  #  f"noiseless{N:d}"

if __name__ == "__main__":
    rootdir = os.path.join("/home/jarret/PycharmProjects/CompositeSpS",
                           "exps",
                           name,)

    df = pd.read_csv(os.path.join(rootdir, "results.csv"))
    coupled = df[df["coupled"] == True]
    decoupled = df[df["coupled"] == False]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    # Time comparison
    ax = axes[0]
    ax.set_title("Time comparison")
    ax.set_ylabel("Time (s)")
    ax.scatter(range(len(coupled)), coupled['total-time'], label="Coupled (total)", marker='x', color="tab:blue")
    ax.scatter(range(len(coupled)), coupled['opt-time'], label="Coupled (solve)", color="tab:blue", marker='.')
    ax.scatter(range(len(decoupled)), decoupled['total-time'], label="Decoupled (total)", color="tab:orange", marker='x')
    ax.scatter(range(len(decoupled)), decoupled['opt-time'], label="Decoupled (solve)", color="tab:orange", marker='.')
    ax.set_xticks(list(range(len(decoupled))), [str(c[0]) + '\n' + str(c[1]) for c in zip(decoupled[['lambda1']].values.reshape(-1), decoupled[['lambda2']].values.reshape(-1))])
    ax.legend()
    ax.set_yscale('log')

    # Error sparse comparison
    ax = axes[1]
    ax.set_title("Error comparison (sparse)")
    ax.set_ylabel("Relative error")
    ax.scatter(range(len(coupled)), coupled['sparse-err-l1'], label="Coupled (l1)", color="tab:blue", marker='x')
    ax.scatter(range(len(coupled)), coupled['sparse-err-l2'], label="Coupled (l2)", color="tab:blue", marker='.')
    ax.scatter(range(len(decoupled)), decoupled['sparse-err-l1'], label="Decoupled (l1)", color="tab:orange", marker='x')
    ax.scatter(range(len(decoupled)), decoupled['sparse-err-l2'], label="Decoupled (l2)", color="tab:orange", marker='.')
    ax.set_xticks(list(range(len(decoupled))), [str(c[0]) + '\n' + str(c[1]) for c in zip(decoupled[['lambda1']].values.reshape(-1), decoupled[['lambda2']].values.reshape(-1))])
    ax.set_ylim(top=1.05)
    ax.legend()

    # Error smooth comparison
    ax = axes[2]
    ax.set_title("Error comparison (smooth)")
    ax.set_ylabel("Relative error")
    ax.scatter(range(len(coupled)), coupled['smooth-err-l1'], label="Coupled (l1)", color="tab:blue", marker='x')
    ax.scatter(range(len(coupled)), coupled['smooth-err-l2'], label="Coupled (l2)", color="tab:blue", marker='.')
    ax.scatter(range(len(coupled)), coupled['smooth-err-l1-center'], label="Coupled - center (l1)", color="tab:blue", marker='x')
    ax.scatter(range(len(coupled)), coupled['smooth-err-l2-center'], label="Coupled - center (l2)", color="tab:blue", marker='.')
    ax.scatter(range(len(decoupled)), decoupled['smooth-err-l1'], label="Decoupled (l1)", color="tab:orange", marker='x')
    ax.scatter(range(len(decoupled)), decoupled['smooth-err-l2'], label="Decoupled (l2)", color="tab:orange", marker='.')
    ax.scatter(range(len(coupled)), coupled['smooth-err-l1-center'], label="Coupled - center (l1)", color="tab:orange", marker='x')
    ax.scatter(range(len(coupled)), coupled['smooth-err-l2-center'], label="Coupled - center (l2)", color="tab:orange", marker='.')

    ax.set_xticks(list(range(len(decoupled))), [str(c[0]) + '\n' + str(c[1]) for c in zip(decoupled[['lambda1']].values.reshape(-1), decoupled[['lambda2']].values.reshape(-1))])
    ax.set_ylim(top=1.05)
    ax.legend()

    fig.subplots_adjust(top=0.93, bottom=0.065, left=0.09, right=0.95, hspace=0.35, wspace=0.2)
    fig.suptitle(f"Image size N={N:d}")
    fig.show()
