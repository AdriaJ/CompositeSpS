import os
import pandas as pd
import matplotlib.pyplot as plt

from src import *

plt.rc('text', usetex=False)
# plt.rc('text.latex', unicode = False)
plt.rc('svg', fonttype='none')

filename = "time_bench1e-4.csv"

if __name__ == "__main__":
    print(plt.rcParams['svg.fonttype'])

    plt.style.use('default')
    df = pd.read_csv(os.path.join(os.getcwd(), '..', 'exps', filename))

    df_coupled = df[df['coupled']]
    df_decoupled = df[df['coupled'] == False]
    Ns = df['N'].unique()

    s = 3
    fig, ax = plt.subplots(1, 1, figsize=(s * 1.6, s))

    ax.plot(Ns ** 2, df_coupled.groupby('N')['solving time'].median(), label=f"Coupled", marker='o', c='tab:blue')
    ax.fill_between(Ns ** 2, df_coupled.groupby('N')['solving time'].quantile(0.25),
                    df_coupled.groupby('N')['solving time'].quantile(0.75), alpha=0.5, color='tab:blue')
    ax.plot(Ns ** 2, df_decoupled.groupby('N')['solving time'].median(), label=f"Decoupled", marker='o', c='tab:orange')
    ax.fill_between(Ns ** 2, df_decoupled.groupby('N')['solving time'].quantile(0.25),
                    df_decoupled.groupby('N')['solving time'].quantile(0.75), alpha=0.5, color='tab:orange')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Image size (pixels)", fontsize=11)
    ax.set_ylabel("Solving time (s)", fontsize=11)
    ax.tick_params(labelsize=11)
    # ax.set_xticks(fontsize=11)
    # ax.set_title("Time performances")
    ax.legend(fontsize=11)
    plt.subplots_adjust(top=0.88,
                        bottom=0.19,
                        left=0.15,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    # plt.show()
    plt.savefig('testfig1.pdf')


