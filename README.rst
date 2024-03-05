============
CompositeSpS
============

This python repository provides the code accompanying the paper *"A Decoupled Approach
for Composite Sparse-plus-Smooth Penalized Optimization"*.

The optimization problem considered can be stated as follows:

.. math::

    \underset{\x_1, \x_2 \in \R^N} {\arg\min} \frac{1}{2} & \lVert\y - \mathbf{A}(\x_1 + \x_2)\rVert_2^2 + \lambda_1 \lVert\L_1\x_1\rVert_1 + \frac{\lambda_2}{2} \lVert\L_2\x_2\rVert_2^2.

The code implements:

* a coupled method, that directly solves the problem above,
* a decoupled approach, that relies on a representer theorem presented in the companion article, that allows to solve two smaller dimensions problems.

We illustrate that, in the application case considered here, the decoupled approach leads to
significant time speedups, while providing a reconstruction of similar or higher quality
than the coupled method.

Installation
===========

Instructions are provided in the `src/howtoinstall.txt` file.

.. code-block:: bash

   $ git clone git@github.com:AdriaJ/CompositeSpS.git
   $ cd CompositeSpS
   $ conda create -n compsps python=3.10
   $ conda activate compsps
   $ conda install numpy matplotlib
   $ pip install finufft
   $ pip install pyxu
   $ conda install jupyter

Description of the files
========================

The folder `src/` contains the code to simulate the optimization problem, including the simulation of the
input composite signal and the sampling operator, as well as the numerical solvers, for both the coupled
and decoupled methods.

The folder `scripts/` provides the codes to reproduce the results displayed in the article, as well as other
to experiment with the decoupled approach and play with the reconstruction parameters. Additionally, scripts
are provided to plot the results in a handy and insightful manner.

Citation
========

For citing this package, please refer to the ArXiv preprint, until further notice of acceptance.

::

    @misc{jarret2024composite,
        title={{A Decoupled Approach for Composite Sparse-plus-Smooth Penalized Optimization}},
        author={Jarret, Adrian and Costa, Valérie and Fageot, Julien},
        year={2024},
        eprint={todo},
        archivePrefix={arXiv},
        primaryClass={todo}
    }