# opencl_fdfd

**opencl_fdfd** is a 3D Finite Difference Frequency Domain (FDFD)
electromagnetic solver implemented in Python and OpenCL.


**Capabilities:**
* Arbitrary distributions of the following:
    * Dielectric constant (`epsilon`)
    * Magnetic permeabilty (`mu`)
    * Perfect electric conductor (`PEC`)
    * Perfect magnetic conductor (`PMC`)
* Variable-sized rectangular grids
    * Stretched-coordinate PMLs (complex cell sizes allowed)

Currently, only periodic boundary conditions are included.
PEC/PMC boundaries can be implemented by drawing PEC/PMC cells near the edges.
Bloch boundary conditions are not included but wouldn't be very hard to add.

The default solver `opencl_fdfd.cg_solver(...)` located in main.py
implements the E-field wave operator directly (ie, as a list of OpenCL
instructions rather than a matrix). Additionally, there is a slower
(and slightly more versatile) solver in `csr.py` which attempts to solve
an arbitrary sparse matrix in compressed sparse row (CSR) format using
the same conjugate gradient method as the default solver. The CSR solver
is significantly slower, but can be very useful for testing alternative
formulations of the FDFD electromagnetic wave equation.

Currently, this solver only uses a single GPU or other OpenCL accelerator;
generalization to multiple GPUs should be pretty straightforward
(ie, just copy over edge values during the matrix multiplication step).


## Installation

**Dependencies:**
* python 3 (written and tested with 3.7)
* numpy
* pyopencl
* jinja2
* [meanas](https://mpxd.net/code/jan/meanas) (>=0.5)


Install with pip, via git:
```bash
pip install git+https://mpxd.net/code/jan/opencl_fdfd.git@release
```


## Use

See the documentation for `opencl_fdfd.cg_solver(...)`
(located in ```main.py```) for details about how to call the solver.
The FDFD arguments are identical to those in
`meanas.solvers.generic(...)`, and a few solver-specific
arguments are available.

An alternate (slower) FDFD solver and a general gpu-based sparse matrix
solver is available in `csr.py`. These aren't particularly
well-optimized, and something like
[MAGMA](http://icl.cs.utk.edu/magma/index.html) would probably be a
better choice if you absolutely need to solve arbitrary sparse matrices
and can tolerate writing and compiling C/C++ code. Still, they're
usually quite a bit faster than the scipy.linalg solvers.
