"""
 opencl_fdfd OpenCL 3D FDFD solver

 opencl_fdfd is a 3D Finite Difference Frequency Domain (FDFD) electromagnetic
  solver implemented in Python and OpenCL.

  Its capabilities include:
  - Arbitrary distributions of the following:
    - Dielectric constant (epsilon)
    - Magnetic permeabilty (mu)
    - Perfect electric conductor (PEC)
    - Perfect magnetic conductor (PMC)
  - Variable-sized rectangular grids
    - Stretched-coordinate PMLs (complex cell sizes allowed)

  Currently, only periodic boundary conditions are included.
  PEC/PMC boundaries can be implemented by drawing PEC/PMC cells near the edges.
  Bloch boundary conditions are not included but wouldn't be very hard to add.

  The default solver (opencl_fdfd.cg_solver(...)) located in main.py implements
   the E-field wave operator directly (ie, as a list of OpenCL instructions
   rather than a matrix). Additionally, there is a slower (and slightly more
   versatile) solver in csr.py which attempts to solve an arbitrary sparse
   matrix in compressed sparse row (CSR) format using the same conjugate gradient
   method as the default solver. The CSR solver is significantly slower, but can
   be very useful for testing alternative formulations of the FDFD wave equation.

  Currently, this solver only uses a single GPU or other OpenCL accelerator; generalization
  to multiple GPUs should be pretty straightforward (ie, just copy over edge values during the
  matrix multiplication step).


  Dependencies:
    - meanas    ( https://mpxd.net/code/jan/meanas )
    - numpy
    - pyopencl
    - jinja2
"""

from .main import cg_solver

__author__ = 'Jan Petykiewicz'
__version__ = '0.4'
version = __version__
