from typing import List, Dict, Any
import time

import numpy
from numpy.linalg import norm
import pyopencl
import pyopencl.array

import fdfd_tools.operators

from . import ops


class CSRMatrix(object):
    """
    Matrix stored in Compressed Sparse Row format, in GPU RAM.
    """
    row_ptr = None      # type: pyopencl.array.Array
    col_ind = None      # type: pyopencl.array.Array
    data = None         # type: pyopencl.array.Array

    def __init__(self,
                 queue: pyopencl.CommandQueue,
                 m: 'scipy.sparse.csr_matrix'):
        self.row_ptr = pyopencl.array.to_device(queue, m.indptr)
        self.col_ind = pyopencl.array.to_device(queue, m.indices)
        self.data = pyopencl.array.to_device(queue, m.data.astype(numpy.complex128))


def cg(a: 'scipy.sparse.csr_matrix',
       b: numpy.ndarray,
       max_iters: int = 10000,
       err_threshold: float = 1e-6,
       context: pyopencl.Context = None,
       queue: pyopencl.CommandQueue = None,
       verbose: bool = False,
       ) -> numpy.ndarray:
    """
    General conjugate-gradient solver for sparse matrices, where A @ x = b.

    :param a: Matrix to solve (CSR format)
    :param b: Right-hand side vector (dense ndarray)
    :param max_iters: Maximum number of iterations
    :param err_threshold: Error threshold for successful solve, relative to norm(b)
    :param context: PyOpenCL context. Will be created if not given.
    :param queue: PyOpenCL command queue. Will be created if not given.
    :param verbose: Whether to print statistics to screen.
    :return: Solution vector x; returned even if solve doesn't converge.
    """

    start_time = time.perf_counter()

    if context is None:
        context = pyopencl.create_some_context(False)

    if queue is None:
        queue = pyopencl.CommandQueue(context)

    def load_field(v, dtype=numpy.complex128):
        return pyopencl.array.to_device(queue, v.astype(dtype))

    r = load_field(b)
    x = pyopencl.array.zeros_like(r)
    v = pyopencl.array.zeros_like(r)
    p = pyopencl.array.zeros_like(r)

    alpha = 1.0 + 0j
    rho = 1.0 + 0j
    errs = []

    m = CSRMatrix(queue, a)

    '''
    Generate OpenCL kernels
    '''
    a_step = ops.create_a_csr(context)
    xr_step = ops.create_xr_step(context)
    rhoerr_step = ops.create_rhoerr_step(context)
    p_step = ops.create_p_step(context)
    dot = ops.create_dot(context)

    '''
    Start the solve
    '''
    start_time2 = time.perf_counter()

    _, err2 = rhoerr_step(r, [])
    b_norm = numpy.sqrt(err2)
    print('b_norm check: ', b_norm)

    success = False
    for k in range(max_iters):
        if verbose:
            print('[{:06d}] rho {:.4} alpha {:4.4}'.format(k, rho, alpha), end=' ')

        rho_prev = rho
        e = xr_step(x, p, r, v, alpha, [])
        rho, err2 = rhoerr_step(r, e)

        errs += [numpy.sqrt(err2) / b_norm]

        if verbose:
            print('err', errs[-1])

        if errs[-1] < err_threshold:
            success = True
            break

        e = p_step(p, r, rho/rho_prev, [])
        e = a_step(v, m, p, e)
        alpha = rho / dot(p, v, e)

        if k % 1000 == 0:
            print(k)

    '''
    Done solving
    '''
    time_elapsed = time.perf_counter() - start_time

    x = x.get()

    if success:
        print('Success', end='')
    else:
        print('Failure', end=', ')
    print(', {} iterations in {} sec: {} iterations/sec \
                  '.format(k, time_elapsed, k / time_elapsed))
    print('final error', errs[-1])
    print('overhead {} sec'.format(start_time2 - start_time))

    print('Final residual:', norm(a @ x - b) / norm(b))
    return x


def cg_solver(omega: complex,
              dxes: List[List[numpy.ndarray]],
              J: numpy.ndarray,
              epsilon: numpy.ndarray,
              mu: numpy.ndarray = None,
              pec: numpy.ndarray = None,
              pmc: numpy.ndarray = None,
              adjoint: bool = False,
              solver_opts: Dict[str, Any] = None,
              ) -> numpy.ndarray:
    """
    Conjugate gradient FDFD solver using CSR sparse matrices, mainly for
     testing and development since it's much slower than the solver in main.py.

    All ndarray arguments should be 1D arrays. To linearize a list of 3 3D ndarrays,
     either use fdfd_tools.vec() or numpy:
     f_1D = numpy.hstack(tuple((fi.flatten(order='F') for fi in [f_x, f_y, f_z])))

    :param omega: Complex frequency to solve at.
    :param dxes: [[dx_e, dy_e, dz_e], [dx_h, dy_h, dz_h]] (complex cell sizes)
    :param J: Electric current distribution (at E-field locations)
    :param epsilon: Dielectric constant distribution (at E-field locations)
    :param mu: Magnetic permeability distribution (at H-field locations)
    :param pec: Perfect electric conductor distribution
        (at E-field locations; non-zero value indicates PEC is present)
    :param pmc: Perfect magnetic conductor distribution
        (at H-field locations; non-zero value indicates PMC is present)
    :param adjoint: If true, solves the adjoint problem.
    :param solver_opts: Passed as kwargs to opencl_fdfd.csr.cg(**solver_opts)
    :return: E-field which solves the system.
    """

    if solver_opts is None:
        solver_opts = dict()

    b0 = -1j * omega * J
    A0 = fdfd_tools.operators.e_full(omega, dxes, epsilon=epsilon, mu=mu, pec=pec, pmc=pmc)

    Pl, Pr = fdfd_tools.operators.e_full_preconditioners(dxes)

    if adjoint:
        A = (Pl @ A0 @ Pr).H
        b = Pr.H @ b0
    else:
        A = Pl @ A0 @ Pr
        b = Pl @ b0

    x = cg(A.tocsr(), b, **solver_opts)

    if adjoint:
        x0 = Pl.H @ x
    else:
        x0 = Pr @ x

    return x0
