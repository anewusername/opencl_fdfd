"""
Default FDFD solver

This file holds the default FDFD solver, which uses an E-field wave
operator implemented directly as OpenCL arithmetic (rather than as
a matrix).
"""

from typing import List, Optional, cast
import time
import logging

import numpy
from numpy.typing import NDArray, ArrayLike
from numpy.linalg import norm
import pyopencl
import pyopencl.array

import meanas.fdfd.operators

from . import ops


__author__ = 'Jan Petykiewicz'

logger = logging.getLogger(__name__)


def cg_solver(
        omega: complex,
        dxes: List[List[NDArray]],
        J: ArrayLike,
        epsilon: ArrayLike,
        mu: Optional[ArrayLike] = None,
        pec: Optional[ArrayLike] = None,
        pmc: Optional[ArrayLike] = None,
        adjoint: bool = False,
        max_iters: int = 40000,
        err_threshold: float = 1e-6,
        context: Optional[pyopencl.Context] = None,
        ) -> NDArray:
    """
    OpenCL FDFD solver using the iterative conjugate gradient (cg) method
     and implementing the diagonalized E-field wave operator directly in
     OpenCL.

    All ndarray arguments should be 1D arrays. To linearize a list of 3 3D ndarrays,
     either use meanas.fdmath.vec() or numpy:
     f_1D = numpy.hstack(tuple((fi.flatten(order='F') for fi in [f_x, f_y, f_z])))

    Args:
        omega: Complex frequency to solve at.
        dxes: [[dx_e, dy_e, dz_e], [dx_h, dy_h, dz_h]] (complex cell sizes)
        J: Electric current distribution (at E-field locations)
        epsilon: Dielectric constant distribution (at E-field locations)
        mu: Magnetic permeability distribution (at H-field locations)
        pec: Perfect electric conductor distribution
            (at E-field locations; non-zero value indicates PEC is present)
        pmc: Perfect magnetic conductor distribution
            (at H-field locations; non-zero value indicates PMC is present)
        adjoint: If true, solves the adjoint problem.
        max_iters: Maximum number of iterations. Default 40,000.
        err_threshold: If (r @ r.conj()) / norm(1j * omega * J) < err_threshold, success.
            Default 1e-6.
        context: PyOpenCL context to run in. If not given, construct a new context.

    Returns:
        E-field which solves the system. Returned even if we did not converge.
    """
    start_time = time.perf_counter()

    shape = [dd.size for dd in dxes[0]]

    b = -1j * omega * numpy.array(J, copy=False)

    '''
        ** In this comment, I use the following notation:
           M* = conj(M),
           M.T = transpose(M),
           M' = ctranspose(M),
           M N = dot(M, N)

        This solver uses a symmetrized wave operator M = (L A R) = (L A R).T
         (where L = inv(R) are diagonal preconditioner matrices) when
         solving the wave equation; therefore, it solves the problem
            M y = d
         => (L A R) (inv(R) x) = (L b)
         => A x = b
         with x = R y

        From the fact that M is symmetric, we can write
         (L A R)* = M* = M' = (L A R)' = R' A' L' = R* A' L*
        We obtain M* by conjugating all of our arguments (except J).

        Then we solve
         (R* A' L*) v = (R* b)
        and obtain x:
          x = L* v

        We can accomplish all this simply by conjugating everything (except J) and
         reversing the order of L and R
    '''
    epsilon = numpy.array(epsilon, copy=False)
    if adjoint:
        # Conjugate everything
        dxes = [[numpy.conj(dd) for dd in dds] for dds in dxes]
        omega = numpy.conj(omega)
        epsilon = numpy.conj(epsilon)
        if mu is not None:
            mu = numpy.conj(mu)

    L, R = meanas.fdfd.operators.e_full_preconditioners(dxes)

    if adjoint:
        b_preconditioned = R @ b
    else:
        b_preconditioned = L @ b

    '''
        Allocate GPU memory and load in data
    '''
    if context is None:
        context = pyopencl.create_some_context(interactive=True)

    queue = pyopencl.CommandQueue(context)

    def load_field(v, dtype=numpy.complex128):
        return pyopencl.array.to_device(queue, v.astype(dtype))

    r = load_field(b_preconditioned)  # load preconditioned b into r
    H = pyopencl.array.zeros_like(r)
    x = pyopencl.array.zeros_like(r)
    v = pyopencl.array.zeros_like(r)
    p = pyopencl.array.zeros_like(r)

    alpha = 1.0 + 0j
    rho = 1.0 + 0j
    errs = []

    inv_dxes = [[load_field(1 / numpy.array(dd, copy=False)) for dd in dds] for dds in dxes]
    oeps = load_field(-omega ** 2 * epsilon)
    Pl = load_field(L.diagonal())
    Pr = load_field(R.diagonal())

    if mu is None:
        invm = load_field(numpy.array([]))
    else:
        invm = load_field(1 / numpy.array(mu, copy=False))
        mu = numpy.array(mu, copy=False)

    if pec is None:
        gpec = load_field(numpy.array([]), dtype=numpy.int8)
    else:
        gpec = load_field(numpy.array(pec, dtype=bool, copy=False), dtype=numpy.int8)

    if pmc is None:
        gpmc = load_field(numpy.array([]), dtype=numpy.int8)
    else:
        gpmc = load_field(numpy.array(pmc, dtype=bool, copy=False), dtype=numpy.int8)

    '''
    Generate OpenCL kernels
    '''
    has_mu, has_pec, has_pmc = [q is not None for q in (mu, pec, pmc)]

    a_step_full = ops.create_a(context, shape, has_mu, has_pec, has_pmc)
    xr_step = ops.create_xr_step(context)
    rhoerr_step = ops.create_rhoerr_step(context)
    p_step = ops.create_p_step(context)
    dot = ops.create_dot(context)

    def a_step(E, H, p, events):
        return a_step_full(E, H, p, inv_dxes, oeps, invm, gpec, gpmc, Pl, Pr, events)

    '''
    Start the solve
    '''
    start_time2 = time.perf_counter()

    _, err2 = rhoerr_step(r, [])
    b_norm = numpy.sqrt(err2)
    logging.debug(f'b_norm check: {b_norm}')

    success = False
    for k in range(max_iters):
        do_print = (k % 100 == 0)
        if do_print:
            logger.debug(f'[{k:06d}] rho {rho:.4} alpha {alpha:4.4}')

        rho_prev = rho
        e = xr_step(x, p, r, v, alpha, [])
        rho, err2 = rhoerr_step(r, e)

        errs += [numpy.sqrt(err2) / b_norm]

        if do_print:
            logger.debug(f'err {errs[-1]}')

        if errs[-1] < err_threshold:
            success = True
            break

        e = p_step(p, r, rho/rho_prev, [])
        e = a_step(v, H, p, e)
        alpha = rho / dot(p, v, e)

        if k % 1000 == 0:
            logger.info(f'iteration {k}')

    '''
    Done solving
    '''
    time_elapsed = time.perf_counter() - start_time

    # Undo preconditioners
    if adjoint:
        x = (Pl * x).get()
    else:
        x = (Pr * x).get()

    if success:
        logger.info('Solve success')
    else:
        logger.warning('Solve failure')
    logger.info(f'{k} iterations in {time_elapsed} sec: {k / time_elapsed} iterations/sec')
    logger.debug(f'final error {errs[-1]}')
    logger.debug(f'overhead {start_time2 - start_time} sec')

    A0 = meanas.fdfd.operators.e_full(omega, dxes, epsilon, mu).tocsr()
    if adjoint:
        # Remember we conjugated all the contents of A earlier
        A0 = A0.T

    residual = norm(A0 @ x - b) / norm(b)
    logger.info(f'Post-everything residual: {residual}')
    return x

