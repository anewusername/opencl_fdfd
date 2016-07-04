import numpy
from numpy.linalg import norm
import pyopencl
import pyopencl.array

import time

import fdfd_tools.operators

from . import ops


def cg_solver(omega, dxes, J, epsilon, mu=None, pec=None, pmc=None, adjoint=False,
              max_iters=40000, err_thresh=1e-6, context=None, verbose=False):
    start_time = time.perf_counter()

    b = -1j * omega * J

    shape = [d.size for d in dxes[0]]

    '''
        ** In this comment, I use the notation M* = conj(M),
           M.T = transpose(M), M' = ctranspose(M), M N = dot(M, N)

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
    if adjoint:
        # Conjugate everything
        dxes = [[numpy.conj(d) for d in dd] for dd in dxes]
        omega = numpy.conj(omega)
        epsilon = numpy.conj(epsilon)
        if mu is not None:
            mu = numpy.conj(mu)

    L, R = fdfd_tools.operators.e_full_preconditioners(dxes)

    if adjoint:
        b_preconditioned = R @ b
    else:
        b_preconditioned = L @ b

    '''
        Allocate GPU memory and load in data
    '''
    if context is None:
        context = pyopencl.create_some_context(False)

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

    inv_dxes = [[load_field(1 / d) for d in dd] for dd in dxes]
    oeps = load_field(-omega ** 2 * epsilon)
    Pl = load_field(L.diagonal())
    Pr = load_field(R.diagonal())

    mu = numpy.ones_like(epsilon)
    # pec = numpy.zeros_like(epsilon)
    # pmc = numpy.zeros_like(epsilon)

    if mu is None:
        invm = load_field(numpy.array([]))
    else:
        invm = load_field(1 / mu)

    if pec is None:
        gpec = load_field(numpy.array([]), dtype=numpy.int8)
    else:
        gpec = load_field(pec, dtype=numpy.int8)

    if pmc is None:
        gpmc = load_field(numpy.array([]), dtype=numpy.int8)
    else:
        gpmc = load_field(pmc, dtype=numpy.int8)

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

        if errs[-1] < err_thresh:
            success = True
            break

        e = p_step(p, r, rho/rho_prev, [])
        e = a_step(v, H, p, e)
        alpha = rho / dot(p, v, e)

        if k % 1000 == 0:
            print(k)

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
        print('Success', end='')
    else:
        print('Failure', end=', ')
    print(', {} iterations in {} sec: {} iterations/sec \
                  '.format(k, time_elapsed, k / time_elapsed))
    print('final error', errs[-1])
    print('overhead {} sec'.format(start_time2 - start_time))

    A0 = fdfd_tools.operators.e_full(omega, dxes, epsilon, mu).tocsr()
    if adjoint:
        # Remember we conjugated all the contents of A earlier
        A0 = A0.T
    print('Post-everything residual:', norm(A0 @ x - b) / norm(b))
    return x