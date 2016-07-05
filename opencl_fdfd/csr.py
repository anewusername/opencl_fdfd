import numpy
from numpy.linalg import norm
import pyopencl
import pyopencl.array

import time


from . import ops


class CSRMatrix(object):
    row_ptr = None      # type: pyopencl.array.Array
    col_ind = None      # type: pyopencl.array.Array
    data = None         # type: pyopencl.array.Array

    def __init__(self, queue, m):
        self.row_ptr = pyopencl.array.to_device(queue, m.indptr)
        self.col_ind = pyopencl.array.to_device(queue, m.indices)
        self.data = pyopencl.array.to_device(queue, m.data.astype(numpy.complex128))


def cg(a, b, max_iters=10000, err_thresh=1e-6, context=None, queue=None, verbose=False):
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

        if errs[-1] < err_thresh:
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

    print('Post-everything residual:', norm(a @ x - b) / norm(b))
    return x

