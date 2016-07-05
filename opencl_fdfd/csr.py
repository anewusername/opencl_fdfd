import numpy

import pyopencl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

import time


def type_to_C(float_type: numpy.float32 or numpy.float64) -> str:
    """
    Returns a string corresponding to the C equivalent of a numpy type.

    :param float_type: numpy type: float32, float64, complex64, complex128
    :return: string containing the corresponding C type (eg. 'double')
    """
    types = {
        numpy.float32: 'float',
        numpy.float64: 'double',
        numpy.complex64: 'cfloat_t',
        numpy.complex128: 'cdouble_t',
    }
    if float_type not in types:
        raise Exception('Unsupported type')

    return types[float_type]


def create_ops(context):
    preamble = '''
    #define PYOPENCL_DEFINE_CDOUBLE
    #include <pyopencl-complex.h>
    '''

    ctype = type_to_C(numpy.complex128)

    # -------------------------------------

    spmv_source = '''
    int start = m_row_ptr[i];
    int stop = m_row_ptr[i+1];
    cdouble_t dot = cdouble_new(0.0, 0.0);

    int col_ind, d_ind;
    for (int j=start; j<stop; j++) {
        col_ind = m_col_ind[j];
        d_ind = j;

        dot = cdouble_add(dot, cdouble_mul(v_in[col_ind], m_data[d_ind]));
    }
    v_out[i] = dot;
    '''

    v_out_args = ctype + ' *v_out, int v_len_half'
    m_args = 'int m_nnz, int *m_row_ptr, int *m_col_ind, ' + ctype + ' *m_data'
    v_in_args = ctype + ' *v_in'

    spmv_kernel = ElementwiseKernel(context, operation=spmv_source, preamble=preamble,
                                    arguments=', '.join((v_out_args, m_args, v_in_args)))

    def spmv(v_out, m, v_in, e):
        return spmv_kernel(v_out, (v_out.size - 1)//2,
                           m.data.size, m.row_ptr, m.col_ind, m.data,
                           v_in, wait_for=e)

    # -------------------------------------

    update_xr_source = '''
    x[i] = cdouble_add(x[i], cdouble_mul(alpha, p[i]));
    r[i] = cdouble_sub(r[i], cdouble_mul(alpha, v[i]));
    '''

    xr_args = ', '.join([ctype + ' ' + f for f in ('*x', '*p', '*r', '*v', 'alpha')])

    xr_kernel = ElementwiseKernel(context, operation=update_xr_source, preamble=preamble,
                                  arguments=xr_args)

    def xr_update(x, p, r, v, alpha, e):
        return xr_kernel(x, p, r, v, alpha, wait_for=e)

    # -------------------------------------

    update_ri_source = '''
    (double3)(r[i].real * r[i].real, \
              r[i].real * r[i].imag, \
              r[i].imag * r[i].imag)
    '''

    ri_dtype = pyopencl.array.vec.double3

    ri_kernel = ReductionKernel(context, preamble=preamble,
                                dtype_out=ri_dtype,
                                neutral='(double3)(0.0, 0.0, 0.0)',
                                map_expr=update_ri_source,
                                reduce_expr='a+b',
                                arguments=ctype + ' *r')

    def ri_update(r, e):
        g = ri_kernel(r, wait_for=e).astype(ri_dtype).get()
        rr, ri, ii = [g[q] for q in 'xyz']
        rho = rr + 2j * ri - ii
        err = rr + ii
        return rho, err

    # -------------------------------------

    update_p_source = '''
    p[i] = cdouble_add(r[i], cdouble_mul(beta, p[i]));
    '''
    p_args = ctype + ' *p, ' + ctype + ' *r, ' + ctype + ' beta'

    p_kernel = ElementwiseKernel(context, preamble=preamble, operation=update_p_source,
                                 arguments=p_args)

    def p_update(p, r, beta, e):
        return p_kernel(p, r, beta, wait_for=e)

    ops = {
        'spmv': spmv,
        'p_update': p_update,
        'ri_update': ri_update,
        'xr_update': xr_update,
    }

    return ops


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

    ops = create_ops(context)

    x = pyopencl.array.zeros(queue, dtype=numpy.complex128, shape=b.shape)
    v = pyopencl.array.empty_like(x)
    p = pyopencl.array.zeros_like(x)
    r = pyopencl.array.to_device(queue, b)
    alpha = 1.0 + 0j
    rho = 1.0 + 0j
    errs = []

    m = CSRMatrix(queue, a)

    e = ops['spmv'](v, m, x, [])
    e = ops['xr_update'](x, p, r, v, 0.0, [e])
    _, err2 = ops['ri_update'](r, [e])

    b_norm = numpy.sqrt(err2)
    print('b_norm check: ', b_norm)

    start_time2 = time.perf_counter()

    for k in range(max_iters):
        if verbose:
            print('[{:06d}] rho {:.4} alpha {:4.4}'.format(k, rho, alpha), end=' ')
        rho_prev = rho
        e = ops['xr_update'](x, p, r, v, alpha, [e])
        rho, err2 = ops['ri_update'](r, [e])
        errs += [numpy.sqrt(err2) / b_norm]

        if verbose:
            print('err', errs[-1])

        if errs[-1] < err_thresh:
            time_elapsed = time.perf_counter() - start_time
            print('Success, {} iterations in {} sec: {} iterations/sec'.format(k,
                   time_elapsed, k/time_elapsed))
            print('overhead', start_time2-start_time)
            return x.get(), errs, True
        e = ops['p_update'](p, r, rho/rho_prev, [])
        e.wait()
        ops['spmv'](v, m, p, [e]).wait()
        alpha = rho / pyopencl.array.dot(p, v).get()

        if k % 1000 == 0:
            print(k)

    return x.get(), errs, False
