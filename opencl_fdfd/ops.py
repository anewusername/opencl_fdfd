import numpy
import jinja2

import pyopencl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

# Create jinja2 env on module load
jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'kernels'))


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


preamble = '''
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

'''

ctype = type_to_C(numpy.complex128)


def ptrs(*args):
    return [ctype + ' *' + s for s in args]


def create_a(context, shape, mu=False, pec=False, pmc=False):

    common_source = jinja_env.get_template('common.cl').render(shape=shape,
                                                               ctype=ctype)

    pec_arg = ['char *pec']
    pmc_arg = ['char *pmc']
    des = [ctype + ' *inv_de' + a for a in 'xyz']
    dhs = [ctype + ' *inv_dh' + a for a in 'xyz']

    p2e_source = jinja_env.get_template('p2e.cl').render(pec=pec,
                                                         ctype=ctype)
    P2E_kernel = ElementwiseKernel(context,
                                   name='P2E',
                                   preamble=preamble,
                                   operation=p2e_source,
                                   arguments=', '.join(ptrs('E', 'p', 'Pr') + pec_arg))

    e2h_source = jinja_env.get_template('e2h.cl').render(mu=mu,
                                                         pmc=pmc,
                                                         common_cl=common_source)
    E2H_kernel = ElementwiseKernel(context,
                                   name='E2H',
                                   preamble=preamble,
                                   operation=e2h_source,
                                   arguments=', '.join(ptrs('E', 'H', 'inv_mu') + pmc_arg + des))

    h2e_source = jinja_env.get_template('h2e.cl').render(pec=pec,
                                                         common_cl=common_source)
    H2E_kernel = ElementwiseKernel(context,
                                   name='H2E',
                                   preamble=preamble,
                                   operation=h2e_source,
                                   arguments=', '.join(ptrs('E', 'H', 'oeps', 'Pl') + pec_arg + dhs))

    def spmv(E, H, p, idxes, oeps, inv_mu, pec, pmc, Pl, Pr, e):
        e2 = P2E_kernel(E, p, Pr, pec, wait_for=e)
        e2 = E2H_kernel(E, H, inv_mu, pmc, *idxes[0], wait_for=[e2])
        e2 = H2E_kernel(E, H, oeps, Pl, pec, *idxes[1], wait_for=[e2])
        return [e2]

    return spmv


def create_xr_step(context):
    update_xr_source = '''
    x[i] = add(x[i], mul(alpha, p[i]));
    r[i] = sub(r[i], mul(alpha, v[i]));
    '''

    xr_args = ', '.join(ptrs('x', 'p', 'r', 'v') + [ctype + ' alpha'])

    xr_kernel = ElementwiseKernel(context,
                                  name='XR',
                                  preamble=preamble,
                                  operation=update_xr_source,
                                  arguments=xr_args)

    def xr_update(x, p, r, v, alpha, e):
        return [xr_kernel(x, p, r, v, alpha, wait_for=e)]

    return xr_update


def create_rhoerr_step(context):
    update_ri_source = '''
    (double3)(r[i].real * r[i].real, \
              r[i].real * r[i].imag, \
              r[i].imag * r[i].imag)
    '''

    ri_dtype = pyopencl.array.vec.double3

    ri_kernel = ReductionKernel(context,
                                name='RHOERR',
                                preamble=preamble,
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

    return ri_update


def create_p_step(context):
    update_p_source = '''
    p[i] = add(r[i], mul(beta, p[i]));
    '''
    p_args = ptrs('p', 'r') + [ctype + ' beta']

    p_kernel = ElementwiseKernel(context,
                                 name='P',
                                 preamble=preamble,
                                 operation=update_p_source,
                                 arguments=', '.join(p_args))

    def p_update(p, r, beta, e):
        return [p_kernel(p, r, beta, wait_for=e)]

    return p_update


def create_dot(context):
    dot_dtype = numpy.complex128

    dot_kernel = ReductionKernel(context,
                                 name='dot',
                                 preamble=preamble,
                                 dtype_out=dot_dtype,
                                 neutral='zero',
                                 map_expr='mul(p[i], v[i])',
                                 reduce_expr='add(a, b)',
                                 arguments=ptrs('p', 'v'))

    def ri_update(p, v, e):
        g = dot_kernel(p, v, wait_for=e)
        return g.get()

    return ri_update


def create_a_csr(context):
    spmv_source = '''
    int start = m_row_ptr[i];
    int stop = m_row_ptr[i+1];
    dtype dot = zero;

    int col_ind, d_ind;
    for (int j=start; j<stop; j++) {
        col_ind = m_col_ind[j];
        d_ind = j;

        dot = add(dot, mul(v_in[col_ind], m_data[d_ind]));
    }
    v_out[i] = dot;
    '''

    v_out_args = ctype + ' *v_out'
    m_args = 'int *m_row_ptr, int *m_col_ind, ' + ctype + ' *m_data'
    v_in_args = ctype + ' *v_in'

    spmv_kernel = ElementwiseKernel(context,
                                    name='csr_spmv',
                                    preamble=preamble,
                                    operation=spmv_source,
                                    arguments=', '.join((v_out_args, m_args, v_in_args)))

    def spmv(v_out, m, v_in, e):
        return [spmv_kernel(v_out, m.row_ptr, m.col_ind, m.data, v_in, wait_for=e)]

    return spmv
