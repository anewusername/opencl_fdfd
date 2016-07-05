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


def shape_source(shape) -> str:
    """
    Defines sx, sy, sz C constants specifying the shape of the grid in each of the 3 dimensions.

    :param shape: [sx, sy, sz] values.
    :return: String containing C source.
    """
    sxyz = """
// Field sizes
const int sx = {shape[0]};
const int sy = {shape[1]};
const int sz = {shape[2]};
""".format(shape=shape)
    return sxyz

# Defines dix, diy, diz constants used for stepping in the x, y, z directions in a linear array
#  (ie, given Ex[i] referring to position (x, y, z), Ex[i+diy] will refer to position (x, y+1, z))
dixyz_source = """
// Convert offset in field xyz to linear index offset
const int dix = 1;
const int diy = sx;
const int diz = sx * sy;
"""

# Given a linear index i and shape sx, sy, sz, defines x, y, and z
#  as the 3D indices of the current element (i).
xyz_source = """
// Convert linear index to field index (xyz)
const int z = i / (sx * sy);
const int y = (i - z * sx * sy) / sx;
const int x = (i - y * sx - z * sx * sy);
"""

vec_source = """
if (i >= sx * sy * sz) {
    PYOPENCL_ELWISE_CONTINUE;
}

//Pointers into the components of a vectorized vector-field
const int XX = 0;
const int YY = sx * sy * sz;
const int ZZ = sx * sy * sz * 2;
"""

E_ptrs = """
__global cdouble_t *Ex = E + XX;
__global cdouble_t *Ey = E + YY;
__global cdouble_t *Ez = E + ZZ;
"""

H_ptrs = """
__global cdouble_t *Hx = H + XX;
__global cdouble_t *Hy = H + YY;
__global cdouble_t *Hz = H + ZZ;
"""

preamble = '''
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>
'''

ctype = type_to_C(numpy.complex128)


def ptrs(*args):
    return [ctype + ' *' + s for s in args]


def create_a(context, shape, mu=False, pec=False, pmc=False):
    header = shape_source(shape) + dixyz_source + xyz_source
    vec_h = vec_source + E_ptrs + H_ptrs

    pec_arg = ['char *pec']
    pmc_arg = ['char *pmc']
    des = [ctype + ' *inv_de' + a for a in 'xyz']
    dhs = [ctype + ' *inv_dh' + a for a in 'xyz']

    p2e_source = jinja_env.get_template('p2e.cl').render(pec=pec)
    P2E_kernel = ElementwiseKernel(context,
                                   name='P2E',
                                   preamble=preamble,
                                   operation=p2e_source,
                                   arguments=', '.join(ptrs('E', 'p', 'Pr') + pec_arg))

    e2h_source = jinja_env.get_template('e2h.cl').render(mu=mu,
                                                         pmc=pmc,
                                                         dixyz_source=header,
                                                         vec_source=vec_h)
    E2H_kernel = ElementwiseKernel(context,
                                   name='E2H',
                                   preamble=preamble,
                                   operation=e2h_source,
                                   arguments=', '.join(ptrs('E', 'H', 'inv_mu') + pmc_arg + des))

    h2e_source = jinja_env.get_template('h2e.cl').render(pec=pec,
                                                         dixyz_source=header,
                                                         vec_source=vec_h)
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
    x[i] = cdouble_add(x[i], cdouble_mul(alpha, p[i]));
    r[i] = cdouble_sub(r[i], cdouble_mul(alpha, v[i]));
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
    p[i] = cdouble_add(r[i], cdouble_mul(beta, p[i]));
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
                                 neutral='cdouble_new(0.0, 0.0)',
                                 map_expr='cdouble_mul(p[i], v[i])',
                                 reduce_expr='cdouble_add(a, b)',
                                 arguments=ptrs('p', 'v'))

    def ri_update(p, v, e):
        g = dot_kernel(p, v, wait_for=e)
        return g.get()

    return ri_update
