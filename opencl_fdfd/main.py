import numpy
from numpy.linalg import norm

import jinja2
import pyopencl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

import time

import fdfd_tools.operators


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

# Source code for updating the E field; maxes use of dixyz_source.
maxwell_E_source = """
// E update equations
int imx, imy, imz;
if ( x == 0 ) {
  imx = i + (sx - 1) * dix;
} else {
  imx = i - dix;
}

if ( y == 0 ) {
  imy = i + (sy - 1) * diy;
} else {
  imy = i - diy;
}

if ( z == 0 ) {
  imz = i + (sz - 1) * diz;
} else {
  imz = i - diz;
}

// E update equations

{% if pec -%}
if (pec[XX + i]) {
    Ex[i] = cdouble_new(0.0, 0.0);
} else
{%- endif -%}
{
    cdouble_t tEx = cdouble_mul(Ex[i], oeps[XX + i]);
    cdouble_t Dzy = cdouble_mul(cdouble_sub(Hz[i], Hz[imy]), inv_dhy[y]);
    cdouble_t Dyz = cdouble_mul(cdouble_sub(Hy[i], Hy[imz]), inv_dhz[z]);
    tEx = cdouble_add(tEx, cdouble_sub(Dzy, Dyz));
    Ex[i] = cdouble_mul(tEx, Pl[XX + i]);
}

{% if pec -%}
if (pec[YY + i]) {
    Ey[i] = cdouble_new(0.0, 0.0);
} else
{%- endif -%}
{
    cdouble_t tEy = cdouble_mul(Ey[i], oeps[YY + i]);
    cdouble_t Dxz = cdouble_mul(cdouble_sub(Hx[i], Hx[imz]), inv_dhz[z]);
    cdouble_t Dzx = cdouble_mul(cdouble_sub(Hz[i], Hz[imx]), inv_dhx[x]);
    tEy = cdouble_add(tEy, cdouble_sub(Dxz, Dzx));
    Ey[i] = cdouble_mul(tEy, Pl[YY + i]);
}

{% if pec -%}
if (pec[ZZ + i]) {
    Ez[i] = cdouble_new(0.0, 0.0);
} else
{%- endif -%}
{
    cdouble_t tEz = cdouble_mul(Ez[i], oeps[ZZ + i]);
    cdouble_t Dyx = cdouble_mul(cdouble_sub(Hy[i], Hy[imx]), inv_dhx[x]);
    cdouble_t Dxy = cdouble_mul(cdouble_sub(Hx[i], Hx[imy]), inv_dhy[y]);
    tEz = cdouble_add(tEz, cdouble_sub(Dyx, Dxy));
    Ez[i] = cdouble_mul(tEz, Pl[ZZ + i]);
}
"""

# Source code for updating the H field; maxes use of dixyz_source and assumes mu=0
maxwell_H_source = """
// H update equations
int ipx, ipy, ipz;
if ( x == sx - 1 ) {
  ipx = i - (sx - 1) * dix;
} else {
  ipx = i + dix;
}

if ( y == sy - 1 ) {
  ipy = i - (sy - 1) * diy;
} else {
  ipy = i + diy;
}

if ( z == sz - 1 ) {
  ipz = i - (sz - 1) * diz;
} else {
  ipz = i + diz;
}

{% if pmc -%}
if (pmc[XX + i]) {
    Hx[i] = cdouble_new(0.0, 0.0);
} else
{%- endif -%}
{
    cdouble_t Dzy = cdouble_mul(cdouble_sub(Ez[ipy], Ez[i]), inv_dey[y]);
    cdouble_t Dyz = cdouble_mul(cdouble_sub(Ey[ipz], Ey[i]), inv_dez[z]);

    {%- if mu -%}
    Hx[i] = cdouble_mul(inv_mu[XX + i], cdouble_sub(Dzy, Dyz));
    {%- else -%}
    Hx[i] = cdouble_sub(Dzy, Dyz);
    {%- endif %}
}

{% if pmc -%}
if (pmc[YY + i]) {
    Hy[i] = cdouble_new(0.0, 0.0);
} else
{%- endif -%}
{
    cdouble_t Dxz = cdouble_mul(cdouble_sub(Ex[ipz], Ex[i]), inv_dez[z]);
    cdouble_t Dzx = cdouble_mul(cdouble_sub(Ez[ipx], Ez[i]), inv_dex[x]);

    {%- if mu -%}
    Hy[i] = cdouble_mul(inv_mu[YY + i], cdouble_sub(Dxz, Dzx));
    {%- else -%}
    Hy[i] = cdouble_sub(Dxz, Dzx);
    {%- endif %}
}

{% if pmc -%}
if (pmc[XX + i]) {
    Hx[i] = cdouble_new(0.0, 0.0);
} else
{%- endif -%}
{
    cdouble_t Dyx = cdouble_mul(cdouble_sub(Ey[ipx], Ey[i]), inv_dex[x]);
    cdouble_t Dxy = cdouble_mul(cdouble_sub(Ex[ipy], Ex[i]), inv_dey[y]);

    {%- if mu -%}
    Hz[i] = cdouble_mul(inv_mu[ZZ + i], cdouble_sub(Dyx, Dxy));
    {%- else -%}
    Hz[i] = cdouble_sub(Dyx, Dxy);
    {%- endif %}
}
"""

p2e_source = '''
Ex[i] = cdouble_mul(Pr[XX + i], p[XX + i]);
Ey[i] = cdouble_mul(Pr[YY + i], p[YY + i]);
Ez[i] = cdouble_mul(Pr[ZZ + i], p[ZZ + i]);
'''

preamble = '''
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>
'''

ctype = type_to_C(numpy.complex128)


def ptrs(*args):
    return [ctype + ' *' + s for s in args]


def create_a(context, shape, mu=False, pec=False, pmc=False):
    dhs = [ctype + ' *inv_dh' + a for a in 'xyz']
    des = [ctype + ' *inv_de' + a for a in 'xyz']

    header = shape_source(shape) + dixyz_source + xyz_source + vec_source + E_ptrs
    P2E_kernel = ElementwiseKernel(context,
                                   name='P2E',
                                   preamble=preamble,
                                   operation=header + p2e_source,
                                   arguments=', '.join(ptrs('E', 'p', 'Pr')))

    pmc_arg = ['int *pmc']
    e2h_source = header + H_ptrs + jinja2.Template(maxwell_H_source).render(mu=mu, pmc=pmc)
    E2H_kernel = ElementwiseKernel(context,
                                   name='E2H',
                                   preamble=preamble,
                                   operation=e2h_source,
                                   arguments=', '.join(ptrs('E', 'H', 'inv_mu') + pmc_arg + des))

    pec_arg = ['int *pec']
    h2e_source = header + H_ptrs + jinja2.Template(maxwell_E_source).render(pec=pec)
    H2E_kernel = ElementwiseKernel(context,
                                   name='H2E',
                                   preamble=preamble,
                                   operation=h2e_source,
                                   arguments=', '.join(ptrs('E', 'H', 'oeps', 'Pl') + pec_arg + dhs))

    def spmv(E, H, p, idxes, oeps, inv_mu, pec, pmc, Pl, Pr, e):
        e2 = P2E_kernel(E, p, Pr, wait_for=e)
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
        gpec = load_field(numpy.array([]), dtype=int)
    else:
        gpec = load_field(pec, dtype=int)

    if pmc is None:
        gpmc = load_field(numpy.array([]), dtype=int)
    else:
        gpmc = load_field(pmc, dtype=int)

    '''
    Generate OpenCL kernels
    '''
    has_mu, has_pec, has_pmc = [q is not None for q in (mu, pec, pmc)]

    a_step_full = create_a(context, shape, has_mu, has_pec, has_pmc)
    xr_step = create_xr_step(context)
    rhoerr_step = create_rhoerr_step(context)
    p_step = create_p_step(context)
    dot = create_dot(context)

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