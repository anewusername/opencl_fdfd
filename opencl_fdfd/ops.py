"""
Basic PyOpenCL operations

The functions are mostly concerned with creating and compiling OpenCL
kernels for use by the other solvers.

See kernels/ for any of the .cl files loaded in this file.
"""

from typing import List, Callable, Union, Type, Sequence, Optional, Tuple
import logging

import numpy
from numpy.typing import NDArray, ArrayLike
import jinja2

import pyopencl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel


logger = logging.getLogger(__name__)

# Create jinja2 env on module load
jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'kernels'))

# Return type for the create_opname(...) functions
operation = Callable[..., List[pyopencl.Event]]


def type_to_C(
        float_type: Type,
        ) -> str:
    """
    Returns a string corresponding to the C equivalent of a numpy type.

    Args:
        float_type: numpy type: float32, float64, complex64, complex128

    Returns:
        string containing the corresponding C type (eg. 'double')
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

# Type names
ctype = type_to_C(numpy.complex128)
ctype_bare = 'cdouble'

# Preamble for all OpenCL code
preamble = '''
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

//Defines to clean up operation and type names
#define ctype {ctype}_t
#define zero {ctype}_new(0.0, 0.0)
#define add {ctype}_add
#define sub {ctype}_sub
#define mul {ctype}_mul
'''.format(ctype=ctype_bare)


def ptrs(*args: str) -> List[str]:
    return [ctype + ' *' + s for s in args]


def create_a(
        context: pyopencl.Context,
        shape: ArrayLike,
        mu: bool = False,
        pec: bool = False,
        pmc: bool = False,
        ) -> operation:
    """
    Return a function which performs (A @ p), where A is the FDFD wave equation for E-field.

    The returned function has the signature
    spmv(E, H, p, idxes, oeps, inv_mu, pec, pmc, Pl, Pr, e)
    with arguments (all except e are of type pyopencl.array.Array (or contain it)):
     E      E-field (output)
     H      Temporary variable for holding intermediate H-field values on GPU (same size as E)
     p      p-vector (input vector)
     idxes  list holding [[1/dx_e, 1/dy_e, 1/dz_e],  [1/dx_h, 1/dy_h, 1/dz_h]] (complex cell widths)
     oeps   omega * epsilon
     inv_mu 1/mu
     pec    array of bytes; nonzero value indicates presence of PEC
     pmc    array of bytes; nonzero value indicates presence of PMC
     Pl     Left preconditioner (array containing diagonal entries only)
     Pr     Right preconditioner (array containing diagonal entries only)
     e      List of pyopencl.Event; execution will wait until these are finished.

     and returns a list of pyopencl.Event.

    Args:
        context: PyOpenCL context
        shape: Dimensions of the E-field
        mu: False iff (mu == 1) everywhere
        pec: False iff no PEC anywhere
        pmc: False iff no PMC anywhere

    Returns:
        Function for computing (A @ p)
    """

    common_source = jinja_env.get_template('common.cl').render(shape=shape)

    pec_arg = ['char *pec']
    pmc_arg = ['char *pmc']
    des = [ctype + ' *inv_de' + a for a in 'xyz']
    dhs = [ctype + ' *inv_dh' + a for a in 'xyz']

    '''
    Convert p to initial E (ie, apply right preconditioner and PEC)
    '''
    p2e_source = jinja_env.get_template('p2e.cl').render(pec=pec)
    P2E_kernel = ElementwiseKernel(
        context,
        name='P2E',
        preamble=preamble,
        operation=p2e_source,
        arguments=', '.join(ptrs('E', 'p', 'Pr') + pec_arg),
        )

    '''
    Calculate intermediate H from intermediate E
    '''
    e2h_source = jinja_env.get_template('e2h.cl').render(
        mu=mu,
        pmc=pmc,
        common_cl=common_source,
        )
    E2H_kernel = ElementwiseKernel(
        context,
        name='E2H',
        preamble=preamble,
        operation=e2h_source,
        arguments=', '.join(ptrs('E', 'H', 'inv_mu') + pmc_arg + des),
        )

    '''
    Calculate final E (including left preconditioner)
    '''
    h2e_source = jinja_env.get_template('h2e.cl').render(
        pec=pec,
        common_cl=common_source,
        )
    H2E_kernel = ElementwiseKernel(
        context,
        name='H2E',
        preamble=preamble,
        operation=h2e_source,
        arguments=', '.join(ptrs('E', 'H', 'oeps', 'Pl') + pec_arg + dhs),
        )

    def spmv(
            E: pyopencl.array.Array,
            H: pyopencl.array.Array,
            p: pyopencl.array.Array,
            idxes: Sequence[Sequence[pyopencl.array.Array]],
            oeps: pyopencl.array.Array,
            inv_mu: Optional[pyopencl.array.Array],
            pec: Optional[pyopencl.array.Array],
            pmc: Optional[pyopencl.array.Array],
            Pl: pyopencl.array.Array,
            Pr: pyopencl.array.Array,
            e: List[pyopencl.Event],
            ) -> List[pyopencl.Event]:
        e2 = P2E_kernel(E, p, Pr, pec, wait_for=e)
        e2 = E2H_kernel(E, H, inv_mu, pmc, *idxes[0], wait_for=[e2])
        e2 = H2E_kernel(E, H, oeps, Pl, pec, *idxes[1], wait_for=[e2])
        return [e2]

    logger.debug(f'Preamble: \n{preamble}')
    logger.debug(f'p2e: \n{p2e_source}')
    logger.debug(f'e2h: \n{e2h_source}')
    logger.debug(f'h2e: \n{h2e_source}')

    return spmv


def create_xr_step(context: pyopencl.Context) -> operation:
    """
    Return a function
     xr_update(x, p, r, v, alpha, e)
    which performs the operations
     x += alpha * p
     r -= alpha * v

    after waiting for all in the list e
    and returns a list of pyopencl.Event

    Args:
        context: PyOpenCL context

    Returns:
        Function for performing x and r updates
    """
    update_xr_source = '''
    x[i] = add(x[i], mul(alpha, p[i]));
    r[i] = sub(r[i], mul(alpha, v[i]));
    '''

    xr_args = ', '.join(ptrs('x', 'p', 'r', 'v') + [ctype + ' alpha'])

    xr_kernel = ElementwiseKernel(
        context,
        name='XR',
        preamble=preamble,
        operation=update_xr_source,
        arguments=xr_args,
        )

    def xr_update(
            x: pyopencl.array.Array,
            p: pyopencl.array.Array,
            r: pyopencl.array.Array,
            v: pyopencl.array.Array,
            alpha: complex,
            e: List[pyopencl.Event],
            ) -> List[pyopencl.Event]:
        return [xr_kernel(x, p, r, v, alpha, wait_for=e)]

    return xr_update


def create_rhoerr_step(context: pyopencl.Context) -> Callable[..., Tuple[complex, complex]]:
    """
    Return a function
     ri_update(r, e)
    which performs the operations
     rho = r * r.conj()
     err = r * r

    after waiting for all pyopencl.Event in the list e
    and returns a list of pyopencl.Event

    Args:
        context: PyOpenCL context

    Returns:
        Function for performing x and r updates
    """

    update_ri_source = '''
    (double3)(r[i].real * r[i].real, \
              r[i].real * r[i].imag, \
              r[i].imag * r[i].imag)
    '''

    # Use a vector type (double3) to make the reduction simpler
    ri_dtype = pyopencl.array.vec.double3

    ri_kernel = ReductionKernel(
        context,
        name='RHOERR',
        preamble=preamble,
        dtype_out=ri_dtype,
        neutral='(double3)(0.0, 0.0, 0.0)',
        map_expr=update_ri_source,
        reduce_expr='a+b',
        arguments=ctype + ' *r',
        )

    def ri_update(r: pyopencl.array.Array, e: List[pyopencl.Event]) -> Tuple[complex, complex]:
        g = ri_kernel(r, wait_for=e).astype(ri_dtype).get()
        rr, ri, ii = [g[q] for q in 'xyz']
        rho = rr + 2j * ri - ii
        err = rr + ii
        return rho, err

    return ri_update


def create_p_step(context: pyopencl.Context) -> operation:
    """
    Return a function
     p_update(p, r, beta, e)
    which performs the operation
     p = r + beta * p

    after waiting for all pyopencl.Event in the list e
    and returns a list of pyopencl.Event

    Args:
        context: PyOpenCL context

    Returns:
        Function for performing the p update
    """
    update_p_source = '''
    p[i] = add(r[i], mul(beta, p[i]));
    '''
    p_args = ptrs('p', 'r') + [ctype + ' beta']

    p_kernel = ElementwiseKernel(
        context,
        name='P',
        preamble=preamble,
        operation=update_p_source,
        arguments=', '.join(p_args),
        )

    def p_update(
            p: pyopencl.array.Array,
            r: pyopencl.array.Array,
            beta: complex,
            e: List[pyopencl.Event]) -> List[pyopencl.Event]:
        return [p_kernel(p, r, beta, wait_for=e)]

    return p_update


def create_dot(context: pyopencl.Context) -> Callable[..., complex]:
    """
    Return a function for performing the dot product
     p @ v
    with the signature
     dot(p, v, e) -> complex

    Args:
        context: PyOpenCL context

    Returns:
        Function for performing the dot product
    """
    dot_dtype = numpy.complex128

    dot_kernel = ReductionKernel(
        context,
        name='dot',
        preamble=preamble,
        dtype_out=dot_dtype,
        neutral='zero',
        map_expr='mul(p[i], v[i])',
        reduce_expr='add(a, b)',
        arguments=ptrs('p', 'v'),
        )

    def dot(
            p: pyopencl.array.Array,
            v: pyopencl.array.Array,
            e: List[pyopencl.Event],
            ) -> complex:
        g = dot_kernel(p, v, wait_for=e)
        return g.get()

    return dot


def create_a_csr(context: pyopencl.Context) -> operation:
    """
    Return a function for performing the operation
     (N @ v)
    where N is stored in CSR (compressed sparse row) format.

    The function signature is
     spmv(v_out, m, v_in, e)
    where m is an opencl_fdfd.csr.CSRMatrix
    and v_out, v_in are (dense) vectors (of type pyopencl.array.Array).

    The function waits on all the pyopencl.Event in e before running, and returns
     a list of pyopencl.Event.

    Args:
        context: PyOpenCL context

    Returns:
        Function for sparse (M @ v) operation where M is in CSR format
    """
    spmv_source = '''
    int start = m_row_ptr[i];
    int stop = m_row_ptr[i+1];
    ctype dot = zero;

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

    spmv_kernel = ElementwiseKernel(
        context,
        name='csr_spmv',
        preamble=preamble,
        operation=spmv_source,
        arguments=', '.join((v_out_args, m_args, v_in_args)),
        )

    def spmv(
            v_out,
            m,
            v_in,
            e: List[pyopencl.Event],
            ) -> List[pyopencl.Event]:
        return [spmv_kernel(v_out, m.row_ptr, m.col_ind, m.data, v_in, wait_for=e)]

    return spmv
