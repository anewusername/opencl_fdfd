{#
/* Common code for E, H updates
 *
 * Template parameters:
 *  shape       list of 3 ints specifying shape of fields
 */
#}

/*
 * Field size info
 */
// Field sizes
const int sx = {{shape[0]}};
const int sy = {{shape[1]}};
const int sz = {{shape[2]}};
const size_t field_size = sx * sy * sz;

//Since we use i to index into Ex[], Ey[], ... rather than E[], do nothing if
// i is outside the bounds of Ex[].
if (i >= field_size) {
    PYOPENCL_ELWISE_CONTINUE;
}


/*
 * Array indexing
 */
// Given a linear index i and shape (sx, sy, sz), defines x, y, and z
//  as the 3D indices of the current element (i).
// (ie, converts linear index [i] to field indices (x, y, z)
const int x = i / (sz * sy);
const int y = (i - x * sz * sy) / sz;
const int z = (i - y * sz - x * sz * sy);

// Calculate linear index offsets corresponding to offsets in 3D
// (ie, if E[i] <-> E(x, y, z), then E[i + diy] <-> E(x, y + 1, z)
const int dix = sz * sy;
const int diy = sz;
const int diz = 1;


/*
 * Pointer math
 */
//Pointer offsets into the components of a linearized vector-field
// (eg. Hx = H + XX, where H and Hx are pointers)
const int XX = 0;
const int YY = field_size;
const int ZZ = field_size * 2;

//Define pointers to vector components of each field (eg. Hx = H + XX)
__global ctype *Ex = E + XX;
__global ctype *Ey = E + YY;
__global ctype *Ez = E + ZZ;

__global ctype *Hx = H + XX;
__global ctype *Hy = H + YY;
__global ctype *Hz = H + ZZ;


/*
 * Implement periodic boundary conditions
 *
 * mx ([m]inus [x]) gives the index offset of the adjacent cell in the minus-x direction.
 * In the event that we start at x == 0, we actually want to wrap around and grab the cell
 * x_{-1} == (sx - 1) instead, ie. mx = (sx - 1) * dix .
 *
 * px ([p]lus [x]) gives the index offset of the adjacent cell in the plus-x direction.
 * In the event that we start at x == (sx - 1), we actually want to wrap around and grab
 * the cell x_{+1} == 0 instead, ie. px = -(sx - 1) * dix .
 */
{% for r in 'xyz' %}
int m{{r}} = -di{{r}};
int p{{r}} = +di{{r}};
int wrap_{{r}} = (s{{r}} - 1) * di{{r}};
if ( {{r}} == 0 ) {
  m{{r}} = wrap_{{r}};
} else if ( {{r}} == s{{r}} - 1 ) {
  p{{r}} = -wrap_{{r}};
}
{% endfor %}
