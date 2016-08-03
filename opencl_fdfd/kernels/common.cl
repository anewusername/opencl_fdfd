/* Common code for E, H updates
 *
 * Template parameters:
 *  shape       list of 3 ints specifying shape of fields
 */

// Field sizes
const int sx = {{shape[0]}};
const int sy = {{shape[1]}};
const int sz = {{shape[2]}};

//Since we use i to index into Ex[], E[], ... rather than E[], do nothing if
// i is outside the bounds of Ex[].
if (i >= sx * sy * sz) {
    PYOPENCL_ELWISE_CONTINUE;
}

// Given a linear index i and shape (sx, sy, sz), defines x, y, and z
//  as the 3D indices of the current element (i).
// (ie, converts linear index [i] to field indices (x, y, z)
const int z = i / (sx * sy);
const int y = (i - z * sx * sy) / sx;
const int x = (i - y * sx - z * sx * sy);

// Calculate linear index offsets corresponding to offsets in 3D
// (ie, if E[i] <-> E(x, y, z), then E[i + diy] <-> E(x, y + 1, z)
const int dix = 1;
const int diy = sx;
const int diz = sx * sy;

//Pointer offsets into the components of a linearized vector-field
// (eg. Hx = H + XX, where H and Hx are pointers)
const int XX = 0;
const int YY = sx * sy * sz;
const int ZZ = sx * sy * sz * 2;

//Define pointers to vector components of each field (eg. Hx = H + XX)
__global ctype *Ex = E + XX;
__global ctype *Ey = E + YY;
__global ctype *Ez = E + ZZ;

__global ctype *Hx = H + XX;
__global ctype *Hy = H + YY;
__global ctype *Hz = H + ZZ;
