/*
 * E update equations
 *
 * Template parameters:
 *  pec         False if no PEC anywhere
 *  common_cl   Rendered code from common.cl
 *
 * Arguments:
 *  ctype *E        E-field
 *  ctype *H        H-field
 *  ctype *oeps     omega*epsilon (at E-field locations)
 *  ctype *Pl       Entries of (diagonal) left preconditioner matrix
 *  char  *pec      Boolean mask denoting presence of PEC (at E-field locations)
 *  ctype *inv_dhx  1/dx_h (complex cell widths for x direction at H locations)
 *  ctype *inv_dhy  1/dy_h (complex cell widths for y direction at H locations)
 *  ctype *inv_dhz  1/dz_h (complex cell widths for z direction at H locations)
 *
 */

{{common_cl}}


__global ctype *oeps_x = oeps + XX;
__global ctype *oeps_y = oeps + YY;
__global ctype *oeps_z = oeps + ZZ;

__global char *pec_x = pec + XX;
__global char *pec_y = pec + YY;
__global char *pec_z = pec + ZZ;

__global ctype *Pl_x = Pl + XX;
__global ctype *Pl_y = Pl + YY;
__global ctype *Pl_z = Pl + ZZ;


/*
 * Implement periodic boundary conditions
 *
 * imx gives the index of the adjacent cell in the minus-x direction ([i]ndex [m]inus [x]).
 * In the event that we start at x == 0, we actually want to wrap around and grab the cell
 * where x == (sx - 1) instead, ie. imx = i + (sx - 1) * dix .
 */
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


//Update E components; set them to 0 if PEC is enabled there.
{% if pec -%}
if (pec_x[i] == 0)
{%- endif -%}
{
    ctype tEx = mul(Ex[i], oeps_x[i]);
    ctype Dzy = mul(sub(Hz[i], Hz[imy]), inv_dhy[y]);
    ctype Dyz = mul(sub(Hy[i], Hy[imz]), inv_dhz[z]);
    tEx = add(tEx, sub(Dzy, Dyz));
    Ex[i] = mul(tEx, Pl_x[i]);
}

{% if pec -%}
if (pec_y[i] == 0)
{%- endif -%}
{
    ctype tEy = mul(Ey[i], oeps_y[i]);
    ctype Dxz = mul(sub(Hx[i], Hx[imz]), inv_dhz[z]);
    ctype Dzx = mul(sub(Hz[i], Hz[imx]), inv_dhx[x]);
    tEy = add(tEy, sub(Dxz, Dzx));
    Ey[i] = mul(tEy, Pl_y[i]);
}

{% if pec -%}
if (pec_z[i] == 0)
{%- endif -%}
{
    ctype tEz = mul(Ez[i], oeps_z[i]);
    ctype Dyx = mul(sub(Hy[i], Hy[imx]), inv_dhx[x]);
    ctype Dxy = mul(sub(Hx[i], Hx[imy]), inv_dhy[y]);
    tEz = add(tEz, sub(Dyx, Dxy));
    Ez[i] = mul(tEz, Pl_z[i]);
}

/*
 * End E update equations
 */
