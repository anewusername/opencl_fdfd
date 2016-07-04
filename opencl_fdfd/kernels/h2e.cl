/*
 *
 * E update equations
 *
 */

//Define sx, x, dix (and y, z versions of those)
{{dixyz_source}}

//Define vectorized fields and pointers (eg. Hx = H + XX)
{{vec_source}}


// Wrap indices if necessary
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

/*
 * End H update equations
 */