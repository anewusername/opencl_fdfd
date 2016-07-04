/*
 *
 * H update equations
 *
 */

//Define sx, x, dix (and y, z versions of those)
{{dixyz_source}}

//Define vectorized fields and pointers (eg. Hx = H + XX)
{{vec_source}}


// Wrap indices if necessary
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


//Update H components; set them to 0 if PMC is enabled there.
// Also divide by mu only if requested.
{% if pmc -%}
if (pmc[XX + i] != 0) {
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
if (pmc[YY + i] != 0) {
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
if (pmc[ZZ + i] != 0) {
    Hz[i] = cdouble_new(0.0, 0.0);
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

/*
 * End H update equations
 */