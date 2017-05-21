/*
 * H update equations
 *
 * Template parameters:
 *  mu          False if (mu == 1) everywhere
 *  pmc         False if no PMC anywhere
 *  common_cl   Rendered code from common.cl
 *
 * Arguments:
 *  ctype *E        E-field
 *  ctype *H        H-field
 *  ctype *inv_mu   1/mu (at H-field locations)
 *  char  *pmc      Boolean mask denoting presence of PMC (at H-field locations)
 *  ctype *inv_dex  1/dx_e (complex cell widths for x direction at E locations)
 *  ctype *inv_dey  1/dy_e (complex cell widths for y direction at E locations)
 *  ctype *inv_dez  1/dz_e (complex cell widths for z direction at E locations)
 *
 */

{{common_cl}}

////////////////////////////////////////////////////////////////////////////

__global ctype *inv_mu_x = inv_mu + XX;
__global ctype *inv_mu_y = inv_mu + YY;
__global ctype *inv_mu_z = inv_mu + ZZ;

__global char *pmc_x = pmc + XX;
__global char *pmc_y = pmc + YY;
__global char *pmc_z = pmc + ZZ;


//Update H components; set them to 0 if PMC is enabled at that location.
//Mu division and PMC conditional are only included if {mu} and {pmc} are true
{% if pmc -%}
if (pmc_x[i] != 0) {
    Hx[i] = zero;
} else
{%- endif -%}
{
    ctype Dzy = mul(sub(Ez[i + py], Ez[i]), inv_dey[y]);
    ctype Dyz = mul(sub(Ey[i + pz], Ey[i]), inv_dez[z]);
    ctype x_curl = sub(Dzy, Dyz);

    {%- if mu %}
    Hx[i] = mul(inv_mu_x[i], x_curl);
    {%- else %}
    Hx[i] = x_curl;
    {%- endif %}
}

{% if pmc -%}
if (pmc_y[i] != 0) {
    Hy[i] = zero;
} else
{%- endif -%}
{
    ctype Dxz = mul(sub(Ex[i + pz], Ex[i]), inv_dez[z]);
    ctype Dzx = mul(sub(Ez[i + px], Ez[i]), inv_dex[x]);
    ctype y_curl = sub(Dxz, Dzx);

    {%- if mu %}
    Hy[i] = mul(inv_mu_y[i], y_curl);
    {%- else %}
    Hy[i] = y_curl;
    {%- endif %}
}

{% if pmc -%}
if (pmc_z[i] != 0) {
    Hz[i] = zero;
} else
{%- endif -%}
{
    ctype Dyx = mul(sub(Ey[i + px], Ey[i]), inv_dex[x]);
    ctype Dxy = mul(sub(Ex[i + py], Ex[i]), inv_dey[y]);
    ctype z_curl = sub(Dyx, Dxy);

    {%- if mu %}
    Hz[i] = mul(inv_mu_z[i], z_curl);
    {%- else %}
    Hz[i] = z_curl;
    {%- endif %}
}

/*
 * End H update equations
 */
