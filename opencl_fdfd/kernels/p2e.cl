
{%- if pec -%}
if (pec[i] != 0) {
    E[i] = cdouble_new(0.0, 0.0);
} else
{%- endif -%}
{
    E[i] = cdouble_mul(Pr[i], p[i]);
}
