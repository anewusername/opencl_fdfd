/*
 * Apply PEC and preconditioner.
 *
 * Template parameters:
 *  pec         false iff no PEC anywhere
 *
 * Arguments:
 *  ctype *E    (output) E-field
 *  ctype *Pr   Entries of (diagonal) right preconditioner matrix
 *  ctype *p    (input vector)
 *
 */


{%- if pec -%}
if (pec[i] != 0) {
    E[i] = zero;
} else
{%- endif -%}
{
    E[i] = mul(Pr[i], p[i]);
}
