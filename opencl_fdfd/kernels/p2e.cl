/*
 * Apply PEC and preconditioner.
 *
 * Template parameters:
 *  ctype       name of complex type (eg. cdouble)
 *  pec         false iff no PEC anyhwere
 *
 * Arguments:
 *  ctype *E    (output) E-field
 *  ctype *Pr   Entries of (diagonal) right preconditioner matrix
 *  ctype *p    (input vector)
 *
 */


//Defines to clean up operation names
#define ctype {{ctype}}_t
#define zero {{ctype}}_new(0.0, 0.0)
#define mul {{ctype}}_mul


{%- if pec -%}
if (pec[i] != 0) {
    E[i] = zero;
} else
{%- endif -%}
{
    E[i] = mul(Pr[i], p[i]);
}
