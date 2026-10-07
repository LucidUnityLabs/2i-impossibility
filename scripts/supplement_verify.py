"""Exact finite checks supporting only explicitly scoped representation/arithmetic statements."""
import sympy as sp
from exact_2i import group, require
from verification_io import cli


def arithmetic():
    z,x = sp.symbols('z x')
    cyclo = sp.Poly(sp.cyclotomic_poly(14,z),z,domain=sp.QQ)
    # z=exp(pi*i/7), z^7=-1: sin(pi/14)=(z^3-z^4)/2.
    s = (z**3-z**4)/2
    p = sp.Poly(8*x**3-4*x**2-4*x+1,x,domain=sp.QQ)
    require(sp.rem(sp.Poly(p.as_expr().subs(x,s),z),cyclo).is_zero,
            'exact cyclotomic substitution failed')
    require(p.is_irreducible is True,'claimed minimal polynomial is reducible')
    require(p.degree() == 3,'wrong algebraic degree')
    # For an algebraic number, its MONIC minimal polynomial must be in Z[x]
    # to be an algebraic integer. Nonmonicity alone without irreducibility is insufficient.
    monic = p.monic()
    integral = all(c.q == 1 for c in monic.all_coeffs())
    require(not integral,'unexpected algebraic integer')
    field_degree = int(sp.totient(group().exponent))
    require(field_degree % p.degree() != 0,'degree argument does not exclude containment')
    scaled = sp.Poly(x**3-x**2-2*x+1,x)
    require(sp.rem(sp.Poly(scaled.as_expr().subs(x,2*s),z),cyclo).is_zero,
            'scaled-number polynomial failed')
    require(scaled.is_irreducible,'2*sin(pi/14) must have degree three')
    return {'minimal_polynomial':'8*x^3-4*x^2-4*x+1', 'degree':p.degree(),
            'algebraic_integer':integral, 'cyclotomic_cap_degree':field_degree,
            'contained_in_Q_zeta60':False,
            'twice_sine_minpoly':'x^3-x^2-2*x+1',
            'literal_character_value_of_any_finite_group':False,
            'scope':'Literal traces and field containment only; not mixing angles or all extensions.'}


def molien(g, limit=60):
    actual = [g.invariant_spin(n) for n in range(limit+1)]
    expected = [sum(1 for shift in (0,30) for a in range(limit//12+1)
                    for b in range(limit//20+1) if 12*a+20*b+shift == n)
                for n in range(limit+1)]
    require(actual == expected,'binary-icosahedral Molien coefficient mismatch')
    return actual


def build():
    """Exact 2I table, binary-doublet Molien coefficients through degree 60, and cubic arithmetic."""
    g = group()
    require(g.exponent == 60,'unexpected exponent')
    return {'group':g.certificate(), 'arithmetic':arithmetic(),
            'doublet_molien_coefficients':molien(g),
            'not_verified':['Universal chirality, arbitrary extensions, compactification classification.']}


if __name__ == '__main__':
    raise SystemExit(cli(build,__file__))
