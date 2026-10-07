"""Exact CRT, character fields, FS data and modular dimensions. No uncomputed fit estimates."""
from collections import Counter
from fractions import Fraction
from itertools import product
from math import gcd, lcm, prod
import sympy as sp
from exact_2i import group, require
from exact_sl2 import character_data, json_data
from supplement_verify import arithmetic
from verification_io import cli


def sl2_order(N):
    require(type(N) is int and N >= 2, 'level must be an integer >= 2')
    result = Fraction(N**3)
    for p in sp.factorint(N):
        result *= Fraction(p*p-1,p*p)
    require(result.denominator == 1,'nonintegral group order')
    return int(result)


def crt(a,b,c):
    require(type(a) is int and 0 <= a < 2 and type(b) is int and 0 <= b < 5
            and type(c) is int and 0 <= c < 7, 'bad CRT residue')
    return (35*a+56*b+50*c)%70


def verify_crt():
    triples = list(product(range(2),range(5),range(7)))
    lifted = [crt(*x) for x in triples]
    require(set(lifted) == set(range(70)),'CRT is not bijective')
    for x in triples:
        a = crt(*x)
        require((a%2,a%5,a%7) == x,'CRT inverse failed')
        for y in triples:
            b = crt(*y)
            require(crt(*((u+v)%p for u,v,p in zip(x,y,(2,5,7)))) == (a+b)%70,
                    'CRT not additive')
            require(crt(*((u*v)%p for u,v,p in zip(x,y,(2,5,7)))) == (a*b)%70,
                    'CRT not multiplicative')
    return {'residue_bijection_size':len(triples), 'ring_pairs_checked':len(triples)**2,
            'consequence':'Entrywise CRT respects matrix multiplication and determinant, hence induces SL2 isomorphism.'}


def principal_modular(N):
    require(type(N) is int and N >= 3,'torsion-free projective formulas require N >= 3')
    mu = sl2_order(N)
    c = Fraction(mu,2*N)
    genus = 1+Fraction(mu,24)-c/2
    require(c.denominator == genus.denominator == 1,'nonintegral modular invariant')
    c,g = int(c),int(genus)
    dimensions = {}
    for k in (2,4,6,8,10,12):
        S = g if k == 2 else (k-1)*(g-1)+(k//2-1)*c
        E = c-1 if k == 2 else c
        require(S >= 0 and E >= 0,'negative modular dimension')
        dimensions[str(k)] = {'cusp':S, 'eisenstein':E, 'modular':S+E}
    return {'SL2_index':mu, 'PSL2_index':mu//2, 'cusps':c, 'genus':g,
            'dimensions':dimensions}


def gamma0_genus(N):
    """Exact genus formula, using prime-power solvability for elliptic points."""
    require(type(N) is int and N >= 1,'invalid Gamma0 level')
    factors = sp.factorint(N)
    mu = Fraction(N)
    for p in factors:
        mu *= Fraction(p+1,p)
    divisors = sp.divisors(N)
    cusps = sum(int(sp.totient(gcd(int(d),N//int(d)))) for d in divisors)
    e2 = 0 if N%4 == 0 else prod(1+int(sp.kronecker_symbol(-4,p)) for p in factors)
    e3 = 0 if N%9 == 0 else prod(1+int(sp.kronecker_symbol(-3,p)) for p in factors)
    g = 1+mu/12-Fraction(e2,4)-Fraction(e3,3)-Fraction(cusps,2)
    require(g.denominator == 1 and g >= 0,'invalid Gamma0 genus')
    return int(g)


def new_weight2(N):
    new = {}
    for d0 in sp.divisors(N):
        d = int(d0)
        old = sum(len(sp.divisors(d//e))*dim for e,dim in new.items() if d%e == 0)
        new[d] = gamma0_genus(d)-old
        require(new[d] >= 0,'negative newspace dimension')
    return new


def character_field(sl7):
    K = sp.QQ.algebraic_field(sp.sqrt(2),sp.sqrt(-7))
    values = [v for r in sl7['records'] for v in r['character']]
    for v in values:
        K.from_sympy(v)  # Exact membership; CoercionFailed must stop the run.
    require(K.ext.minpoly.degree() == 4,'wrong SL2(7) field degree')
    require(any(sp.simplify(v-sp.sqrt(2)) == 0 for v in values),'sqrt(2) witness missing')
    require(any(sp.simplify(2*v+1-sp.sqrt(-7)) == 0 for v in values),'sqrt(-7) witness missing')
    L = sp.QQ.algebraic_field(sp.sqrt(5),sp.sqrt(2),sp.sqrt(-7))
    degree = L.ext.minpoly.degree()
    require(degree == 8 and degree%arithmetic()['degree'] != 0,'cubic exclusion failed')
    return {'SL2_7_character_field':'Q(sqrt(2),sqrt(-7))', 'SL2_7_field_degree':4,
            'Gamma70_character_field':'Q(sqrt(5),sqrt(2),sqrt(-7))',
            'Gamma70_field_degree':degree, 'sine_or_twice_sine_in_character_field':False,
            'proof':'All computed values lie in the stated field and generate it; cubic degree does not divide eight.'}


def build():
    """CRT ring isomorphism; exact SL2(7) class algebra; Gamma70 representations and correctly normalized modular dimensions."""
    g = group()
    s3,sl7 = character_data(2),character_data(7)
    fs_counts,dim_counts = Counter(),Counter()
    triples3 = []
    for a,k,b in product(s3['records'],g.chars,sl7['records']):
        d = a['dimension']*g.dims[k]*b['dimension']
        fs = a['fs']*g.fs[k]*b['fs']
        fs_counts[fs] += 1
        dim_counts[d] += 1
        if d == 3:
            triples3.append([a['label'],k,b['label']])
    order = s3['order']*g.n*sl7['order']
    require(sum(d*d*n for d,n in dim_counts.items()) == order == sl2_order(70),
            'Gamma70 completeness failed')
    require(sum(fs_counts.values()) == 297 and len(triples3) == 8,'Gamma70 count failed')
    mod = principal_modular(70)
    require((mod['cusps'],mod['genus'],mod['dimensions']['2']['modular']) == (1728,9217,10944),
            'X(70) regression failed')
    new = new_weight2(70)
    require(new[14] == 1 and new[35] == 3 and new[70] == 1,'Gamma0 newspace regression failed')
    return {'crt':verify_crt(), 'order':order,
            'exponent':lcm(s3['exponent'],g.exponent,sl7['exponent']),
            'SL2_7_certificate':json_data(sl7), 'character_fields':character_field(sl7),
            'fs_counts':{str(k):v for k,v in sorted(fs_counts.items())},
            'dimension_counts':{str(k):v for k,v in sorted(dim_counts.items())},
            'dimension_three_irreps':triples3, 'Gamma70_modular':mod,
            'Gamma0_weight2_new_dimensions':{str(k):v for k,v in new.items()},
            'not_verified':['No fit, q-expansion or phenomenological exclusion is computed.',
                            'Tensor-product irreps are irreducible; factorization does not prohibit mixed character fields.',
                            'Newspace dimension is not the number of Galois orbits.']}


if __name__ == '__main__':
    raise SystemExit(cli(build,__file__))
