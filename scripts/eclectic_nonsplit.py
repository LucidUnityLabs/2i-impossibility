"""Exact finite examples and counterexamples; undefined structures stay unverified."""
from collections import Counter
from itertools import permutations, product
import sympy as sp
from exact_2i import group, require, Zphi
from verification_io import cli


def parity(p):
    return sum(p[i] > p[j] for i in range(len(p)) for j in range(i+1,len(p)))%2


def compose(p,q):
    return tuple(p[q[i]] for i in range(len(p)))


def delta_elements(n):
    require(type(n) is int and 2 <= n <= 35, 'supported Delta parameter: integer 2..35')
    return tuple((a,b,(-a-b)%n,p) for a,b in product(range(n),repeat=2)
                 for p in permutations(range(3)))


def delta_multiply(x,y,n):
    a,b,p = x[:3],y[:3],x[3]
    inverse = tuple(p.index(i) for i in range(3))
    v = tuple((a[i]+b[inverse[i]])%n for i in range(3))
    return (*v,compose(p,y[3]))


def delta_certificate(n):
    require(n in (3,5), 'full finite checks are implemented for n=3,5')
    elements = delta_elements(n)
    elements_set = set(elements)
    require(len(elements_set) == 6*n*n, 'wrong Delta order')
    identity = (0,0,0,(0,1,2))
    # Exact integer-state closure, not rounded complex matrix keys.
    for x in elements:
        require(delta_multiply(identity,x,n) == x == delta_multiply(x,identity,n),
                'Delta identity failure')
        has_inverse = False
        for y in elements:
            xy = delta_multiply(x,y,n)
            require(xy in elements_set, 'Delta closure failure')
            has_inverse |= xy == identity and delta_multiply(y,x,n) == identity
        require(has_inverse, 'Delta inverse failure')
    z = sp.Symbol('z')
    mod = sp.Poly(sp.cyclotomic_poly(n,z),z,domain=sp.QQ)
    def reduced(expr):
        return sp.rem(sp.Poly(expr,z,domain=sp.QQ),mod).as_expr()
    values, norm = [],0
    for x in elements:
        p = x[3]
        sign = (-1)**parity(p)
        # rho(x)=sgn(p)*D(z^a,z^b,z^c)*P_p has determinant 1.
        require((sum(x[:3])%n == 0) and sign**3 * (-1)**parity(p) == 1,
                'Delta SU3 determinant failure')
        trace = reduced(sign*sum(z**x[i] for i in range(3) if p[i] == i))
        conjugate = reduced(sign*sum(z**((-x[i])%n) for i in range(3) if p[i] == i))
        values.append(trace)
        norm += reduced(trace*conjugate)
    norm = reduced(norm)/len(elements)
    require(norm == 1, 'natural Delta representation is not irreducible')
    return {'order':len(elements), 'dimension':3, 'character_norm':int(norm),
            'distinct_trace_values':sorted({sp.sstr(v) for v in values}),
            'cyclotomic_modulus':sp.sstr(mod.as_expr()),
            'phi_occurrences':sum(reduced(v-(1+z+z**4)) == 0 for v in values) if n == 5 else 0,
            'trace_note':'Trace-value buckets are not conjugacy classes; equality is exact in Q[z]/Phi_n.',
            'sine_excluded_from_character_field':int(sp.totient(n))%3 != 0,
            'icosahedral_subgroup_excluded_by_order':len(elements)%120 != 0}


def nonsplit_s3_extension():
    perms = tuple(permutations(range(3)))
    E = tuple((p,t) for p in perms for t in range(4) if parity(p) == t%2)
    def mul(x,y):
        return compose(x[0],y[0]),(x[1]+y[1])%4
    e = ((0,1,2),0)
    require(len(E) == 12,'wrong central-extension order')
    require(all(mul(x,y) in E for x in E for y in E),'extension closure')
    kernel = [x for x in E if x[0] == e[0]]
    require(len(kernel) == 2 and all(mul(x,y) == mul(y,x) for x in kernel for y in E),
            'kernel is not central C2')
    transposition = (1,0,2)
    preimages = [x for x in E if x[0] == transposition]
    require(len(preimages) == 2 and all(mul(x,x) != e and mul(mul(x,x),mul(x,x)) == e
                                      for x in preimages), 'nonsplitting witness failed')
    return {'order':12,'kernel_order':2,'transposition_lift_orders':[4,4],
            'split':False, 'definition':'{(g,t) in S3 x C4: sign(g)=t mod 2}',
            'proof':'A section would send a transposition of order two to an involutive preimage; there is none.'}


def cyclic_twist_counterexample():
    # omega(a,b,c)=(-1)^(a*b*c) is a normalized 3-cocycle on C2.
    def w(a,b,c):
        return (-1)**(a*b*c)
    for a,b,c,d in product(range(2),repeat=4):
        require(w(b,c,d)*w(a,(b+c)%2,d)*w(a,b,c) ==
                w((a+b)%2,c,d)*w(a,b,(c+d)%2),'3-cocycle equation failure')
    # On the flux-one centralizer, the slant 2-cocycle has alpha(1,1)=-1.
    alpha = w(1,1,1)*w(1,1,1)//w(1,1,1)
    require(alpha == -1,'wrong projective square')
    require(sp.I**2 == alpha and (-sp.I)**2 == alpha, 'projective characters failed')
    return {'base_group_exponent':2,'projective_generator_values':['I','-I'],
            'twist_order':4,'old_expG_cyclotomic_cap_refuted':True}


def cross_prime_counterexample():
    """An actual irreducible of 2I x D14 has a character value of degree six."""
    z,a,y = sp.symbols('z a y')
    cyclo = sp.Poly(sp.cyclotomic_poly(7,z),z,domain=sp.QQ)
    def red(expr):
        return sp.rem(sp.Poly(expr,z,domain=sp.QQ),cyclo).as_expr()
    rotations = [red(z**k+z**((-k)%7)) for k in range(7)]
    require(red(sum(v*v for v in rotations)) == 14,
            'D14 two-dimensional character is not irreducible')
    t = z+z**6
    require(red(t**3+t**2-2*t-1) == 0,'real seventh-root polynomial failed')
    # y=phi*t; eliminate phi using phi^2-phi-1=0.
    minimal = sp.Poly(sp.resultant(a*a-a-1,y**3+a*y**2-2*a*a*y-a**3,a),y)
    require(minimal.degree() == 6 and minimal.is_irreducible,
            'cross-prime character does not have degree six')
    require(group().chars['2a'].count(Zphi(0,1)) > 0,
            'phi is absent from the natural 2I character')
    return {'group':'2I x D14 (D14 has order 14)', 'group_order':120*14,
            'representation':'natural 2I doublet outer-tensored with D14 doublet',
            'dimension':4, 'character_norm':1,
            'character_value':'phi*(zeta7+zeta7^(-1))',
            'minimal_polynomial':sp.sstr(minimal.as_expr()), 'degree':6,
            'meaning':'An irreducible outer tensor product can carry both character fields.',
            'not_claimed':'Unscaled sin(pi/14) is not a character value; no physical model is constructed.'}


def build():
    """Exact Delta(54)/Delta(150), a non-split S3 extension, and a C2 cocycle counterexample; not an exhaustive classification."""
    deltas = {f'Delta_{6*n*n}':delta_certificate(n) for n in (3,5)}
    fiber_order, product_order = 6*35**2,(6*5**2)*(6*7**2)
    require(product_order == 6*fiber_order,'shared-S3 fiber-product check failed')
    exponent = group().exponent
    return {'delta_examples':deltas, 'nonsplit_S3_C2':nonsplit_s3_extension(),
            'twisted_C2':cyclic_twist_counterexample(),
            'cross_prime_irreducible':cross_prime_counterexample(),
            'Delta7350_CRT':{'fiber_product_order':fiber_order,'direct_product_order':product_order,
                 'structure':'Delta(150) x_{S3} Delta(294), not the direct product'},
            'twisted_2I_FSexp':{'status':'theorem_applied_not_cocycles_enumerated',
                 'divisibility_bound':exponent**2,
                 'reference':'Ng-Schauenburg, arXiv:math/0601012v3, Theorem 9.2',
                 'scope':'FS-exponent/canonical twist; not arbitrary categorical traces'},
            'unresolved_claims':[
                 {'claim':'specific Delta(108)','status':'not_verified','needed':'presentation or SmallGroup identifier'},
                 {'claim':'Mp(2,Z/14) splitting','status':'not_verified','needed':'specified cocycle/extension and splitting proof'},
                 {'claim':'2-group with nonabelian pi2=SL(2,7)','status':'invalid_definition',
                  'needed':'crossed module with abelian ker(boundary), action and Postnikov class'},
                 {'claim':'all finite groups/quasi-Hopf algebras excluded','status':'not_established'}]}


if __name__ == '__main__':
    raise SystemExit(cli(build,__file__))
