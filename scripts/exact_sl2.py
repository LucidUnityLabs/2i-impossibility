"""Exact character tables of SL(2,F_p), computed from integer class operators.

The class operators used here are the transpose/similar version obtained
by multiplying a fixed representative. They have the usual class-sum
central characters as eigenvalues. No character-table data file is used.
"""
from collections import Counter
from functools import lru_cache
from itertools import product
from math import lcm
import sympy as sp
from exact_2i import require, VerificationError


def multiply(a,b,p):
    w,x,y,z = a
    v,r,s,t = b
    return ((w*v+x*s)%p, (w*r+x*t)%p, (y*v+z*s)%p, (y*r+z*t)%p)


def exact_integer(x, message):
    x = sp.simplify(x)
    require(x.is_Integer is True, f'{message}: {x}')
    return int(x)


@lru_cache(maxsize=None)
def character_data(p):
    require(type(p) is int and p in (2,5,7), 'supported primes are 2, 5, 7')
    elements = tuple(a for a in product(range(p),repeat=4) if (a[0]*a[3]-a[1]*a[2])%p == 1)
    n = len(elements)
    require(n == p*(p*p-1), 'wrong SL2 order')
    index = {a:i for i,a in enumerate(elements)}
    e = index[(1,0,0,1)]
    table = tuple(tuple(index[multiply(a,b,p)] for b in elements) for a in elements)
    inv = tuple(index[(a[3],-a[1]%p,-a[2]%p,a[0])] for a in elements)
    orders = []
    for i in range(n):
        x = e
        for k in range(1,n+1):
            x = table[x][i]
            if x == e:
                orders.append(k)
                break
        else:
            raise VerificationError('unbounded matrix order')
    classes, seen = [],set()
    for i in range(n):
        if i in seen:
            continue
        cls = tuple(sorted({table[table[g][i]][inv[g]] for g in range(n)}))
        require(not seen.intersection(cls),'class overlap')
        seen.update(cls)
        classes.append(cls)
    classes.sort(key=lambda c:(orders[c[0]],len(c),c[0]))
    sizes = tuple(map(len,classes))
    k = len(classes)
    class_of = {x:ci for ci,cls in enumerate(classes) for x in cls}
    id_class = class_of[e]
    mats = []
    for cls in classes:
        M = sp.zeros(k)
        for j,cj in enumerate(classes):
            for a in cls:
                M[class_of[table[a][cj[0]]],j] += 1
        mats.append(M)
    # Deterministic search for a separating element of the commutative algebra.
    # Failure is explicit; a degenerate spectrum is never rounded or accepted.
    eigenvectors = None
    for base in (2,3,5,7,11):
        A = sum((base**i*M for i,M in enumerate(mats)), sp.zeros(k))
        ev = A.eigenvects()
        if len(ev) == k and all(mult == 1 and len(vs) == 1 for _,mult,vs in ev):
            eigenvectors = [vs[0] for _,_,vs in ev]
            break
    require(eigenvectors is not None, 'could not find a simple exact spectrum')
    rows = []
    for v in eigenvectors:
        pivot = next(j for j in range(k) if v[j] != 0)
        lambdas = []
        for M in mats:
            lam = sp.simplify((M*v)[pivot]/v[pivot])
            require(all(sp.simplify(x) == 0 for x in M*v-lam*v),
                    'not a simultaneous class-algebra eigenvector')
            lambdas.append(lam)
        denom = sum(lam*sp.conjugate(lam)/s for lam,s in zip(lambdas,sizes))
        d = exact_integer(sp.sqrt(sp.simplify(n/denom)), 'nonintegral dimension')
        require(d > 0, 'nonpositive dimension')
        chars = tuple(sp.simplify(sp.expand_complex(lam*d/s)) for lam,s in zip(lambdas,sizes))
        require(chars[id_class] == d, 'identity character mismatch')
        rows.append((d,chars))
    rows.sort(key=lambda row:(row[0],tuple(sp.srepr(x) for x in row[1])))
    def inner(a,b):
        return sp.simplify(sum(s*x*sp.conjugate(y) for s,x,y in zip(sizes,a,b))/n)
    for i,(_,a) in enumerate(rows):
        for j,(_,b) in enumerate(rows):
            require(inner(a,b) == int(i == j), 'SL2 character orthogonality failed')
    require(sum(d*d for d,_ in rows) == n, 'SL2 characters incomplete')
    squares = []
    for cls in classes:
        target = {class_of[table[x][x]] for x in cls}
        require(len(target) == 1, 'ill-defined square-class map')
        squares.append(target.pop())
    records = []
    for idx,(d,chars) in enumerate(rows):
        fs = exact_integer(sum(s*chars[c] for s,c in zip(sizes,squares))/n, 'bad FS')
        require(fs in (-1,0,1), 'invalid FS value')
        records.append({'label':f'SL2_{p}_{idx+1}', 'dimension':d, 'fs':fs,
                        'character':chars})
    require(sum(r['dimension']*r['fs'] for r in records) ==
            sum(table[x][x] == e for x in range(n)), 'SL2 Frobenius count failed')
    return {'p':p, 'order':n, 'exponent':lcm(*orders),
            'classes':[{'size':len(c), 'order':orders[c[0]], 'representative':elements[c[0]]}
                       for c in classes], 'records':records}


def json_data(data):
    return {**data, 'records':[{**r, 'character':[sp.sstr(c) for c in r['character']]}
                               for r in data['records']]}
