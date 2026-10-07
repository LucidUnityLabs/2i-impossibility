"""Reconstruct principal-A1 branching from E6/E8 root and weight systems.

Nodes 0,...,rank-2 form a chain; the last node attaches to node 2.
The E6 weight at end node 0 is minuscule, with a 27-element Weyl orbit.
"""
from collections import Counter
from fractions import Fraction
from exact_2i import require


def cartan(rank):
    require(rank in (6,8),'only E6 and E8 are implemented')
    A = [[2*int(i == j) for j in range(rank)] for i in range(rank)]
    for i,j in [(i,i+1) for i in range(rank-2)]+[(2,rank-1)]:
        A[i][j] = A[j][i] = -1
    return A


def principal_h(A):
    n = len(A)
    aug = [list(map(Fraction,row))+[Fraction(2)] for row in A]
    for i in range(n):
        pivot = next(k for k in range(i,n) if aug[k][i])
        aug[i],aug[pivot] = aug[pivot],aug[i]
        scale = aug[i][i]
        aug[i] = [x/scale for x in aug[i]]
        for j in range(n):
            if j != i:
                scale = aug[j][i]
                aug[j] = [x-scale*y for x,y in zip(aug[j],aug[i])]
    return tuple(row[-1] for row in aug)


def orbit(weight,A):
    seen, queue = {tuple(weight)},[tuple(weight)]
    while queue:
        w = queue.pop()
        for i,row in enumerate(A):
            v = tuple(w[j]-w[i]*row[j] for j in range(len(A)))
            if v not in seen:
                seen.add(v)
                queue.append(v)
        require(len(seen) <= 1000, 'unexpected Weyl orbit; check Cartan input')
    return seen


def branch(rank, representation='adjoint'):
    A = cartan(rank)
    h = principal_h(A)
    if representation == 'adjoint':
        weights = list(orbit(A[0],A)) + [(0,)*rank]*rank
        require(len(weights) == {6:78,8:248}[rank], 'wrong root-system size')
    else:
        require(rank == 6 and representation == '27','unsupported highest weight')
        weights = list(orbit((1,0,0,0,0,0),A))
        require(len(weights) == 27,'wrong minuscule weight orbit')
    counts = Counter()
    for w in weights:
        x = sum(a*b for a,b in zip(w,h))
        require(x.denominator == 1,'nonintegral A1 weight')
        counts[int(x)] += 1
    require(all(counts[x] == counts[-x] for x in counts),'asymmetric A1 weights')
    highest = {}
    while counts:
        n = max(counts)
        m = counts[n]
        require(n >= 0 and m > 0,'not an A1 representation')
        highest[n] = m
        for w in range(-n,n+1,2):
            counts[w] -= m
            require(counts[w] >= 0,'negative A1 weight multiplicity')
            if counts[w] == 0:
                del counts[w]
    require(sum((n+1)*m for n,m in highest.items()) == len(weights),'branching dimension mismatch')
    return dict(sorted(highest.items()))
