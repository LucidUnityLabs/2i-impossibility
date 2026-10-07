"""Exact 2I irreps, faithful real-eight spinor lifts, and tensor invariants."""
from collections import Counter
from itertools import combinations_with_replacement, product
from exact_2i import group, require, Zphi, ZERO
from verification_io import cli


def real_partitions(g, target=8):
    require(type(target) is int and target >= 0, 'real target dimension must be nonnegative')
    labels = list(g.chars)
    rd = [g.dims[k]*(2 if g.fs[k] == -1 else 1) for k in labels]
    require(all(g.fs[k] in (-1,1) for k in labels),'complex-type pairing not implemented')
    result = []
    def visit(i, left, uses):
        if i == len(labels):
            if left == 0:
                result.append(tuple(uses))
            return
        for m in range(left//rd[i]+1):
            visit(i+1,left-m*rd[i],uses+[m])
    visit(0,target,[])
    return labels,result


def spinor_characters_su2(g, torus_weights):
    """Construct both half-spinors from a coherent SU2 torus lift.

    Vector weights are +/- each entry. Spinor weights are half-signed sums;
    their integer SU2 highest-weight decompositions are checked, not assumed.
    """
    require(bool(torus_weights) and all(type(w) is int and w >= 0 for w in torus_weights),
            'invalid orthogonal torus weights')
    counts = [Counter(),Counter()]
    for signs in product((-1,1),repeat=len(torus_weights)):
        value = sum(e*w for e,w in zip(signs,torus_weights))
        require(value%2 == 0,'spinor torus weights are not SU2-integral')
        counts[signs.count(-1)%2][value//2] += 1
    chars = []
    for weights in counts:
        summands = []
        while weights:
            highest = max(weights)
            multiplicity = weights[highest]
            require(highest >= 0 and multiplicity > 0,'not a nonnegative SU2 character')
            summands.append(g.scale(multiplicity,g.su2(highest)))
            for weight in range(highest,-highest-1,-2):
                weights[weight] -= multiplicity
                require(weights[weight] >= 0,'spinor SU2 weight subtraction failed')
                if weights[weight] == 0:
                    del weights[weight]
        chars.append(g.add(*summands))
    return tuple(chars)


def spinor_search(g):
    c, mul, add, scale = g.chars,g.mul,g.add,g.scale
    labels,parts = real_partitions(g)
    # Every entry is an actual homomorphism 2I -> Spin(4)=SU(2)xSU(2).
    # Each pair consists of the two complex half-spinor characters of dimension 2.
    two = scale(2,c['1'])
    blocks = {'H2a':(c['2a'],two), 'H2b':(c['2b'],two),
              'R4':(c['2a'],c['2b']), 'R3a+1':(c['2a'],c['2a']),
              'R3b+1':(c['2b'],c['2b']), '4R1':(two,two)}
    lifted = {}
    for a,b in combinations_with_replacement(blocks,2):
        A,B = blocks[a]
        C,D = blocks[b]
        vector = add(mul(A,B),mul(C,D))
        S = add(mul(A,C),mul(B,D))
        T = add(mul(A,D),mul(B,C))
        # Equal vector modules may admit a global chirality swap only.
        canonical_pair = tuple(sorted((S,T)))
        if vector in lifted:
            require(lifted[vector] == canonical_pair,'inconsistent Spin(4)+Spin(4) lifts')
        lifted[vector] = canonical_pair
    # For realified Sym^3(C^2), the SU(2) lift gives these half-spinors.
    v4 = scale(2,c['4H'])
    torus = tuple(sorted((abs(w) for w in range(3,-4,-2)),reverse=True))
    lifted[v4] = tuple(sorted(spinor_characters_su2(g,torus)))
    faithful = []
    for uses in parts:
        vector = add(*(scale(m*(2 if g.fs[k] == -1 else 1),c[k])
                       for k,m in zip(labels,uses)))
        require(vector[g.id_class] == Zphi(8),'not real dimension eight')
        kernel = [ci for ci,v in enumerate(vector) if v == Zphi(8)]
        if kernel != [g.id_class]:
            continue
        # Exhaustiveness guard: an unhandled faithful vector is an error, not a skip.
        require(vector in lifted,'faithful real-eight module lacks a constructed lift')
        S,T = lifted[vector]
        require(S[g.id_class] == T[g.id_class] == Zphi(8),'bad half-spinor dimension')
        ds,dt = g.decompose(S),g.decompose(T)
        faithful.append({'real_pieces':{k:m for k,m in zip(labels,uses) if m},
                         'complexification':g.decompose(vector),
                         'half_spinors':[ds,dt],
                         'singlets':[ds.get('1',0),dt.get('1',0)]})
    winners = [r for r in faithful if 3 in r['singlets']]
    require(len(faithful) == 12,'unexpected faithful search size')
    require(len(winners) == 1 and winners[0]['real_pieces'] == {'4H':1},
            'three-singlet uniqueness claim failed')
    return {'total_real_eight_partitions':len(parts), 'faithful_partitions':faithful,
            'unique_three_singlet_piece':winners[0]['real_pieces'],
            'chirality_convention':'unordered pair; swap is global, not classwise'}


def tensor_invariants(g, max_degree=12):
    require(type(max_degree) is int and max_degree >= 0,'bad symmetric-power degree')
    answer = []
    for n in range(max_degree+1):
        # Harmonic decomposition Sym^n(R^3)_C = V_n + V_(n-2) + ... .
        char = g.add(*(g.su2(2*j) for j in range(n,-1,-2)))
        require(char[g.id_class] == Zphi((n+1)*(n+2)//2),'wrong Sym^n dimension')
        answer.append(g.average(g.mul(g.mul(g.chars['2a'],g.chars['2a']),char)))
    require(answer[:7] == [1,1,1,1,1,2,3][:len(answer[:7])], 'low-degree invariant regression')
    return answer


def build():
    """Irrep/FS table; all faithful real-eight modules with constructed lifts; low-degree invariants."""
    g = group()
    return {'group':g.certificate(), 'spinor_search':spinor_search(g),
            'tensor_invariants_by_degree':tensor_invariants(g),
            'not_verified':['No assertion about a Dirac index or physical generation count.']}


if __name__ == '__main__':
    raise SystemExit(cli(build,__file__))
