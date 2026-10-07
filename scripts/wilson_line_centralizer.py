"""Exact fixed-adjoint dimensions for specified SU2 maps; no exhaustive classification claim."""
from itertools import combinations, product
from exact_2i import group, require
from principal_weights import branch
from verification_io import cli
from verify_2I_reps import spinor_characters_su2


def fixed(decomposition, expected_dimension):
    require(all(type(n) is int and n >= 0 and type(m) is int and m > 0
                for n,m in decomposition.items()), 'invalid A1 branching')
    require(sum((n+1)*m for n,m in decomposition.items()) == expected_dimension,
            'branching has wrong total dimension')
    return sum(m*group().invariant_spin(n) for n,m in decomposition.items())


def natural_j32():
    """The stated Spin(8) block inside Spin(16), including its E8 triality conjugate."""
    g = group()
    c = g.chars
    vector8 = g.scale(2,c['4H'])
    vector16 = g.add(vector8,g.scale(8,c['1']))
    squares = g.power_classes(2)
    wedge2 = tuple((x*x-vector16[j]).div(2) for x,j in zip(vector16,squares))
    spin8a,spin8b = spinor_characters_su2(g,(3,3,1,1))
    spin16 = g.scale(8,g.add(spin8a,spin8b))
    adjoint = g.add(wedge2,spin16)
    require((g.average(wedge2),g.average(spin16),g.average(adjoint)) == (31,24,55),
            'natural j=3/2 invariant regression')
    # E8 roots scaled by 2; simultaneous D4 triality is an E8 root automorphism.
    roots = set()
    for i,j in combinations(range(8),2):
        for a,b in product((-2,2),repeat=2):
            r = [0]*8
            r[i],r[j] = a,b
            roots.add(tuple(r))
    roots.update(r for r in product((-1,1),repeat=8) if r.count(-1)%2 == 0)
    H = ((1,1,1,1),(1,1,-1,-1),(1,-1,1,-1),(1,-1,-1,1))
    def triality(r):
        values = [sum(H[i][j]*r[offset+j] for j in range(4))
                  for offset in (0,4) for i in range(4)]
        require(all(v%2 == 0 for v in values),'nonintegral triality image')
        return tuple(v//2 for v in values)
    require(len(roots) == 240 and {triality(r) for r in roots} == roots,
            'simultaneous triality does not preserve E8 roots')
    require(triality((3,3,1,1,0,0,0,0)) == (4,2,0,0,0,0,0,0),
            'triality A1 torus mismatch')
    # Conjugate vector: V_2 (dimension 5) + eleven trivials in SO(16).
    # 248=(10,1)+(1,55)+(5,11)+(4,32) under Spin(5) x Spin(11).
    branching = {0:55,2:1,3:32,4:11,6:1}
    conjugate_adjoint = g.add(*(g.scale(m,g.su2(n)) for n,m in branching.items()))
    require(adjoint == conjugate_adjoint,'triality branching equality failed')
    require(fixed(branching,248) == 55,'extra invariants beyond so(11)')
    return {'D8_fixed_dimension':31,'spinor_fixed_dimension':24,'E8_fixed_dimension':55,
            'E8_fixed_lie_type':'B5 = so(11)', 'rank':5,'coxeter_number':10,
            'E8_roots_preserved_by_simultaneous_triality':240,
            'proof':'Triality gives the 5+11 orthogonal splitting; so(11) is an exhibited commuting subalgebra and exhausts all 55 invariants.',
            'global_centralizer_group':'not identified'}


def build():
    """Specified E8/E6 embeddings, with all mixed E6 x A2 summands; dimensions, not a 56-embedding scan."""
    e8,e6,r27 = branch(8),branch(6),branch(6,'27')
    require(e8 == {2:1,14:1,22:1,26:1,34:1,38:1,46:1,58:1},'E8 principal weights')
    require(e6 == {2:1,8:1,10:1,14:1,16:1,22:1},'E6 principal weights')
    require(r27 == {0:1,8:1,16:1},'E6 minuscule principal weights')
    p8,p6,p27 = fixed(e8,248),fixed(e6,78),fixed(r27,27)
    # E8 = (78,1)+(1,8)+(27,3)+(27*,3*).
    principal_e6_in_e8 = p6+8+6*p27
    # E6 -> A5 x A1: 78=(35,1)+(1,3)+(20,2), 27=(15,1)+(6*,2).
    regular_e6 = fixed({0:35,2:1,1:20},78)
    regular_27 = fixed({0:15,1:6},27)
    regular_e6_in_e8 = regular_e6+8+6*regular_27
    # E8 -> E7 x A1: 248=(133,1)+(1,3)+(56,2).
    regular_e8 = fixed({0:133,2:1,1:56},248)
    require((p8,p6,p27,principal_e6_in_e8,regular_e6_in_e8,regular_e8) ==
            (0,0,1,14,133,133),'centralizer regression failed')
    return {'natural_j_3_over_2':natural_j32(), 'principal_branchings_twice_spin':{
                'E8':{str(k):v for k,v in e8.items()},
                'E6':{str(k):v for k,v in e6.items()},
                'E6_27':{str(k):v for k,v in r27.items()}},
            'fixed_adjoint_dimensions':{'principal_E8':p8,'principal_E6':p6,
                 'principal_E6_in_E8':principal_e6_in_e8,
                 'regular_A1_in_E6_in_E8':regular_e6_in_e8,
                 'regular_A1_in_E8':regular_e8},
            'global_form_warning':'Principal E6/E8 maps kill the central involution of 2I; their image is A5.',
            'not_verified':['No Lie type is inferred from dimension alone; natural-j=3/2 type uses an exhibited so(11) subalgebra.',
                            'No exhaustive scan of E8 embeddings or Wilson-line homomorphisms.',
                            'No statement that a zero-dimensional centralizer group is trivial.']}


if __name__ == '__main__':
    raise SystemExit(cli(build,__file__))
