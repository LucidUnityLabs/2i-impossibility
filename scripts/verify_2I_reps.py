"""
Rigorous verification of four claims about the binary icosahedral group 2I,
its 8-dim representations, and tensor-product invariants.

Claims:
  1. |2I|=120 with 9 irreps of dims {1,2,2,3,3,4,4,5,6}.
  2. j=3/2 realified gives a faithful irreducible 8-dim real rep of 2I;
     embedding in SO(8) produces 8_s -> 3*[1] + [5] under 2I.
  3. Among faithful real 8-dim reps of 2I, the j=3/2 route uniquely produces
     exactly 3 singlets in the 8_s branching.
  4. dim( (2 tensor 2 tensor Sym^n(3))^{2I} ) = 1 for n=0..4, >=2 for n=5.

Method: direct quaternion construction of 2I, brute-force character table
from conjugacy classes, spinor character formula for SO(8) branching.
"""

import json
import itertools
import numpy as np
from fractions import Fraction

# ---------------------------------------------------------------------------
# 1. CONSTRUCT 2I AS 120 UNIT QUATERNIONS
# ---------------------------------------------------------------------------
# Quaternions stored as tuples (w, x, y, z) of exact sympy values.
import sympy as sp

phi = (1 + sp.sqrt(5)) / 2         # golden ratio
iphi = (sp.sqrt(5) - 1) / 2        # 1/phi = phi - 1
half = sp.Rational(1, 2)

def q_mul(a, b):
    """Hamilton quaternion product."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def q_inv(a):
    """Unit-quaternion inverse = conjugate."""
    w, x, y, z = a
    return (w, -x, -y, -z)

def q_norm2(a):
    w, x, y, z = a
    return sp.simplify(w*w + x*x + y*y + z*z)

def build_2I():
    elements = set()
    # 8 pure units: +/- 1, +/- i, +/- j, +/- k
    for sign in (1, -1):
        elements.add((sign, 0, 0, 0))
        elements.add((0, sign, 0, 0))
        elements.add((0, 0, sign, 0))
        elements.add((0, 0, 0, sign))
    # 16 half-integer: (+/- 1 +/- i +/- j +/- k)/2
    for s in itertools.product((1, -1), repeat=4):
        q = tuple(sp.Rational(x, 2) for x in s)
        elements.add(q)
    # 96 golden: even permutations of (0, +/- 1, +/- phi, +/- 1/phi)/2,
    # with an even number of minus signs (so that the element lies in 2I).
    # Standard construction: take all even permutations of the pattern
    # (0, +/- 1/2, +/- phi/2, +/- (1/phi)/2) giving 12 * 8 = 96 elements.
    base_values = (0, half, phi*half, iphi*half)
    # even permutations of (0,1,2,3)
    even_perms = [
        (0,1,2,3), (0,2,3,1), (0,3,1,2),
        (1,0,3,2), (1,2,0,3)[::1], (1,3,2,0),  # careful — list below
    ]
    # Use A4 = alternating group on 4 letters, 12 permutations, as sign-permutations.
    from itertools import permutations
    def perm_sign(p):
        n = len(p)
        inv = 0
        for i in range(n):
            for j in range(i+1, n):
                if p[i] > p[j]:
                    inv += 1
        return 1 if inv % 2 == 0 else -1
    A4 = [p for p in permutations(range(4)) if perm_sign(p) == 1]
    assert len(A4) == 12
    for p in A4:
        vals = tuple(base_values[p[i]] for i in range(4))
        for signs in itertools.product((1, -1), repeat=4):
            # Only signs on the three nonzero slots matter; sign on the zero slot
            # duplicates. Deduplicate via the set.
            q = tuple(sp.nsimplify(signs[i] * vals[i]) for i in range(4))
            elements.add(q)
    return elements

def q_key(q):
    """Canonical hashable key — use string of simplified sympy form."""
    return tuple(sp.sqrtdenest(sp.simplify(c)) for c in q)

print("=" * 72)
print("CLAIM 1: 2I has 9 irreps with dims {1,2,2,3,3,4,4,5,6}")
print("=" * 72)

print("\n[1a] Building 2I as unit quaternions...")
raw = build_2I()
# Deduplicate under canonical key
canon = {}
for q in raw:
    k = q_key(q)
    canon[k] = q
elements = list(canon.values())
print(f"     constructed |2I| = {len(elements)} (expected 120)")
assert len(elements) == 120, f"Wrong element count: {len(elements)}"
print("     PASS: |2I| = 120")

# Verify all have norm 1
print("\n[1b] Verifying all elements have norm 1...")
for q in elements:
    n = q_norm2(q)
    assert sp.simplify(n - 1) == 0, f"Element {q} has norm {n}"
print("     PASS: all 120 elements are unit quaternions")

# Verify closure
print("\n[1c] Verifying group closure under quaternion multiplication...")
elt_keys = {q_key(q) for q in elements}
for a in elements:
    for b in elements:
        c = q_mul(a, b)
        if q_key(c) not in elt_keys:
            raise AssertionError(f"Closure failure: {a}*{b} = {c} not in 2I")
print("     PASS: 2I is closed (all 14400 products land in 2I)")

# ---------------------------------------------------------------------------
# Build Cayley table with integer indices for speed
# ---------------------------------------------------------------------------
key_to_idx = {q_key(q): i for i, q in enumerate(elements)}
N = 120
cayley = np.zeros((N, N), dtype=np.int64)
for i, a in enumerate(elements):
    for j, b in enumerate(elements):
        cayley[i, j] = key_to_idx[q_key(q_mul(a, b))]

# Identity index
identity = (sp.Integer(1), sp.Integer(0), sp.Integer(0), sp.Integer(0))
e_idx = key_to_idx[q_key(identity)]

# Inverse table
inv_idx = np.zeros(N, dtype=np.int64)
for i, a in enumerate(elements):
    inv_idx[i] = key_to_idx[q_key(q_inv(a))]

# ---------------------------------------------------------------------------
# Conjugacy classes
# ---------------------------------------------------------------------------
print("\n[1d] Computing conjugacy classes...")
classes = []
assigned = [False] * N
for i in range(N):
    if assigned[i]:
        continue
    cls = set()
    for g in range(N):
        # g i g^{-1}
        x = cayley[cayley[g, i], inv_idx[g]]
        cls.add(int(x))
    for x in cls:
        assigned[x] = True
    classes.append(sorted(cls))
classes.sort(key=lambda c: (len(c), c[0]))
print(f"     number of conjugacy classes = {len(classes)} (expected 9)")
assert len(classes) == 9
class_sizes = [len(c) for c in classes]
print(f"     class sizes = {class_sizes}")
# Expected 2I class sizes: 1, 1, 12, 12, 12, 12, 20, 20, 30
assert sorted(class_sizes) == [1, 1, 12, 12, 12, 12, 20, 20, 30], \
    f"Unexpected class sizes {class_sizes}"
print("     PASS: conjugacy class sizes match 2I")

# Order of a representative in each class
def order(i):
    x = i
    k = 1
    while x != e_idx:
        x = cayley[x, i]
        k += 1
        if k > N:
            raise RuntimeError("order exceeded")
    return k

class_orders = [order(c[0]) for c in classes]
print(f"     element orders (per class) = {class_orders}")
# Expected orders: 1, 2 (center -1), 3, 6, 5, 10, 5, 10, 4 (in some order)

# ---------------------------------------------------------------------------
# Character table via Burnside / Dixon-style solution
# We'll compute the character table by diagonalizing class-sum operators.
# ---------------------------------------------------------------------------
print("\n[1e] Computing character table via class algebra...")

# Build class multiplication constants c_ijk: C_i * C_j = sum_k c_ijk C_k
k_classes = len(classes)
# Record class of each element
class_of = np.zeros(N, dtype=np.int64)
for ci, cls in enumerate(classes):
    for x in cls:
        class_of[x] = ci

# For efficiency, pick a representative from each class and count hits
# For each pair (i,j), we need matrix M_i where (M_i)_{jk} = c_ijk
# via: fix g in C_j, count h in C_i such that h*g lies in C_k.
# Then (M_i)_{kj} = #{(a,b): a in C_i, b in C_j, a*b = fixed element of C_k}
# A cleaner way: for a fixed k-representative z in C_k,
#   c_ijk = #{(a,b): a in C_i, b in C_j, a*b = z} (independent of z in C_k).
# Then eigenvalues of M_i acting on columns give class-sum eigenvalues,
# which equal (|C_i| * chi(C_i) / chi(1)) for each irrep chi.

M_list = []
for ci in range(k_classes):
    Mi = np.zeros((k_classes, k_classes), dtype=np.int64)
    # For each class j, pick a representative g_j, count over a in C_i of a*g_j
    for cj in range(k_classes):
        g = classes[cj][0]
        for a in classes[ci]:
            prod = cayley[a, g]
            Mi[class_of[prod], cj] += 1
    M_list.append(Mi)

# Simultaneously diagonalize M_1, M_2, ..., M_{k-1} over C.
# Trick: form a random linear combination, diagonalize, then read off
# eigenvalues of each M_i in the eigenvector basis.
np.random.seed(42)
coeffs = np.random.randn(k_classes)
Mrand = sum(c * M for c, M in zip(coeffs, M_list))
eigvals, eigvecs = np.linalg.eig(Mrand)
# Columns of eigvecs are eigenvectors v_alpha; M_i v_alpha = lambda_{i,alpha} v_alpha
# We have lambda_{i,alpha} = |C_i| * chi_alpha(C_i) / chi_alpha(1)
# So chi_alpha(C_i) = lambda_{i,alpha} * chi_alpha(1) / |C_i|
# and we can normalize chi_alpha(1) via orthogonality:
# sum_i |C_i| |chi(C_i)|^2 = |G|
# i.e., sum_i |C_i| * |lambda_{i,alpha}|^2 * chi(1)^2 / |C_i|^2 = |G|
# => chi(1)^2 = |G| / sum_i (|lambda_{i,alpha}|^2 / |C_i|)

char_table = np.zeros((k_classes, k_classes), dtype=complex)
dims = []
for alpha in range(k_classes):
    v = eigvecs[:, alpha]
    lams = np.array([(M_list[ci] @ v / v)[np.argmax(np.abs(v))]
                     for ci in range(k_classes)])
    # Actually compute lambda_i cleanly: use M_i v / v (componentwise where v != 0)
    # Better: lambda_i = v^dagger M_i v / v^dagger v (since v is eigenvector)
    lams_clean = np.zeros(k_classes, dtype=complex)
    for ci in range(k_classes):
        Miv = M_list[ci] @ v
        # pick component with largest |v|
        idx = np.argmax(np.abs(v))
        lams_clean[ci] = Miv[idx] / v[idx]
    lams = lams_clean
    # chi(1)^2 = |G| / sum_i |lambda_i|^2 / |C_i|
    csz = np.array([len(classes[ci]) for ci in range(k_classes)])
    denom = np.sum(np.abs(lams)**2 / csz)
    dim2 = N / denom
    dim = np.sqrt(dim2.real)
    dims.append(dim)
    # chi(C_i) = lambda_i * dim / |C_i|
    for ci in range(k_classes):
        char_table[alpha, ci] = lams[ci] * dim / csz[ci]

dims = np.array(dims)
dims_rounded = np.round(dims).astype(int)
print(f"     raw irrep dimensions (unsorted, rounded) = {sorted(dims_rounded.tolist())}")
# Check for valid Burnside
sq_sum = int(np.sum(dims_rounded**2))
print(f"     sum dim^2 = {sq_sum} (expected {N})")
assert sq_sum == N, f"Burnside fails: {sq_sum} != {N}"
# Check the dimension multiset
expected_dims = sorted([1, 2, 2, 3, 3, 4, 4, 5, 6])
assert sorted(dims_rounded.tolist()) == expected_dims, \
    f"Irrep dims are {sorted(dims_rounded.tolist())}, expected {expected_dims}"
print(f"     PASS: irrep dims = {sorted(dims_rounded.tolist())} and sum of squares = 120")

# Snap characters to their rounded dim for the identity column
# Find identity class index
id_class = class_of[e_idx]
# Reorder irreps by dim, then sort within dim-2 and dim-3 and dim-4 by
# a distinguishing character value for determinism.
# For subsequent computations we just need consistent integer-dimension labels.
order_idx = np.argsort(dims_rounded, kind='stable')
dims_sorted = dims_rounded[order_idx]
char_table = char_table[order_idx, :]

print(f"     sorted irrep dims = {dims_sorted.tolist()}")

# Verify orthogonality of character table
print("\n[1f] Verifying character-table orthogonality...")
class_sizes_arr = np.array(class_sizes, dtype=float)
gram = np.zeros((k_classes, k_classes), dtype=complex)
for a in range(k_classes):
    for b in range(k_classes):
        gram[a, b] = np.sum(class_sizes_arr * char_table[a] * char_table[b].conj()) / N
err = np.max(np.abs(gram - np.eye(k_classes)))
print(f"     max |<chi_a, chi_b> - delta_ab| = {err:.3e}")
assert err < 1e-6, "Character orthogonality failed"
print("     PASS: character table is orthonormal")

# ---------------------------------------------------------------------------
# CLAIM 2: j=3/2 rep gives faithful irreducible 8-dim real rep of 2I
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("CLAIM 2: j=3/2 realified is faithful irreducible real 8-dim;")
print("         8_s of SO(8) branches as 3*[1] + [5] under 2I.")
print("=" * 72)

# The character of the j=3/2 rep of SU(2) on a rotation angle alpha is:
#   chi_{3/2}(alpha) = sin(2 alpha) / sin(alpha/2)
# which simplifies to 2 cos(alpha/2) + 2 cos(3 alpha/2).
# For a unit quaternion g = (w, v), it corresponds to rotation by 2*theta where
# w = cos(theta). So alpha = 2*theta, and chi_{3/2}(g) depends only on w.
# Actually in SU(2), a quaternion q = cos(t) + sin(t) u corresponds to an SU(2)
# element with eigenvalues e^{i t}, e^{-i t}. Then j=3/2 eigenvalues are
# e^{3it}, e^{it}, e^{-it}, e^{-3it} and character = 2cos(3t) + 2cos(t).

def char_j(j, w):
    """Character of spin-j SU(2) rep on element with quaternion scalar w = cos(t).
    Returns chi_j = sin((2j+1) t) / sin(t) as a sympy expression,
    equivalently the sum e^{i k t} for k = -2j, -2j+2, ..., 2j."""
    # Use the Chebyshev-style sum: chi_j(t) = sum_{m=-j}^{j} e^{i 2 m t}.
    # With w = cos(t): chi_j = U_{2j}(w) where U_n is Chebyshev of 2nd kind? Actually:
    # chi_j = sin((2j+1) t)/sin(t) = U_{2j}(cos t) for integer 2j.
    # For half-integer j we need cos(t/? ) -- let's just directly use 2j+1 terms.
    # q = cos t + sin t * unit_vec  =>  SU(2) elt has eigenvalues e^{+/- i t}.
    # Spin-j char = sum_{m=-j}^{j} e^{i 2 m t}.
    # For half-integer j, 2m runs over odd integers.
    t = sp.acos(w)
    J2 = int(2*j)
    ms = [sp.Rational(k, 2) for k in range(-J2, J2+1, 2)]
    # Actually 2m for m in -j..j in steps 1 gives -2j, -2j+2, ..., 2j.
    total = sum(sp.cos(2*m*t) for m in ms)  # real part; imaginary parts cancel
    return sp.simplify(total)

# Character of j=3/2 on each conjugacy class (complex dim 4)
print("\n[2a] Building j=3/2 character on each class...")

def w_of(q):
    return q[0]

chi_32 = []
for ci in range(k_classes):
    g = elements[classes[ci][0]]
    w = w_of(g)
    # chi_{3/2}(t) with cos(t) = w:
    # eigenvalues at spin 3/2 are e^{+/- i 3t}, e^{+/- i t}
    # character = 2 cos(3t) + 2 cos(t)
    # cos(3t) = 4 w^3 - 3 w; cos(t) = w
    c3 = 4*w**3 - 3*w
    c1 = w
    chi = sp.simplify(2*c3 + 2*c1)
    chi_32.append(chi)
print(f"     chi_{{3/2}} on classes: {[str(x) for x in chi_32]}")

# Decompose chi_{3/2} over the computed char_table (which has the 9 irreps).
# Inner product: <chi, phi> = (1/|G|) sum_i |C_i| chi(C_i) conj(phi(C_i))
chi_32_vec = np.array([complex(sp.N(c, 30)) for c in chi_32])
decomp_32 = []
for alpha in range(k_classes):
    ip = np.sum(class_sizes_arr * chi_32_vec * char_table[alpha].conj()) / N
    decomp_32.append(ip.real)
decomp_32_rounded = [int(round(x)) for x in decomp_32]
print(f"     dims of irreps =       {dims_sorted.tolist()}")
print(f"     j=3/2 decomposition =  {decomp_32_rounded}")
# j=3/2 is a 4-dim complex irrep — should be exactly one of the 4-dim irreps with coeff 1
nonzero = [(d, c) for d, c in zip(dims_sorted.tolist(), decomp_32_rounded) if c != 0]
print(f"     nonzero components: {nonzero}")
# It must be a single 4-dim irrep
assert len(nonzero) == 1 and nonzero[0][0] == 4 and nonzero[0][1] == 1, \
    "j=3/2 is not a single 4-dim irrep"
# Identify which of the two 4-dim irreps it is
four_indices = [i for i, d in enumerate(dims_sorted.tolist()) if d == 4]
j32_irrep_idx = next(i for i in four_indices if decomp_32_rounded[i] == 1)
print(f"     PASS: j=3/2 is the 4-dim irrep at index {j32_irrep_idx}")

# Now realify: the realification chi_R of a complex irrep chi is chi + chi*
# (as a real rep of real-dim 2n), with character chi_R(g) = 2 Re chi(g).
# Check whether this real rep is irreducible over R.
# Theorem: complex irrep V is:
#   (a) real type -> realification reducible = V_R + V_R
#   (b) complex type -> realification is irreducible over R
#   (c) quaternionic type -> realification is irreducible over R (but V has structure)
# Type is detected by Frobenius-Schur indicator
#   FS(chi) = (1/|G|) sum_g chi(g^2)
# Value +1: real, 0: complex, -1: quaternionic.
def fs_indicator(chi_vec):
    # chi_vec is chi on each class
    # Need chi(g^2) for each g.
    total = 0.0 + 0.0j
    for ci in range(k_classes):
        g = classes[ci][0]
        g2 = cayley[g, g]
        c2 = class_of[g2]
        # chi(g^2) where g ranges over C_i: since g and g' in C_i are conjugate,
        # g'^2 and g^2 are also conjugate (in fact in the same class), so
        # chi(g^2) is constant on C_i.
        total += len(classes[ci]) * chi_vec[c2]
    return total / N

fs_32 = fs_indicator(char_table[j32_irrep_idx])
print(f"     Frobenius-Schur indicator of j=3/2 = {fs_32.real:.6f}")
# j=3/2 is well-known to be quaternionic (indicator -1)
print(f"     (Expected: quaternionic, FS = -1)")
assert abs(fs_32 + 1) < 1e-6, f"FS indicator is {fs_32}, expected -1"
print("     PASS: j=3/2 is quaternionic -> realification is irreducible 8-dim real")

# Faithfulness: the representation is faithful iff its character takes the value
# chi(1) = 4 only on the identity class.
chi_at_id = char_table[j32_irrep_idx, id_class]
print(f"\n[2b] Faithfulness check: chi(1) = {chi_at_id.real:.3f}, chi on other classes:")
faithful = True
for ci in range(k_classes):
    if ci == id_class:
        continue
    val = char_table[j32_irrep_idx, ci]
    if abs(val - chi_at_id) < 1e-6:
        faithful = False
        print(f"       class {ci} (order {class_orders[ci]}): chi = {val}  !! matches identity")
print(f"     Faithful? {faithful}")
assert faithful, "j=3/2 is not faithful"
print("     PASS: j=3/2 is faithful")

# ---------------------------------------------------------------------------
# SO(8) spinor branching via the Spin(8) character formula
# ---------------------------------------------------------------------------
# For g in H acting on R^8 (the vector rep of SO(8)) with eigenvalues
# {e^{+/- i theta_k}} for k = 1..4, the spinor character is
#   chi_{8_s}(g) = (1/2) [ prod_k 2 cos(theta_k/2)
#                        + i^4 prod_k 2 sin(theta_k/2) ]
#               = (1/2) [ prod_k 2 cos(theta_k/2)
#                        + prod_k 2 sin(theta_k/2) ]  (signs depend on lift).
# For our purposes we use the spin lift coming from Spin(8) and compute both
# 8_s and 8_c (they're interchangeable under triality). Using:
#   chi_V(g) = sum_k 2 cos(theta_k)
# For j=3/2 realified on g with quaternion scalar w = cos(t):
# complex eigenvalues are e^{+/- 3 i t}, e^{+/- i t}, each already paired.
# Real rep therefore has eigenpairs (e^{+3it}, e^{-3it}) and (e^{+it}, e^{-it}),
# each occurring WITH MULTIPLICITY 2 (since complex dim 4 -> real dim 8 = 4 pairs).
# Wait: complex rep dim 4 -> real dim 8 as 4 conjugate pairs. The 4 complex
# eigenvalues are already in 2 conjugate pairs (3t, -3t), (t, -t).
# Each conjugate pair e^{+/- i theta} gives one real 2-dim block => 1 angle.
# Realification: 4 complex eigvals -> 4 real 2-dim blocks -> 4 thetas.
# Each complex conjugate pair is ONE 2-plane in the complex rep but becomes TWO
# 2-planes in the realification? Let's think again.
# If complex rep M on C^n has eigenvalues lambda_1,...,lambda_n, then realification
# on R^{2n} has the same multiset of eigenvalues, counted as complex eigenvalues
# of the real 2n x 2n matrix. In real form, each conjugate pair (lambda, bar lambda)
# contributes one 2 x 2 rotation block. So:
#   complex eigvals {e^{3it}, e^{it}, e^{-it}, e^{-3it}}
# pair as (e^{3it}, e^{-3it}) and (e^{it}, e^{-it}) -- that's 2 pairs.
# Realification is 8-dim, so it should have 4 pairs. Where do the other 2 come from?
# The j=3/2 rep is already unitary on C^4, so the real underlying space is R^8 with
# complex structure J. The eigenvalues of the SAME element on R^8 (as a real vector
# space) are the SAME multiset of complex numbers {e^{3it}, e^{-3it}, e^{it}, e^{-it}},
# but each element now counts with real-multiplicity 2 (since dim_R = 2 dim_C).
# So as real operator, eigenvalues are e^{+/- 3it} (mult 2) and e^{+/- it} (mult 2).
# That gives 4 conjugate pairs and 4 theta-values (3t, 3t, t, t). Correct.
print("\n[2c] Computing 8_s branching under 2I (via j=3/2 realification as R^8)...")

# For each class, compute theta list (3t, 3t, t, t) where cos(t) = w.
# Then chi_{8_s} via spinor character formula.
# We use sympy exactly.

def spinor_characters(theta_list):
    """Given 4 angles, return (chi_V, chi_8s, chi_8c) on SO(8) lift.
    chi_V = sum 2 cos(theta_k)
    chi_8s = prod cos(theta_k/2)*2 / 2 + prod sin(theta_k/2)*2 / 2
    (The prefactor 2 per factor comes from the dim-2 block.)
    More precisely, Spin(8) weights for 8_s are (eps_1/2,...,eps_4/2) with
    product eps_1 eps_2 eps_3 eps_4 = +1; for 8_c product = -1.
    chi_{8_s} = sum_{even # -} exp(i (eps . theta)/2)
             = (1/2)[prod_k(e^{i th_k/2}+e^{-i th_k/2})
                   + prod_k(e^{i th_k/2}-e^{-i th_k/2})]
             = (1/2)[prod_k 2 cos(th_k/2) + (2i)^4 prod_k sin(th_k/2)/(2^4?) ]
    Careful expansion: prod(e^{i a_k}-e^{-i a_k}) = prod(2 i sin a_k) = (2i)^4 prod sin a_k
    = 16 prod sin a_k (real).
    prod(e^{i a_k}+e^{-i a_k}) = prod(2 cos a_k) = 16 prod cos a_k.
    chi_{8_s} = (1/2)(16 prod cos(th/2) + 16 prod sin(th/2))
              = 8 prod cos(th/2) + 8 prod sin(th/2).
    Hmm but then chi_{8_s}(identity) = 8*1 + 8*0 = 8. Correct (dim of 8_s).
    chi_V(identity) = 8. Correct.
    """
    chi_V = sum(2 * sp.cos(th) for th in theta_list)
    cos_prod = sp.Mul(*[sp.cos(th/2) for th in theta_list])
    sin_prod = sp.Mul(*[sp.sin(th/2) for th in theta_list])
    chi_8s = 8 * cos_prod + 8 * sin_prod
    chi_8c = 8 * cos_prod - 8 * sin_prod
    return sp.simplify(chi_V), sp.simplify(chi_8s), sp.simplify(chi_8c)

# For each class we need theta_k. For an element with quaternion scalar w,
# the 2I element as SU(2) has eigenvalues e^{+/- i t} with cos t = w.
# We solve for t numerically in [0, pi], then use theta = (3t, 3t, t, t).
# But for exact character arithmetic with the realified j=3/2 rep,
# we can just directly plug cos(3t/2), sin(3t/2), cos(t/2), sin(t/2) where
# cos t = w, and use half-angle: cos(t/2) = sqrt((1+w)/2), sin(t/2)=sqrt((1-w)/2)
# with appropriate sign. However the lift to Spin(8) has a sign ambiguity on
# pairs of theta's (because theta -> theta + 2 pi changes exp(i theta/2) sign).
# Since 2I is a binary (= already a double cover) group, one natural lift sends
# the quaternion itself to its image in Spin(8). For the 8_s character we
# therefore choose signs consistently with the ambient SU(2) double cover.
#
# A clean way: the Spin lift of the image of g under the 8-dim real rep
# is obtained by choosing half-angles that respect the group law. For the
# realification of j=3/2, the natural half-angles are 3t/2 and t/2 where t
# is the SU(2) half-angle (i.e. q = cos(t) + sin(t) u means the SO(3) rotation
# angle is 2t). Then (cos(3t/2), cos(t/2)) are real algebraic numbers of w.
#
# The subtlety: in the realification R^8, each complex eigenvalue e^{+/- i theta_k}
# gives a 2-plane on which g rotates by theta_k in SOME orientation. The spinor
# character depends on the "oriented" half-angle. Because j=3/2 is a well-defined
# SU(2) rep, the lift to Spin(8) is canonical up to a global 8_s <-> 8_c swap.
#
# Practical approach: compute chi_{8_s} on each class using theta = (3t,3t,t,t)
# and half-angles (3t/2, 3t/2, t/2, t/2) with a consistent choice of sign. Verify
# that the result is a valid character (sum with chi_8c = chi_V of rank-16 Cliff...)
# and decomposes over 2I irreps with non-negative integer multiplicities.
# If a sign is wrong, some multiplicity will be negative — we then flip that class's
# half-angle sign.

# Build theta = (3t, 3t, t, t) per class as exact sympy.
print("     theta lists per class (as functions of SU(2) half-angle t with cos t = w):")
V_chars = []
S_chars = []
C_chars = []
for ci in range(k_classes):
    g = elements[classes[ci][0]]
    w = sp.nsimplify(sp.sympify(g[0]), [sp.sqrt(5)])
    # SU(2) half-angle t with cos t = w
    # Use sympy acos then reduce
    t = sp.acos(w)
    # Simplify common values
    t = sp.simplify(t)
    theta_list = [3*t, 3*t, t, t]
    chi_V, chi_s, chi_c = spinor_characters(theta_list)
    # Simplify
    chi_V = sp.nsimplify(sp.simplify(chi_V), [sp.sqrt(5)], rational=False)
    chi_s = sp.nsimplify(sp.simplify(chi_s), [sp.sqrt(5)], rational=False)
    chi_c = sp.nsimplify(sp.simplify(chi_c), [sp.sqrt(5)], rational=False)
    V_chars.append(chi_V)
    S_chars.append(chi_s)
    C_chars.append(chi_c)
    print(f"       class {ci} (size {class_sizes[ci]}, order {class_orders[ci]}): "
          f"w={w}, chi_V={chi_V}, chi_s={chi_s}, chi_c={chi_c}")

# Sanity check: chi_V on identity should be 8
assert sp.simplify(V_chars[id_class] - 8) == 0

# Double-check: chi_V is the character of the realified j=3/2, which is
# 2 * Re(chi_{3/2}) = 2 * chi_{3/2} (since j=3/2 has real character for 2I
# when the rep is quaternionic? Actually chi_{3/2} is always real because
# eigenvalues come in conjugate pairs). Let's verify chi_V matches 2*chi_{3/2}.
print("\n     Cross-check: chi_V (realification character) =?= 2 * chi_{3/2}")
ok = True
for ci in range(k_classes):
    diff = sp.simplify(V_chars[ci] - 2 * chi_32[ci])
    if diff != 0:
        ok = False
        print(f"       class {ci}: diff = {diff}")
assert ok, "Realification character doesn't match 2*chi_{3/2}"
print("     PASS: realification character matches.")

# Decompose chi_8s over 2I irreps
def decompose(chi_vec_sym):
    """Decompose a character (sympy-valued on each class) over the computed irreps."""
    # numeric values
    vec = np.array([complex(sp.N(c, 40)) for c in chi_vec_sym])
    mults = []
    for alpha in range(k_classes):
        ip = np.sum(class_sizes_arr * vec * char_table[alpha].conj()) / N
        mults.append(ip)
    return mults

print("\n[2d] Decomposing 8_s (and 8_c) over 2I irreps:")
mults_s = decompose(S_chars)
mults_c = decompose(C_chars)
print(f"     irrep dims:        {dims_sorted.tolist()}")
print(f"     8_s multiplicities: {[f'{m.real:+.3f}' for m in mults_s]}")
print(f"     8_c multiplicities: {[f'{m.real:+.3f}' for m in mults_c]}")

mults_s_round = [int(round(m.real)) for m in mults_s]
mults_c_round = [int(round(m.real)) for m in mults_c]
# Save for the final summary (variables get reused later)
j32_mults_s = list(mults_s_round)
j32_mults_c = list(mults_c_round)
err_s = max(abs(m.real - r) + abs(m.imag) for m, r in zip(mults_s, mults_s_round))
err_c = max(abs(m.real - r) + abs(m.imag) for m, r in zip(mults_c, mults_c_round))
print(f"     integrality error (8_s): {err_s:.2e}")
print(f"     integrality error (8_c): {err_c:.2e}")
# At least one of 8_s, 8_c should yield non-negative integer multiplicities.
# If both are valid, our theta-sign convention is consistent and the two
# correspond to the two Spin(8) spinor branchings.

def pretty_decomp(mults_round):
    parts = []
    for d, m in zip(dims_sorted.tolist(), mults_round):
        if m != 0:
            parts.append(f"{m}*[{d}]")
    return " + ".join(parts) if parts else "0"

print(f"     8_s = {pretty_decomp(mults_s_round)}")
print(f"     8_c = {pretty_decomp(mults_c_round)}")

# Expected: 3 singlets and one 5-dim irrep in 8_s.
singlets_s = sum(m for d, m in zip(dims_sorted.tolist(), mults_s_round) if d == 1)
fives_s = sum(m for d, m in zip(dims_sorted.tolist(), mults_s_round) if d == 5)
singlets_c = sum(m for d, m in zip(dims_sorted.tolist(), mults_c_round) if d == 1)
fives_c = sum(m for d, m in zip(dims_sorted.tolist(), mults_c_round) if d == 5)
print(f"     8_s: {singlets_s} singlets, {fives_s} [5]'s")
print(f"     8_c: {singlets_c} singlets, {fives_c} [5]'s")

# Pick whichever spinor has the 3+[5] branching and call it 8_s.
if singlets_s == 3 and fives_s == 1:
    s_branch = mults_s_round
    branch_name = "8_s"
elif singlets_c == 3 and fives_c == 1:
    s_branch = mults_c_round
    branch_name = "8_c (relabeled as 8_s)"
else:
    s_branch = None
    branch_name = None

claim2_ok = (s_branch is not None)
if claim2_ok:
    print(f"     PASS: {branch_name} branching = 3*[1] + [5]")
else:
    print("     FAIL: neither 8_s nor 8_c has 3 singlets + [5]")

# Also sanity-check by recomputing with absolute-value convention: in fact,
# the |chi_{8_s}| and |chi_{8_c}| as multisets are the same; only the labels
# differ. So the physics-relevant statement "one of the two Spin(8) spinors
# has 3 singlets + [5]" is what we verify.

# ---------------------------------------------------------------------------
# CLAIM 3: uniqueness of j=3/2 among faithful real 8-dim reps of 2I.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("CLAIM 3: j=3/2 is the unique faithful real 8-dim rep of 2I")
print("         whose 8_s branching contains exactly 3 singlets.")
print("=" * 72)

# Enumerate all decompositions into real irreps of 2I with total real-dim = 8.
# First classify the 2I irreps by real type (R, C, H) via FS indicator.
fs_all = []
for alpha in range(k_classes):
    fs = fs_indicator(char_table[alpha])
    fs_all.append(fs.real)
print(f"\n[3a] FS indicators by irrep dim:")
for d, f in zip(dims_sorted.tolist(), fs_all):
    type_name = {1: "real", 0: "complex", -1: "quaternionic"}[int(round(f))]
    print(f"      dim {d}: FS = {f:+.3f} ({type_name})")

# Real irreducible real reps of 2I:
# - type R (FS=+1): the complex irrep V has a real form V_R of dim = dim_C(V)
# - type C (FS=0): V and V* are distinct; real irreducible = V + V* as real rep,
#     with real dim = 2 * dim_C(V)
# - type H (FS=-1): realification of V is irreducible over R with real dim = 2 * dim_C(V)
# Complex-type irreps pair up: chi and conj(chi).
# We need to figure out which complex-type irreps pair with which.

# Find complex-type pairs
complex_pairs = []  # list of (i, j) pairs with chi_j = conj(chi_i)
used = set()
for i in range(k_classes):
    if abs(fs_all[i]) > 0.5:  # real or quaternionic
        continue
    if i in used:
        continue
    # find j with chi_j = conj(chi_i)
    for j in range(k_classes):
        if j == i:
            continue
        if abs(fs_all[j]) > 0.5:
            continue
        if np.max(np.abs(char_table[j] - char_table[i].conj())) < 1e-6:
            complex_pairs.append((i, j))
            used.add(i); used.add(j)
            break
print(f"     complex-conjugate irrep pairs: {complex_pairs}")

# Enumerate all real 8-dim representations:
# Each real-irreducible piece has one of:
#   - dim d, from a type-R irrep of dim d
#   - dim 2d, from a type-C irrep pair (contributes V + V*)
#   - dim 2d, from a type-H irrep (realification)
# Build list of "real-irreducible real reps" with their dim and which complex
# irreps they decompose into (for 8_s branching we need the complex character).
# (real_irr_list is built correctly below in [3d] with the H-type "2 copies on
# complexification" convention.)


# Enumerate all multisets of real-irr pieces totaling 8.
from functools import lru_cache

def enumerate_partitions(pieces, target):
    """Enumerate multisets of pieces (list of (real_dim, mult_vec, label) tuples)
    whose real_dims sum to target.
    Returns list of (usage-count tuples aligned with pieces)."""
    results = []
    n = len(pieces)
    dims_only = [p[0] for p in pieces]

    def recurse(idx, remaining, chosen):
        if remaining == 0:
            # pad with zeros
            results.append(tuple(chosen + [0] * (n - idx)))
            return
        if idx == n:
            return
        d = dims_only[idx]
        max_use = remaining // d if d > 0 else 0
        for use in range(max_use + 1):
            chosen.append(use)
            recurse(idx + 1, remaining - use * d, chosen)
            chosen.pop()

    recurse(0, target, [])
    return results

# (The actual partition enumeration happens after real_irr_list is finalized below.)

# For each partition, compute:
#   - complex multiplicity vector over the 9 irreps (sum of contributing pieces)
#   - real character chi_V on classes
#   - check faithfulness
#   - compute 8_s branching via the spinor formula
# For the 8_s branching we need an 8-dim real rep of 2I and its image in SO(8).
# The spinor character depends on the rotation angles on each 2-plane, which
# depend on the actual real structure, not merely the complex decomposition.
# HOWEVER: given the complex character chi_V (real-valued since V is real),
# the rotation-angle multiset is determined by the complex eigenvalues of V(g),
# which are determined by the complex multiplicities in the decomposition.
# So chi_{8_s} depends only on the multiplicity pattern (up to the overall
# Spin(8) lift sign which gives 8_s <-> 8_c).
#
# For each element g of 2I, V(g) has complex eigenvalues determined by
# (irrep chi on g, multiplicity). These eigenvalues come in conjugate pairs
# (since V is a real rep, its complex eigenvalues are conjugate-closed); then
# theta_k is the set of argument magnitudes.
#
# We can extract chi(g^{1/2}) implicitly by using the Spin(8) character formula
# as a function of the *fundamental* eigenvalues. An efficient way:
# sum_k 2 cos(theta_k/2) + sum_k 2 sin(theta_k/2) terms; note that
# chi_{8_s}(g) + chi_{8_c}(g) = 8 * prod cos(theta_k/2)
# and this product only involves the eigenvalue multiset. We can compute it
# per element g by diagonalizing V(g).
#
# BUT for our 2I action, we don't need actual matrices; we can just find the
# eigenvalue multiset of V(g) from the character: the eigenvalues of V(g) are
# the multiset union of {eigenvalues of each constituent irrep on g} weighted
# by multiplicity. Each irrep, restricted to g, acts as diagonalizable over C
# with eigenvalues on the unit circle; the multiset of those eigenvalues is
# determined purely by the character of that irrep on <g>, which we can compute
# per-element using chi(g^k) for k = 0, 1, ..., order(g)-1.

# Let's build, for each class c and each irrep alpha, the eigenvalue multiset
# of rho_alpha(g) for g in c. Since characters are class functions and we
# pick a representative g_c, the eigenvalues of rho_alpha(g_c) are determined
# by its character on powers: chi(g), chi(g^2), ..., chi(g^{ord(g)-1}).
# If g has order n, its eigenvalues are n-th roots of unity; the multiplicity
# of zeta_n^k is (1/n) sum_j chi(g^j) zeta_n^{-jk}.

def eigs_of_rho_on_g(alpha, g_idx):
    """Return complex eigenvalues of rho_alpha(g) as a multiset (array of complex numbers)."""
    n = order(g_idx)
    # chi(g^j) for j = 0..n-1
    xs = []
    x = e_idx
    for j in range(n):
        xs.append(char_table[alpha, class_of[x]])
        x = cayley[x, g_idx]
    # Fourier: mult of e^{2 pi i k / n} is (1/n) sum_j chi(g^j) e^{-2 pi i j k / n}
    eigs = []
    for k in range(n):
        zk = np.exp(-2j * np.pi * k / n)
        mult = (1.0 / n) * sum(xs[j] * zk**j for j in range(n))
        m = int(round(mult.real))
        if abs(mult.imag) > 1e-6 or abs(mult.real - m) > 1e-4:
            # numerical noise — but mult should be a non-negative integer
            m = max(0, m)
        for _ in range(m):
            eigs.append(np.exp(2j * np.pi * k / n))
    # Sanity: total count equals dim of rho
    d = int(round(char_table[alpha, id_class].real))
    if len(eigs) != d:
        # try with more permissive rounding
        eigs = []
        for k in range(n):
            zk = np.exp(-2j * np.pi * k / n)
            mult = (1.0 / n) * sum(xs[j] * zk**j for j in range(n))
            m = int(round(mult.real))
            for _ in range(max(0, m)):
                eigs.append(np.exp(2j * np.pi * k / n))
        if len(eigs) != d:
            raise RuntimeError(f"eigendecomposition count mismatch: got {len(eigs)}, expected {d}")
    return eigs

# Precompute eigenvalues of each irrep on each class representative
eigs_per_alpha_per_class = []
for alpha in range(k_classes):
    rec = []
    for ci in range(k_classes):
        g = classes[ci][0]
        rec.append(eigs_of_rho_on_g(alpha, g))
    eigs_per_alpha_per_class.append(rec)

def spinor_branch_multiplicities(mult_vec):
    """Given a complex-irrep multiplicity vector summing to 8 real-dim
    (equivalent to complex-dim when counting sympy rep, but here the real-dim
    constraint is encoded by the partition already), compute 8_s and 8_c
    branching multiplicities over 2I irreps.
    chi_{8_s}(g) = 8 * prod_k cos(theta_k / 2) + 8 * prod_k sin(theta_k / 2)
    chi_{8_c}(g) = 8 * prod_k cos(theta_k / 2) - 8 * prod_k sin(theta_k / 2)
    with the 4 thetas coming from the eigenvalue multiset of V(g), pairing
    conjugate eigenvalues into 2-planes of rotation angle theta.
    """
    chi_s = np.zeros(k_classes, dtype=complex)
    chi_c = np.zeros(k_classes, dtype=complex)
    for ci in range(k_classes):
        # Build the complex eigenvalue multiset of V(g)
        eigs = []
        for alpha, m in enumerate(mult_vec):
            if m == 0:
                continue
            e = eigs_per_alpha_per_class[alpha][ci]
            eigs.extend(e * m)
        # V must be a real rep, so complex eigenvalues should be conjugate-closed.
        # Pair them into 4 pairs (real_dim = 2 * #pairs; we want 4 pairs from
        # complex_dim = ... wait). Careful: the input mult_vec is a COMPLEX
        # multiplicity vector. The TOTAL COMPLEX DIMENSION of V as a complex rep
        # is sum m_i dim_i. For a real 8-dim rep, when we complexify:
        #   type R (dim d, used once) -> complex dim d, counted once
        #   type C pair (dim d each, used once each) -> complex dim 2d
        #   type H (dim d, used once) -> complex dim d (realified then complexified = d+d)
        # Hmm, this needs care. Let me re-derive.
        # A REAL rep V_R has complex dim (V_R tensor C) = real_dim. So the complexification
        # of an 8-dim real rep is an 8-dim complex rep.
        # Real pieces:
        #   type R: irrep chi of complex dim d has a real form V_R of real dim d;
        #     its complexification is chi (complex dim d).
        #   type C pair (chi, chi*): real-irreducible real rep has real dim 2d,
        #     complexification = chi + chi*, complex dim 2d.
        #   type H: quaternionic irrep of complex dim d; the real-irreducible real
        #     rep is the realification (real dim 2d), complexification = chi + chi
        #     (i.e., 2 copies of chi), complex dim 2d.
        # So mult_vec in our real_irr_list encoding:
        #   type R piece [d]_R: mult_vec has chi:1
        #   type C piece [d]_C+[d]_C*: mult_vec has chi:1, chi*:1
        #   type H piece [d]_H: mult_vec has chi:2 (two copies of the quaternionic irrep)
        # This is exactly what we want to track for spinor characters.
        # Fix: when building real_irr_list for H-type we should set mult 2, not 1.
        pass
    return chi_s, chi_c  # placeholder; see below

# NOTE: needed to correct real_irr_list for H-type (2 copies in complexification).
real_irr_list = []
for alpha in range(k_classes):
    if abs(fs_all[alpha] - 1) < 0.5:
        d = int(dims_sorted[alpha])
        mult_vec = np.zeros(k_classes, dtype=int)
        mult_vec[alpha] = 1
        real_irr_list.append((d, mult_vec, f"[{d}]_R"))
    elif abs(fs_all[alpha] + 1) < 0.5:
        d = int(dims_sorted[alpha])
        mult_vec = np.zeros(k_classes, dtype=int)
        mult_vec[alpha] = 2   # complexification = 2 copies
        real_irr_list.append((2*d, mult_vec, f"[{d}]_H (realified)"))
for (i, j) in complex_pairs:
    d = int(dims_sorted[i])
    mult_vec = np.zeros(k_classes, dtype=int)
    mult_vec[i] = 1
    mult_vec[j] = 1
    real_irr_list.append((2*d, mult_vec, f"[{d}]_C + [{d}]_C*"))

print(f"\n[3d] Corrected real-irreducible pieces (complex-mult encoding):")
for rd, mv, label in real_irr_list:
    print(f"      {label}: real_dim = {rd}, complex_mult_vec = {mv.tolist()}")

partitions = enumerate_partitions(real_irr_list, 8)
print(f"     Number of multisets summing to real_dim 8: {len(partitions)}")

def partition_complex_mult(p):
    v = np.zeros(k_classes, dtype=int)
    for use, (rd, mv, lbl) in zip(p, real_irr_list):
        v += use * mv
    return v

# Now compute 8_s branching via the spinor formula given the total complex eigenvalue
# multiset of V_C = complexified V.

def spinor_characters_from_mult(mult_vec):
    chi_s_vals = np.zeros(k_classes, dtype=complex)
    chi_c_vals = np.zeros(k_classes, dtype=complex)
    chi_V_vals = np.zeros(k_classes, dtype=complex)
    for ci in range(k_classes):
        eigs = []
        for alpha, m in enumerate(mult_vec):
            if m == 0:
                continue
            eigs.extend(eigs_per_alpha_per_class[alpha][ci] * m)
        # eigs should be a multiset of 8 unit complex numbers, conjugate-closed
        assert len(eigs) == 8, f"eig count {len(eigs)} != 8"
        # Pair them into 4 conjugate pairs. Each pair gives theta_k = arg of one eig.
        eigs = list(eigs)
        thetas = []
        # Match conjugates with tolerance
        eps = 1e-6
        used = [False]*8
        for i in range(8):
            if used[i]:
                continue
            ei = eigs[i]
            # find a partner
            if abs(ei.imag) < eps:
                # real eigenvalue: +/- 1
                # treat as a pair with itself (theta = 0 or pi)
                used[i] = True
                # find another real eig with same value (a real dim-1 real eigenspace
                # pairs with another copy to form a 2-plane of rotation by 0 or pi)
                for j in range(i+1, 8):
                    if used[j]:
                        continue
                    if abs(eigs[j] - ei) < eps:
                        used[j] = True
                        thetas.append(0.0 if ei.real > 0 else np.pi)
                        break
                else:
                    raise RuntimeError("unmatched real eigenvalue (rep not real?)")
            else:
                # complex eigenvalue: find its conjugate
                for j in range(i+1, 8):
                    if used[j]:
                        continue
                    if abs(eigs[j] - np.conj(ei)) < eps:
                        used[i] = True; used[j] = True
                        thetas.append(float(np.angle(ei)))  # theta in (-pi, pi]
                        break
                else:
                    raise RuntimeError("unmatched complex eigenvalue")
        assert len(thetas) == 4
        # Spinor characters
        cos_prod = np.prod([np.cos(t/2) for t in thetas])
        sin_prod = np.prod([np.sin(t/2) for t in thetas])
        # Use absolute-value convention: the Spin(8) lift is defined up to a global
        # 8_s <-> 8_c swap and individual pair-sign flips that change theta by 2pi
        # (giving sign flips on half-angles). We compute both candidates
        # chi_s = 8 cos_prod + 8 sin_prod and chi_c = 8 cos_prod - 8 sin_prod.
        chi_s_vals[ci] = 8 * cos_prod + 8 * sin_prod
        chi_c_vals[ci] = 8 * cos_prod - 8 * sin_prod
        chi_V_vals[ci] = sum(2 * np.cos(t) for t in thetas)
    return chi_V_vals, chi_s_vals, chi_c_vals

def decompose_vec(chi_vec_num):
    mults = []
    for alpha in range(k_classes):
        ip = np.sum(class_sizes_arr * chi_vec_num * char_table[alpha].conj()) / N
        mults.append(ip)
    return mults

def faithful_check_complex_mult(mult_vec):
    """Faithful iff chi_V only equals chi_V(1) on the identity class."""
    chi_id = 0
    for alpha, m in enumerate(mult_vec):
        chi_id += m * char_table[alpha, id_class].real
    for ci in range(k_classes):
        if ci == id_class:
            continue
        chi_c_val = 0
        for alpha, m in enumerate(mult_vec):
            chi_c_val += m * char_table[alpha, ci]
        if abs(chi_c_val - chi_id) < 1e-6:
            return False
    return True

print("\n[3e] Enumerating all real 8-dim reps and their 8_s branchings...")

candidates = []
for p in partitions:
    mv = partition_complex_mult(p)
    # real-rep consistency check: mv must be conjugate-symmetric (chi <-> chi*)
    # For type-R/H this is automatic since chi is real. Fine.
    chi_V, chi_s, chi_c = spinor_characters_from_mult(mv)
    # Decompose both spinors over 2I irreps
    mults_s = decompose_vec(chi_s)
    mults_c = decompose_vec(chi_c)
    # Round
    mults_s_round = [int(round(m.real)) for m in mults_s]
    mults_c_round = [int(round(m.real)) for m in mults_c]
    err_s = max(abs(m.real - r) + abs(m.imag) for m, r in zip(mults_s, mults_s_round))
    err_c = max(abs(m.real - r) + abs(m.imag) for m, r in zip(mults_c, mults_c_round))
    is_faithful = faithful_check_complex_mult(mv)
    # Singlet counts
    sing_s = sum(m for d, m in zip(dims_sorted.tolist(), mults_s_round) if d == 1)
    sing_c = sum(m for d, m in zip(dims_sorted.tolist(), mults_c_round) if d == 1)
    all_nonneg_s = all(m >= 0 for m in mults_s_round) and err_s < 1e-3
    all_nonneg_c = all(m >= 0 for m in mults_c_round) and err_c < 1e-3
    # Label
    label_parts = []
    for use, (rd, _, lbl) in zip(p, real_irr_list):
        if use > 0:
            label_parts.append(f"{use}*{lbl}")
    label = " + ".join(label_parts) if label_parts else "0"
    candidates.append({
        "partition": p,
        "label": label,
        "complex_mult": mv.tolist(),
        "faithful": is_faithful,
        "mults_s": mults_s_round,
        "mults_c": mults_c_round,
        "singlets_s": sing_s,
        "singlets_c": sing_c,
        "valid_s": all_nonneg_s,
        "valid_c": all_nonneg_c,
    })

print(f"     total partitions examined: {len(candidates)}")
print(f"\n     {'idx':>3} {'faith':>5} {'sing_s':>6} {'sing_c':>6} {'label':<45}")
for i, c in enumerate(candidates):
    print(f"     {i:>3} {str(c['faithful']):>5} {c['singlets_s']:>6} "
          f"{c['singlets_c']:>6} {c['label']:<45}")

# Pick out faithful reps with exactly 3 singlets in EITHER 8_s or 8_c.
three_singlet_faithful = [
    c for c in candidates
    if c['faithful'] and (c['singlets_s'] == 3 or c['singlets_c'] == 3)
]
print(f"\n     Faithful real 8-dim reps with 3 singlets (in 8_s or 8_c): "
      f"{len(three_singlet_faithful)}")
for c in three_singlet_faithful:
    print(f"       label: {c['label']}")
    print(f"         8_s = {pretty_decomp(c['mults_s'])}  (sing={c['singlets_s']})")
    print(f"         8_c = {pretty_decomp(c['mults_c'])}  (sing={c['singlets_c']})")

# j=3/2 realification is the [4]_H piece alone (if that 4-dim irrep is quaternionic).
# Which 4-dim irrep? Both 4-dim irreps need to be checked. In 2I, the j=3/2 rep
# is the one with Frobenius-Schur -1 (quaternionic). Let's report.
fs_by_dim = [(d, fs_all[i]) for i, d in enumerate(dims_sorted.tolist())]
print(f"\n     FS by irrep: {fs_by_dim}")

uniqueness_verdict = (len(three_singlet_faithful) == 1)
print(f"\n     Uniqueness of 3-singlet faithful rep: "
      f"{'UNIQUE' if uniqueness_verdict else 'NOT UNIQUE'}")
if not uniqueness_verdict:
    print("     --> Claim 3 is FALSE as stated (multiple faithful 8-dim reps give 3 singlets)")
else:
    print("     --> Claim 3 holds: j=3/2 is unique among faithful 8-dim reps.")

# ---------------------------------------------------------------------------
# CLAIM 4: invariants in 2 tensor 2 tensor Sym^n(3) of 2I.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("CLAIM 4: dim((2 tensor 2 tensor Sym^n(3))^{2I}) = 1 for n=0..4, >=2 for n=5.")
print("=" * 72)

# Identify the "2" and "3" irreps. 2I has two 2-dim irreps (j=1/2 and j=3/2
# projected to... wait no, 2I's 2-dim irreps are both 2-dim).
# The natural "2" in the physicist's notation for 2I is j=1/2 (the defining SU(2) rep).
# 2I has two 2-dim irreps labeled 2 and 2': they are the j=1/2 and the "other" one
# (corresponding to j=1/2 twisted by the outer automorphism of 2I induced by
# phi <-> 1-phi). Similarly 2I has two 3-dim irreps 3 and 3' (j=1 vs its twist).
# For the claim to make sense we should identify the specific 2 and 3 irreps.
# Physics typically picks: "2" = j=1/2 (SU(2) defining), "3" = j=1 (vector of SO(3)).
# The j=1/2 character: chi(g) = trace of g as SU(2) matrix = 2*w where w = q_0.
# The j=1 character: chi(g) = (trace of g as SO(3) matrix) = 1 + 2*cos(2t) = ... = 4w^2 - 1.

print("\n[4a] Identifying the '2' (= j=1/2) and '3' (= j=1) irreps among 2I's irreps...")

# j=1/2 character: 2*w for each class
chi_half = []
for ci in range(k_classes):
    g = elements[classes[ci][0]]
    w = g[0]
    chi_half.append(sp.nsimplify(sp.sympify(2*w), [sp.sqrt(5)], rational=False))

# j=1 character: 4 w^2 - 1
chi_one = []
for ci in range(k_classes):
    g = elements[classes[ci][0]]
    w = g[0]
    chi_one.append(sp.nsimplify(sp.sympify(4*w**2 - 1), [sp.sqrt(5)], rational=False))

# Identify which irrep each is
def match_irrep(chi_vec_sym):
    chi_num = np.array([complex(sp.N(c, 30)) for c in chi_vec_sym])
    for alpha in range(k_classes):
        if np.max(np.abs(char_table[alpha] - chi_num)) < 1e-6:
            return alpha
    return None

idx_2 = match_irrep(chi_half)
idx_3 = match_irrep(chi_one)
print(f"     j=1/2 irrep index: {idx_2} (dim {dims_sorted[idx_2]})")
print(f"     j=1 irrep index:   {idx_3} (dim {dims_sorted[idx_3]})")
assert idx_2 is not None and int(dims_sorted[idx_2]) == 2, "couldn't find 2-dim j=1/2"
assert idx_3 is not None and int(dims_sorted[idx_3]) == 3, "couldn't find 3-dim j=1"

# Sym^n of a 3-dim rep on an element g with eigenvalues (a, b, c):
# char of Sym^n(V) on g = h_n(a, b, c) = complete homogeneous symmetric polynomial.
# Equivalently, [z^n] 1/((1-a z)(1-b z)(1-c z)).
# We compute character of Sym^n(j=1) per class via direct formula with symbolic sympy.

def chi_symn_on_class(alpha, ci, n):
    """Character of Sym^n(rho_alpha) on class ci (symbolic sympy)."""
    g = classes[ci][0]
    eigs = eigs_of_rho_on_g(alpha, g)   # numeric complex eigenvalues
    # h_n(eigs) = sum over multisets of size n of product
    # For small n (up to 6) this is fine. Use generating function:
    #   h_n = [z^n] prod_k 1/(1 - lambda_k z)
    # Numerically:
    from math import comb
    # Use the Newton-Girard-like direct sum via combinations-with-replacement.
    d = len(eigs)
    # Use: h_n = sum_{indices i1 <= i2 <= ... <= in} prod lambda_{i_j}
    # That's combinatorially small: comb(d+n-1, n).
    from itertools import combinations_with_replacement
    total = 0.0 + 0.0j
    for combo in combinations_with_replacement(range(d), n):
        p = 1.0 + 0j
        for idx in combo:
            p *= eigs[idx]
        total += p
    return total

print("\n[4b] Computing invariants in 2 (x) 2 (x) Sym^n(3) for n = 0..6:")
print(f"     {'n':>3} {'dim_inv':>8}")
results_claim4 = {}
for n in range(0, 7):
    total = 0.0 + 0.0j
    for ci in range(k_classes):
        # chi on class: chi_2(ci) * chi_2(ci) * chi_{Sym^n 3}(ci)
        chi2 = char_table[idx_2, ci]
        chisym = chi_symn_on_class(idx_3, ci, n)
        total += class_sizes_arr[ci] * chi2 * chi2 * chisym
    dim_inv = total / N
    dim_inv_rounded = int(round(dim_inv.real))
    results_claim4[n] = dim_inv_rounded
    flag = ""
    expected = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: (">=2",), 6: None}
    print(f"     {n:>3} {dim_inv_rounded:>8}   (complex value: {dim_inv.real:.6f} + {dim_inv.imag:.2e}i)")

# Actually physics claim: the 2 tensor 2 tensor Sym^n(3) invariant dim.
# But wait — 2 tensor 2 decomposes over 2I as either 1 + 3 (if 2 is selfdual)
# or 2*1 + 3 depending on spin structure. For SU(2): j=1/2 tensor j=1/2 = j=0 + j=1.
# Restricted to 2I: 2 tensor 2 = 1 + 3 (in the j=0 + j=1 sense).
# Then 2 tensor 2 tensor Sym^n(3) invariants = dim[(1 + 3) tensor Sym^n 3]^{2I}
# = dim(Sym^n 3)^{2I} + dim(3 tensor Sym^n 3)^{2I}.
# These are known generating functions (Molien series!) and
# give: for the binary icosahedral / H_3 Coxeter invariants the generators are
# at degrees 2, 6, 10 (for the j=1 rep invariants). So Sym^n(3)^{2I} generating
# function is 1/((1-t^2)(1-t^6)(1-t^10)).
# And the covariants in 3 are generated freely at degrees 15 (or so).

# Let's verify the claim as-stated.
print("\n     Claim 4 verdict:")
ok4 = True
for n in [0, 1, 2, 3, 4]:
    exp = 1
    got = results_claim4[n]
    status = "PASS" if got == exp else "FAIL"
    print(f"       n={n}: dim_inv = {got} (expected {exp}) {status}")
    if got != exp:
        ok4 = False
n5 = results_claim4[5]
status5 = "PASS" if n5 >= 2 else "FAIL"
print(f"       n=5: dim_inv = {n5} (expected >=2) {status5}")
if n5 < 2:
    ok4 = False
print(f"     Claim 4: {'PASS' if ok4 else 'FAIL'}")

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

summary = {}
summary["|2I|"] = 120
summary["num_conjugacy_classes"] = len(classes)
summary["class_sizes"] = class_sizes
summary["irrep_dims"] = dims_sorted.tolist()
summary["sum_dim_sq"] = sq_sum
summary["FS_indicators"] = [{"dim": int(d), "fs": float(f)}
                            for d, f in zip(dims_sorted.tolist(), fs_all)]
summary["claim1_PASS"] = (sum(d*d for d in dims_sorted.tolist()) == 120
                          and sorted(dims_sorted.tolist()) == expected_dims)
summary["claim2_branching_found"] = claim2_ok
summary["claim2_branch_name"] = branch_name
summary["claim2_mults_s"] = j32_mults_s
summary["claim2_mults_c"] = j32_mults_c
summary["claim3_num_faithful_3singlet_reps"] = len(three_singlet_faithful)
summary["claim3_unique"] = uniqueness_verdict
summary["claim3_candidates"] = [
    {"label": c["label"],
     "faithful": bool(c["faithful"]),
     "singlets_s": c["singlets_s"],
     "singlets_c": c["singlets_c"],
     "mults_s": c["mults_s"],
     "mults_c": c["mults_c"]}
    for c in candidates
]
summary["claim4_invariant_dims"] = results_claim4
summary["claim4_PASS"] = ok4

print(f"  Claim 1 (dims + Burnside):    {'PASS' if summary['claim1_PASS'] else 'FAIL'}")
print(f"  Claim 2 (j=3/2 -> 3*[1]+[5]): {'PASS' if claim2_ok else 'FAIL'}")
print(f"  Claim 3 (uniqueness):         {'UNIQUE' if uniqueness_verdict else 'NOT UNIQUE'}")
print(f"  Claim 4 (invariants):         {'PASS' if ok4 else 'FAIL'}")

with open("verify_2I_reps_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\n  Results written to verify_2I_reps_results.json")
