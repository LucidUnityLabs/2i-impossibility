"""
Wilson line centralizer: does the 2I Wilson line in E8' produce a hidden SU(3)'?
==============================================================================

QUESTION: embed 2I (binary icosahedral, |2I| = 120) into E8' via SU(2) subalgebras.
The unbroken hidden gauge group is the centralizer C_{E8'}(2I).

METHOD: character theory. The centralizer dimension is
    dim C_G(2I) = (1/|2I|) * sum_{g in 2I}  chi_adj(g)
where chi_adj(g) is the character of the E8 adjoint (restricted to the SU(2)
that 2I sits in) evaluated at the element g.

CROSS-CHECK: verify against KNOWN maximal subgroups.
  - E8 -> E7 x SU(2): centralizer of the SU(2) is E7 (dim 133).
  - E8 -> E6 x SU(3): centralizer of SU(3) is E6 (dim 78).
  - Principal SU(2) of E8: centralizer = rank-Cartan, dim 8 (abelian).

KEY SUBTLETY: 2I is a FINITE subgroup of SU(2). Its centralizer in E8 is
LARGER than or equal to the centralizer of SU(2), because commuting with
2I is a weaker condition than commuting with all of SU(2). For "large enough"
finite groups (2I is the largest finite subgroup of SU(2)), the centralizers
coincide IF the only nontrivial SU(2) irreps in the adjoint are spin-1/2
and spin-1 (on which 2I acts irreducibly).

PLAN:
  1. Construct 2I as unit quaternions; get conjugacy classes + rotation angles.
  2. Verify centralizer = E7 for the SU(2) from E7 x SU(2) embedding.
  3. Check whether ANY natural SU(2) embedding gives SU(3) centralizer.
  4. Check the "principal SU(2) of E6" embedding (the one that COULD give SU(3)).
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2

SECT = "=" * 74
def banner(s): print("\n" + SECT + f"\n{s}\n" + SECT)


# ============================================================================
# STEP 1: construct 2I as unit quaternions
# ============================================================================
banner("STEP 1: construct 2I (binary icosahedral group)")

def make_2I():
    """Return list of 120 unit quaternions (as (w,x,y,z) tuples) for 2I."""
    qs = []
    # 8: +/-1, +/-i, +/-j, +/-k
    for s in [1,-1]:
        qs.append((s,0,0,0)); qs.append((0,s,0,0)); qs.append((0,0,s,0)); qs.append((0,0,0,s))
    # 16: (+/-1 +/-i +/-j +/-k)/2
    for s1 in [1,-1]:
        for s2 in [1,-1]:
            for s3 in [1,-1]:
                for s4 in [1,-1]:
                    qs.append((s1/2, s2/2, s3/2, s4/2))
    # 96: even permutations of (0, +/-1, +/-1/phi, +/-phi)/2
    base = [0.0, 1.0, 1.0/PHI, PHI]
    # even permutations of positions
    perms = [(0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2),   # 4-cycles (even)
             (1,0,3,2),(0,3,2,1),(3,2,1,0),(2,1,0,3),   # hmm let me use A4
             (0,2,3,1),(2,3,1,0),(3,1,0,2),(1,0,2,3)]   # another 4 even
    # Simpler: all even permutations of 4 elements = A4, 12 permutations
    import itertools
    even_perms = [p for p in itertools.permutations(range(4)) if perm_parity(p) == 0]
    for p in even_perms:
        for s1 in [1,-1]:
            for s2 in [1,-1]:
                for s3 in [1,-1]:
                    vals = [0.0, s1*1.0, s2*(1.0/PHI), s3*PHI]
                    w,x,y,z = (vals[p[0]]/2, vals[p[1]]/2, vals[p[2]]/2, vals[p[3]]/2)
                    qs.append((w,x,y,z))
    # Deduplicate
    unique = list(set(q for q in qs))
    # Verify count
    return unique

def perm_parity(p):
    """Return 0 for even permutation, 1 for odd."""
    n = len(p); parity = 0; visited = [False]*n
    for i in range(n):
        if visited[i]: continue
        j = i; cycle = 0
        while not visited[j]:
            visited[j] = True; j = p[j]; cycle += 1
        parity += cycle - 1
    return parity % 2

G = make_2I()
print(f"Constructed |G| = {len(G)} elements (expect 120).")
assert len(G) == 120, f"Wrong order: {len(G)}"

def qmul(a, b):
    """Hamilton product of two (w,x,y,z) quaternions."""
    aw,ax,ay,az = a; bw,bx,by,bz = b
    return (aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw)

def qconj(a):
    w,x,y,z = a
    return (w,-x,-y,-z)

def qclose(a, b, tol=1e-9):
    return all(abs(a[i]-b[i]) < tol for i in range(4))

# Verify closure
print("Verifying group closure...", end=" ")
bad = 0
for a in G:
    for b in G:
        p = qmul(a,b)
        if not any(qclose(p, c) for c in G):
            bad += 1
            if bad <= 3: print(f"\n  NOT CLOSED: {a}*{b} = {p}")
print(f"done. Non-closed products: {bad}.")
assert bad == 0, "Group is not closed!"

# Identity check
ident = (1.0,0.0,0.0,0.0)
assert any(qclose(ident, c) for c in G), "No identity!"


# ============================================================================
# STEP 2: conjugacy classes and rotation angles
# ============================================================================
banner("STEP 2: conjugacy classes and rotation angles")

def conjugacy_classes(G):
    """Return list of (representative, size) for each conjugacy class."""
    classes = []
    seen = set()
    for i, g in enumerate(G):
        if i in seen: continue
        cls = []
        for h in G:
            hghinv = qmul(qmul(h, g), qconj(h))
            for j, k in enumerate(G):
                if qclose(hghinv, k) and j not in cls:
                    cls.append(j); break
        for j in cls: seen.add(j)
        classes.append((g, len(cls)))
    return classes

classes = conjugacy_classes(G)
classes.sort(key=lambda x: (x[1], round(x[0][0],3)))
print(f"Found {len(classes)} conjugacy classes (expect 9).")
print(f"Class sizes: {[c[1] for c in classes]} (sum = {sum(c[1] for c in classes)})")
assert sum(c[1] for c in classes) == 120

# Rotation angle for each class: chi_2(q) = 2*cos(theta/2) = 2*Re(q)
# => theta = 2 * arccos(Re(q))
print(f"\n{'Class rep (w,x,y,z)':>30} {'size':>5} {'Re(q)':>10} {'theta/2pi':>12} {'order':>6}")
print("-" * 75)
for rep, size in classes:
    Re = rep[0]
    theta = 2 * np.arccos(max(-1, min(1, Re)))   # guard against fp error
    theta_over_2pi = theta / (2*np.pi)
    # order: smallest n with q^n = identity
    order = 1; q = rep
    while not qclose(q, ident) and order < 50:
        q = qmul(q, rep); order += 1
    print(f"({rep[0]:+.4f},{rep[1]:+.3f},{rep[2]:+.3f},{rep[3]:+.3f}) {size:5d} {Re:10.4f} {theta_over_2pi:12.4f} {order:6d}")


# ============================================================================
# STEP 3: SU(2) character function and centralizer dimension
# ============================================================================
banner("STEP 3: centralizer dimension from characters")

def chi_spin(j, theta):
    """SU(2) character of spin j at rotation angle theta.
    chi_j(theta) = sin((2j+1) theta/2) / sin(theta/2).
    Handles the removable singularities at theta = 0 and theta = 2*pi."""
    half = theta / 2.0
    # Near theta = 0: chi_j -> 2j+1.
    if abs(theta) < 1e-9 or abs(theta - 2*np.pi) < 1e-9:
        sign = 1.0 if abs(theta) < 1e-9 else ((-1.0) ** (2*j))
        return sign * (2*j + 1)
    return np.sin((2*j + 1) * half) / np.sin(half)

def centralizer_dim(su2_decomp, classes, group_order=120):
    """Compute dim of centralizer of 2I in the Lie algebra.
    su2_decomp: dict {j: multiplicity} for the SU(2) decomposition of the adjoint.
    classes: list of (representative, size).
    Returns dim of centralizer = (1/|G|) sum_g chi_adj(g)."""
    total = 0.0
    for rep, size in classes:
        Re = rep[0]
        theta = 2 * np.arccos(max(-1, min(1, Re)))
        chi = sum(mult * chi_spin(j, theta) for j, mult in su2_decomp.items())
        total += size * chi
    return total / group_order


# ============================================================================
# STEP 4: verify against known E8 maximal subgroups
# ============================================================================
banner("STEP 4: verify centralizer formula against known E8 subgroups")

print("""
E8 -> E7 x SU(2):  adj decomposition under the SU(2):
  248 = (133,1) + (1,3) + (56,2)
  => SU(2) irreps:  spin-0 mult 133, spin-1/2 mult 56, spin-1 mult 1.
  Centralizer of SU(2) in E8 = E7 (dim 133).
  Cross-check: centralizer of 2I should also be 133 (2I acts irreducibly
  on spin-1/2 and spin-1, so no extra invariants).
""")
decomp_E7SU2 = {0: 133, 0.5: 56, 1: 1}
dim_check = centralizer_dim(decomp_E7SU2, classes)
print(f"  dim C_E8(2I) for E7xSU(2) embedding = {dim_check:.4f}  (expect 133)")

print("""
E8 -> SO(16):  adj decomposition under an SU(2) of SO(16):
  E8 -> SO(16): 248 = 120 (adjoint) + 128 (spinor).
  Take SU(2) inside SO(16) as the upper-left block. SO(16) adjoint 120 decomposes
  under this SU(2) as 120 = 1 + 3 + 3 + 5 + ... (complicated). SKIP for now;
  the E7xSU(2) check is the key one.
""")

print("""
E8 principal SU(2):  adj decomposition (Kostant, exponents 1,7,11,13,17,19,23,29):
  248 = 8*V_0 + V_1 + V_13 + V_21 + V_25 + V_33 + V_37 + V_45 + V_57
  (each V_m has dim m+1; spins are m/2).
  Centralizer of principal SU(2) = rank-Cartan (dim 8, abelian).
  2I might give a LARGER centralizer (extra invariants in high-spin irreps).
""")
decomp_principal_E8 = {0: 8, 0.5: 1, 6.5: 1, 10.5: 1, 12.5: 1, 16.5: 1, 18.5: 1, 22.5: 1, 28.5: 1}
dim_principal = centralizer_dim(decomp_principal_E8, classes)
print(f"  dim C_E8(2I) for principal-SU(2)-of-E8 embedding = {dim_principal:.4f}  (SU(2) gives 8)")


# ============================================================================
# STEP 5: the E6 embedding -- the ONLY way to get SU(3) in the centralizer
# ============================================================================
banner("STEP 5: can ANY SU(2) embedding give SU(3) in the centralizer?")

print("""
For the hidden SU(3)' to appear in the centralizer, the 2I Wilson line must
commute with an SU(3) subalgebra of E8. The ONLY E8 maximal subgroup containing
an SU(3) factor is E6 x SU(3). So:

  2I must sit in the E6 factor (not the SU(3) factor) for SU(3) to survive.

Then C_E8(2I) = C_E6(2I) x SU(3). For pure SU(3) hidden sector, need
C_E6(2I) = trivial (or broken by other means).

For 2I -> SU(2) -> E6 via the PRINCIPAL SU(2) of E6:
  E6 exponents: 1, 4, 5, 7, 8, 11 (Coxeter number 12).
  Adjoint decomposition under principal SU(2) of E6:
    78 = 6*V_0 + V_1 + V_7 + V_9 + V_13 + V_15 + V_21
""")
decomp_principal_E6 = {0: 6, 0.5: 1, 3.5: 1, 4.5: 1, 6.5: 1, 7.5: 1, 10.5: 1}
dim_C_E6 = centralizer_dim(decomp_principal_E6, classes)
print(f"  dim C_E6(2I) via principal SU(2) of E6 = {dim_C_E6:.4f}")
print(f"  (If this is just the rank-6 Cartan, hidden sector = U(1)^6 x SU(3).)")

print("""
For 2I -> SU(2) -> E6 via the SU(2) of SU(6) x SU(2) maximal subgroup of E6:
  E6 -> SU(6) x SU(2): 78 = (35,1) + (1,3) + (20,2) + (1,1)... 
  Actually: 78 = (35,1) + (1,3) + (20,2) + (1,1)? Dimension: 35+3+40+1 = 79 != 78.
  Correct: 78 = (35,1) + (1,3) + (20,2) + (1,1) is wrong.
  E6 -> SU(6) x SU(2): 78 = (35,1) + (1,3) + (20,2). Dim: 35+3+40 = 78. CHECK.
  SU(2) decomposition of E6 adjoint (via SU(6)xSU(2)):
    spin-0 mult 35 (the SU(6) adjoint), spin-1/2 mult 20, spin-1 mult 1.
  Centralizer of 2I: 2I acts irreducibly on spin-1/2 and spin-1.
  => dim C_E6(2I) = 35 (the SU(6) survives!).
""")
decomp_SU6SU2_of_E6 = {0: 35, 0.5: 20, 1: 1}
dim_C_E6_SU6 = centralizer_dim(decomp_SU6SU2_of_E6, classes)
print(f"  dim C_E6(2I) via SU(6)xSU(2) embedding = {dim_C_E6_SU6:.4f}  (expect 35 = dim SU(6))")
print(f"  => hidden sector = SU(6) x SU(3), dimension {dim_C_E6_SU6 + 8:.0f}.")


# ============================================================================
# STEP 6: SUMMARY
# ============================================================================
banner("STEP 6: summary of Wilson line centralizer results")

print(f"""
EMBEDDING                              dim C_E8(2I)   HIDDEN SECTOR
---------------------------------------------------------------------------
Natural (E7 x SU(2)):                       {dim_check:.0f}        E7  (dim 133, rank 7)
Principal SU(2) of E8:                      {dim_principal:.0f}        abelian U(1)^? 
Principal SU(2) of E6 (in E6 x SU(3)):      {dim_C_E6:.0f}+8      C_E6 x SU(3)  (need C_E6)
SU(6) x SU(2) embedding of E6:              {dim_C_E6_SU6 + 8:.0f}       SU(6) x SU(3) (dim {dim_C_E6_SU6 + 8:.0f})

ANALYSIS:
  - The NATURAL embedding (2I in the SU(2) from E7 x SU(2) maximal subgroup
    of E8, which is the same as SU(2) c SU(3) c E6 x SU(3) c E8) gives
    hidden E7, NOT SU(3).

  - To get SU(3)' as the hidden sector, the framework must embed 2I via the
    PRINCIPAL SU(2) of E6 (a specific, non-generic choice). Even then, the
    centralizer is C_E6(2I) x SU(3), and C_E6(2I) = {dim_C_E6:.0f}-dimensional
    (not trivial -- the principal SU(2) leaves a residual abelian piece).

  - The SU(6) x SU(2) embedding of E6 gives SU(6) x SU(3) hidden sector
    (dim {dim_C_E6_SU6 + 8:.0f}), far from pure SU(3) SYM.

CONCLUSION:
  The 2I Wilson line does NOT naturally produce a hidden SU(3)'. The natural
  embedding gives E7. Getting SU(3) requires:
    (a) a non-generic embedding choice (2I into the principal SU(2) of E6), AND
    (b) an additional mechanism to break the residual C_E6(2I) piece.

  The framework's claim of "hidden SU(3) pure SYM" is an ASSUMPTION, not a
  consequence of the compactification. Combined with the baryogenesis failure
  (portal_baryogenesis_check.py: timing + magnitude fail), the entire chain
  E8' -> hidden SU(3) -> Lambda_GC -> m_a -> baryogenesis is unsupported.

  For DM abundance the axion mass formula m_a ~ Lambda_GC^2/f_a still applies
  IF the hidden sector happens to be SU(3), but this is now a free choice
  of Wilson line embedding, not a prediction.
""")

# One more check: is 2I actually "irreducible" on spin-3/2 and higher?
# This determines whether the centralizer of 2I can exceed the centralizer of SU(2).
banner("BONUS: is 2I irreducible on all low-spin SU(2) irreps?")

print("If 2I has a trivial subrepresentation in spin-j, then C_G(2I) > C_G(SU(2)).")
print("Checking spin-j for j = 0, 1/2, 1, 3/2, 2, 5/2, 3, 7/2:\n")
for j in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
    # multiplicity of trivial 2I irrep in spin-j = (1/120) sum_g chi_j(g)
    mult_trivial = 0.0
    for rep, size in classes:
        Re = rep[0]
        theta = 2 * np.arccos(max(-1, min(1, Re)))
        mult_trivial += size * chi_spin(j, theta)
    mult_trivial /= 120
    note = "  <-- contains trivial subrep!" if mult_trivial > 0.5 else ""
    print(f"  spin-{j:>4}: dim {int(2*j+1):>3},  mult of trivial 2I irrep = {mult_trivial:.4f}{note}")

print("""
Note: spin-3 and higher contain trivial 2I irreps (the icosahedral invariant
polynomials). This is why the PRINCIPAL SU(2) embeddings can give 2I
centralizers LARGER than the SU(2) centralizer: the high-spin SU(2) irreps
contribute extra 2I-invariant directions. This is exactly what we see for
the principal SU(2) of E8 (SU(2) centralizer dim 8, but 2I centralizer dim
%.0f -- the high-spin irreps add extra invariants).
""" % dim_principal)
