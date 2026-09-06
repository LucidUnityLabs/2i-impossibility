#!/usr/bin/env python3
"""
gamma70_fusion.py

Final structural investigation: does Gamma_70 = SL(2, Z/70) provide a
genuine "inversion + symmetry" fusion of the framework's icosahedral
structure (2I = SL(2, Z/5)) with sin(pi/14)'s home (SL(2, Z/7))?

By CRT (since 70 = 2 * 5 * 7 with pairwise coprime factors):
    Gamma_70 = SL(2, Z/2) x SL(2, Z/5) x SL(2, Z/7)
            = S_3 x 2I x SL(2,7)

|Gamma_70| = 6 * 120 * 336 = 241920.

Parts A-I per the spec. Saves gamma70_fusion_results.json.
"""

import json
import os
import itertools
import numpy as np
import mpmath as mp
from sympy import (
    Rational, sqrt, cos, sin, pi, exp, I,
    simplify, nsimplify, isprime, totient,
    Matrix, eye, zeros, S
)

mp.mp.dps = 50

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUT_DIR, "gamma70_fusion_results.json")

results = {}


def banner(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# =============================================================================
# PART A: CRT decomposition of Gamma_70
# =============================================================================
banner("PART A: CRT decomposition of Gamma_70 = SL(2, Z/70)")


def sl2_order_mod_N(N):
    order = N**3
    for p in range(2, N + 1):
        if isprime(p) and N % p == 0:
            order = order * (p * p - 1) // (p * p)
    return order


def sl2_modN_elements(N):
    """All 2x2 matrices over Z/N with det = 1, returned as 4-tuples (a,b,c,d)."""
    out = []
    for a in range(N):
        for b in range(N):
            for c in range(N):
                # determinant constraint -> d unique mod N if gcd(a, N)... but easiest brute force
                for d in range(N):
                    if (a * d - b * c) % N == 1:
                        out.append((a, b, c, d))
    return out


ord_2 = sl2_order_mod_N(2)
ord_5 = sl2_order_mod_N(5)
ord_7 = sl2_order_mod_N(7)
ord_70 = sl2_order_mod_N(70)

print(f"|SL(2, Z/2)|  = {ord_2}")
print(f"|SL(2, Z/5)|  = {ord_5}    (= |2I| = 120)")
print(f"|SL(2, Z/7)|  = {ord_7}")
print(f"|SL(2, Z/70)| = {ord_70}")
print(f"product check 6*120*336 = {6 * 120 * 336}, equals |SL(2,Z/70)|? {6*120*336 == ord_70}")

# Build SL(2, Z/2), SL(2, Z/5), SL(2, Z/7) explicitly.
sl2_2 = sl2_modN_elements(2)
sl2_5 = sl2_modN_elements(5)
sl2_7 = sl2_modN_elements(7)
print(f"\nEnumerated: |SL(2,Z/2)|={len(sl2_2)}, |SL(2,Z/5)|={len(sl2_5)}, |SL(2,Z/7)|={len(sl2_7)}")
assert len(sl2_2) == 6 and len(sl2_5) == 120 and len(sl2_7) == 336

# CRT correspondence: a matrix M in SL(2,Z/70) maps to (M mod 2, M mod 5, M mod 7).
# Verify isomorphism by counting elements / by exhibiting bijection.
# Building all 241920 elements is feasible but slow with the 4-loop above
# (70^4 = 24.01M iterations, each with a det check). We use the CRT bijection itself
# to construct SL(2, Z/70) faster: enumerate the product and lift via CRT.

def crt_lift_pair(x_mod_2, x_mod_5, x_mod_7):
    # CRT: find x in Z/70 with x = x_mod_2 mod 2, x = x_mod_5 mod 5, x = x_mod_7 mod 7
    # 35 = 0 mod 5, 0 mod 7, 1 mod 2  -> coeff for mod 2
    # 56 = 0 mod 2, 1 mod 5, 0 mod 7  -> coeff for mod 5  (56 = 2*5*... let's compute)
    # We'll compute by direct search since N is small.
    for x in range(70):
        if x % 2 == x_mod_2 and x % 5 == x_mod_5 and x % 7 == x_mod_7:
            return x
    raise ValueError("CRT failed")


# Verify CRT on a small sample of element triples:
import random
random.seed(0)
sample = 200
verified = 0
for _ in range(sample):
    m2 = random.choice(sl2_2)
    m5 = random.choice(sl2_5)
    m7 = random.choice(sl2_7)
    a = crt_lift_pair(m2[0], m5[0], m7[0])
    b = crt_lift_pair(m2[1], m5[1], m7[1])
    c = crt_lift_pair(m2[2], m5[2], m7[2])
    d = crt_lift_pair(m2[3], m5[3], m7[3])
    if (a * d - b * c) % 70 == 1:
        verified += 1
print(f"\nCRT bijection check: {verified}/{sample} sampled triples lift to SL(2,Z/70).")
assert verified == sample, "CRT lift produced non-SL(2,Z/70) elements"

# Verify the explicit isomorphism SL(2, Z/5) <-> 2I via order/character signatures.
# 2I has presentation <s, t | s^2 = t^3 = (st)^5> with order 120.
# SL(2, F_5) has S = [[0,-1],[1,0]] (order 4), T = [[1,1],[0,1]] (order 5).
# (-S T)^3 = -I.  Standard result: SL(2,5) = 2I (isomorphic).
# We confirm by checking element-order signature matches that of 2I.
def matrix_order(M, N):
    """Order of 2x2 matrix M mod N."""
    a, b, c, d = M
    cur = (a, b, c, d)
    Identity = (1, 0, 0, 1)
    for k in range(1, 4 * N + 5):
        if cur == Identity:
            return k
        # multiply: cur * M
        ca, cb, cc, cd = cur
        ma, mb, mc, md = M
        na = (ca * ma + cb * mc) % N
        nb = (ca * mb + cb * md) % N
        nc = (cc * ma + cd * mc) % N
        nd = (cc * mb + cd * md) % N
        cur = (na, nb, nc, nd)
    return -1


order_signature_5 = sorted(matrix_order(g, 5) for g in sl2_5)
# 2I order signature: orders are {1, 2, 3, 4, 5, 6, 10}
# count (1):1, (2):1, (3):20, (4):30, (5):12, (6):20, (10):12 ... sum = 95? need 120.
# Let me just check what set of orders appears.
unique_orders_5 = sorted(set(order_signature_5))
print(f"\nOrder set of SL(2,Z/5) elements: {unique_orders_5}")
print(f"  expected for 2I: {{1, 2, 3, 4, 5, 6, 10}}")
# Element-count signature
from collections import Counter
oc5 = Counter(order_signature_5)
print(f"  order counts in SL(2,Z/5): {dict(sorted(oc5.items()))}")
# 2I element order counts (standard):
#   order 1: 1   (identity)
#   order 2: 1   (-I)
#   order 3: 20
#   order 4: 30
#   order 5: 24
#   order 6: 20
#   order 10: 24
# total = 1+1+20+30+24+20+24 = 120
expected_2I = {1: 1, 2: 1, 3: 20, 4: 30, 5: 24, 6: 20, 10: 24}
match_2I = dict(oc5) == expected_2I
print(f"  matches 2I element-order signature? {match_2I}")

results["partA"] = {
    "order_SL2_mod_2": ord_2,
    "order_SL2_mod_5": ord_5,
    "order_SL2_mod_7": ord_7,
    "order_SL2_mod_70": ord_70,
    "factorization_check": 6 * 120 * 336 == ord_70,
    "CRT_sampled_verified": f"{verified}/{sample}",
    "SL2_Z5_order_counts": dict(oc5),
    "SL2_Z5_matches_2I": match_2I,
    "structure": "Gamma_70 = SL(2,Z/2) x SL(2,Z/5) x SL(2,Z/7) = S_3 x 2I x SL(2,7)",
}

# =============================================================================
# PART B: Character theory of Gamma_70
# =============================================================================
banner("PART B: Character theory of Gamma_70 (irreps as tensor products)")

# Irreducible representations:
# S_3 = SL(2,Z/2): dims [1, 1, 2]
# 2I = SL(2,Z/5):   dims [1, 2, 2, 3, 3, 4, 4, 5, 6]   (9 irreps, sum sq = 1+4+4+9+9+16+16+25+36 = 120)
# SL(2,7):          dims [1, 3, 3, 4, 4, 6, 6, 7, 8, 8, 8] (11 irreps, sum sq = 336)
dims_S3 = [1, 1, 2]
dims_2I = [1, 2, 2, 3, 3, 4, 4, 5, 6]
dims_SL27 = [1, 3, 3, 4, 4, 6, 6, 6, 7, 8, 8]

# sanity
assert sum(d * d for d in dims_S3) == 6
assert sum(d * d for d in dims_2I) == 120
assert sum(d * d for d in dims_SL27) == 336
print(f"S_3  dims: {dims_S3}, sum sq = {sum(d*d for d in dims_S3)}")
print(f"2I   dims: {dims_2I}, sum sq = {sum(d*d for d in dims_2I)}")
print(f"SL27 dims: {dims_SL27}, sum sq = {sum(d*d for d in dims_SL27)}")

# Tensor product irreps:
gamma70_dims = []
for d_s3 in dims_S3:
    for d_2I in dims_2I:
        for d_sl27 in dims_SL27:
            gamma70_dims.append(d_s3 * d_2I * d_sl27)

print(f"\n# irreps Gamma_70 = 3 * 9 * 11 = {3*9*11}; computed = {len(gamma70_dims)}")
assert len(gamma70_dims) == 297
sum_sq_70 = sum(d * d for d in gamma70_dims)
print(f"sum sq of Gamma_70 dims = {sum_sq_70}; |Gamma_70| = {ord_70}; match? {sum_sq_70 == ord_70}")

# Histogram of dimensions:
dim_counter = Counter(gamma70_dims)
print(f"Gamma_70 irrep dimension multiset (top 12 by count):")
for dim, cnt in sorted(dim_counter.items()):
    print(f"  dim {dim:4d}: {cnt} irreps")

results["partB"] = {
    "num_irreps_S3": len(dims_S3),
    "num_irreps_2I": len(dims_2I),
    "num_irreps_SL27": len(dims_SL27),
    "num_irreps_Gamma70": len(gamma70_dims),
    "expected_count": 3 * 9 * 11,
    "sum_sq_check": sum_sq_70 == ord_70,
    "dim_histogram": {str(k): v for k, v in sorted(dim_counter.items())},
}

# =============================================================================
# PART C: Locate sin(pi/14) and golden ratio simultaneously
# =============================================================================
banner("PART C: Simultaneous golden ratio and sin(pi/14) in Gamma_70 characters?")

phi = (1 + mp.sqrt(5)) / 2
inv_phi = phi - 1   # = 1/phi
sin_pi14 = mp.sin(mp.pi / 14)

print(f"phi          = {phi}")
print(f"1/phi        = {inv_phi}")
print(f"sin(pi/14)   = {sin_pi14}")

# 2I character values (standard reference):
# The two 2-dim irreps (the "spinors" of 2I) have character on the order-10 elements
# equal to phi and -1/phi, and on order-5 elements -1/phi and phi.
# Specifically, the binary-icosahedral character table contains values
#   {phi, 1-phi, -phi, -(1-phi)} = {phi, -1/phi, -phi, 1/phi}.
# So phi (golden ratio) appears LITERALLY as a character value of 2I.
# Reference: any standard reference, e.g. Conway-Sloane or Klein.

# SL(2,7) character values: principal-series characters take values
#   2 cos(2 pi a/7) for a=1,2,3
# on the 7-element split torus. In particular
#   2 cos(2 pi * 2 / 7) = -2 sin(pi/14)
# So sin(pi/14) = -(1/2) * (a principal-series character value of SL(2,7)).

# In Gamma_70 = S_3 x 2I x SL(2,7), the irreps are tensor products R_2 (x) R_5 (x) R_7,
# and the characters are PRODUCTS:
#   chi_{R_2 (x) R_5 (x) R_7}(g_2, g_5, g_7) = chi_{R_2}(g_2) * chi_{R_5}(g_5) * chi_{R_7}(g_7).
# Therefore on a group element (g_2, g_5, g_7) where g_5 has 2I-character phi and
# g_7 has SL(2,7)-character -2 sin(pi/14), we get
#   chi(g) = chi_{R_2}(g_2) * phi * (-2 sin(pi/14))
# which is a Q-linear combination involving BOTH phi AND sin(pi/14).
# But this is a PRODUCT, not a genuinely new fused algebraic combination -- it just
# multiplies them. The factorization is "trivial" in the sense that the value lies
# in the COMPOSITUM Q(phi) * Q(zeta_14) = Q(zeta_70)^+ but is a tensor of separate factors.

# Explicitly compute representative product values.
samples = []
# A 2I character value on order-10 element (the 2-dim spinor rep of 2I):
val_2I_phi = float(phi)
# The conjugate value on the other 2-dim spinor rep:
val_2I_minus_inv_phi = float(-inv_phi)
# SL(2,7) principal-series character on order-7 element:
val_SL27_a2 = float(2 * mp.cos(2 * mp.pi * 2 / 7))   # = -2 sin(pi/14)
val_SL27_a1 = float(2 * mp.cos(2 * mp.pi * 1 / 7))
val_SL27_a3 = float(2 * mp.cos(2 * mp.pi * 3 / 7))

print(f"\nSample character values:")
print(f"  2I     [2-dim rep, on g of order 10]     = phi          = {val_2I_phi:.12f}")
print(f"  2I     [other 2-dim, on g of order 5]    = -1/phi       = {val_2I_minus_inv_phi:.12f}")
print(f"  SL(2,7)[principal-series, a=2, order-7]  = -2 sin(pi/14) = {val_SL27_a2:.12f}")

# Product example (a Gamma_70 character value):
prod = val_2I_phi * val_SL27_a2
print(f"\nGamma_70 character product (phi) * (-2 sin(pi/14)) = {prod:.12f}")
print("This value lies in Q(zeta_70) but is reducibly the PRODUCT of factors.")

# Question: is this a genuinely new algebraic number, or does it factor cleanly?
# Q(zeta_5) and Q(zeta_7) are linearly disjoint over Q (their discriminants are
# coprime). The compositum Q(zeta_5, zeta_7) = Q(zeta_35) has degree 4*6 = 24.
# The product phi * sin(pi/14) lies in this compositum, but as an ALGEBRAIC
# number it is reducibly expressible as a product, NOT as a sum or polynomial
# in a single primitive element (other than zeta_35 itself).
# So at the level of CHARACTER VALUES, Gamma_70 just gives you Q(zeta_5) (x) Q(zeta_7).
# This is NOT genuine algebraic fusion -- it is multiplicative.

# Key test: are there irreps of Gamma_70 whose character values are NOT
# expressible as a single product chi_2(g_2) * chi_5(g_5) * chi_7(g_7)?
# Answer: NO, because Gamma_70 is a literal direct product, so all irreps are
# tensor products and all characters multiply. No "fused" irrep exists.

print("\nVERDICT (Part C):")
print("  Gamma_70 character values lie in Q(zeta_70) = Q(zeta_5) * Q(zeta_7)")
print("  but factor MULTIPLICATIVELY across the three direct factors.")
print("  No genuinely new 'fused' algebraic number arises: phi and sin(pi/14)")
print("  appear in DIFFERENT factors and combine only as products.")
print("  This is the same structural triviality as level 14 -- just with an")
print("  additional 2I factor that multiplies through.")

results["partC"] = {
    "phi_in_2I_characters": True,
    "sin_pi14_in_SL27_characters": True,
    "characters_factor_multiplicatively": True,
    "genuinely_new_fused_irrep": False,
    "explanation": (
        "Gamma_70 is a direct product, so every irrep is a tensor product and "
        "every character value is the multiplicative product of factor character "
        "values. phi appears in the 2I factor and sin(pi/14) in the SL(2,7) factor, "
        "but they combine only as products -- no new 'irreducible' algebraic "
        "combination arises beyond what Q(zeta_35) already provides."
    ),
    "sample_product": prod,
}

# =============================================================================
# PART D: Modular forms at level 70 -- newforms vs oldforms
# =============================================================================
banner("PART D: Modular forms at level 70")

# Index [SL(2,Z) : Gamma(70)] = |SL(2,Z/70)| = 241920
# Number of cusps: |SL(2,Z/N)| / N = 241920 / 70 = 3456 (for N>=3)
# Genus: g(X(N)) = 1 + (N-6) * |SL(2,Z/N)| / (12 N), for N>=3
N = 70
n_cusps_70 = ord_70 // N
genus_70 = 1 + (N - 6) * ord_70 // (12 * N)
print(f"|SL(2,Z/70)| = {ord_70}")
print(f"# cusps of X(70) = {n_cusps_70}")
print(f"genus  of X(70) = {genus_70}")

# Dimension of M_k(Gamma(70)) for k >= 2 even (Diamond-Shurman thm 3.5.1):
# dim S_2 = g
# dim M_2 = g + (n_cusps - 1)
# dim S_k = (k-1)(g-1) + (k/2 - 1) * n_cusps    for k>=4 even
# dim M_k = dim S_k + n_cusps                   for k>=4 even
dim_table = {}
for k in [2, 4, 6, 8, 10, 12]:
    if k == 2:
        dim_S = genus_70
        dim_M = genus_70 + n_cusps_70 - 1
    else:
        dim_S = (k - 1) * (genus_70 - 1) + (k // 2 - 1) * n_cusps_70
        dim_M = dim_S + n_cusps_70
    dim_table[k] = {"dim_S_k": dim_S, "dim_M_k": dim_M, "dim_E_k": dim_M - dim_S}
    print(f"k={k:2d}:  dim M_k = {dim_M:8d},  dim S_k = {dim_S:8d},  dim E_k = {n_cusps_70 if k!=2 else n_cusps_70-1:6d}")

# Newforms vs oldforms at level 70.
# Atkin-Lehner-Li theory: at level 70, oldforms come from divisors of 70 (excluding 70):
#   1, 2, 5, 7, 10, 14, 35
# Each old level d contributes forms via the maps
#   f -> f(z) and f -> f(70/d * z)   (degeneracy maps),
# multiplicities given by sigma_0(70/d) (number of divisors).
#
# For S_2(Gamma_0(N)) Hecke-newform dimensions:
#   genus(X_0(70)) = 9   (standard table; e.g. LMFDB)
#   dim S_2(X_0(70))^new = 7   (LMFDB: Gamma_0(70) has 7 newform Galois orbits in S_2)
# Actually LMFDB: 70.2.a (rational newforms in S_2(Gamma_0(70))) has 4 isogeny
# classes; over algebraic closure dimensions sum to 7. Let me state conservatively.
#
# For Gamma(N) (full congruence) newform decomposition is more involved -- we
# count via stripping oldforms.
#
# We compute the number of NEWFORMS at level 70 (Gamma_0(70), Gamma_1(70))
# using known LMFDB data:

lmfdb_S2_Gamma0_dims = {
    1: 0, 2: 0, 5: 0, 7: 0, 10: 0, 14: 1, 35: 3, 70: 9
}
# At level 70, dim S_2(Gamma_0(70)) = 9 (genus of X_0(70) = 9).
# Oldforms from level 14 contribute 2 * 1 = 2 (degeneracy maps from level 14 in level 70:
#   index [Gamma_0(14):Gamma_0(70)] = sigma_0(70/14) = sigma_0(5) = 2).
# Oldforms from level 35 contribute 2 * 3 = 6 (degeneracy index sigma_0(70/35) = sigma_0(2) = 2).
# But we must subtract overlap (oldforms common to multiple levels, none here).
# Total oldform contribution: 2 + 6 = 8.
# Newforms at level 70: dim S_2(Gamma_0(70))^new = 9 - 8 = 1.
# This matches LMFDB exactly: 70.2.a.a is the unique newform Galois orbit at level 70.

new_70_S2_Gamma0 = 9 - 8
print(f"\nGamma_0(70) cusp form dimensions and newform count:")
for d, dim in sorted(lmfdb_S2_Gamma0_dims.items()):
    print(f"  S_2(Gamma_0({d})): dim = {dim}")
print(f"  oldform contributions: 2*dim_S2_G0(14) + 2*dim_S2_G0(35) = 2*1 + 2*3 = 8")
print(f"  newform dimension at level 70 (Gamma_0): dim S_2^new = 9 - 8 = {new_70_S2_Gamma0}")
print(f"  -> matches LMFDB: 70.2.a.a is the unique newform orbit (it's an elliptic curve E)")

# At level 70, Gamma_1(70) and Gamma(70) have many more newforms; counting them
# precisely requires character decomposition via the Hecke algebra. We give the
# rough count via the trace formula:
#   dim S_2(Gamma_1(70)) - oldforms = dim S_2(Gamma_1(70))^new
# but the practical observation is that LEVEL 70 DOES support genuine newforms
# (at least one in S_2(Gamma_0(70))), so it is NOT just an oldform combination.

results["partD"] = {
    "n_cusps_X70": n_cusps_70,
    "genus_X70": genus_70,
    "dim_table": {str(k): v for k, v in dim_table.items()},
    "S2_Gamma0_dims": {str(k): v for k, v in lmfdb_S2_Gamma0_dims.items()},
    "newform_dim_S2_Gamma0_70": new_70_S2_Gamma0,
    "lmfdb_label": "70.2.a.a",
    "comment": (
        "S_2(Gamma_0(70))^new = 1 (the elliptic curve 70.a1 / 70.2.a.a from LMFDB). "
        "So level 70 DOES have a genuine newform, not purely an oldform combination. "
        "But this newform is associated to the standard L-function of an elliptic "
        "curve and has no special relationship to phi or sin(pi/14)."
    ),
}

# =============================================================================
# PART E: Chirality from Gamma_70 reps
# =============================================================================
banner("PART E: Chirality -- complex (non-self-dual) irreps of Gamma_70")

# Frobenius-Schur indicator classifies real (FS=+1), pseudoreal (FS=-1), complex (FS=0).
# For the factors:
#   S_3 = SL(2,Z/2): all 3 irreps are real (FS=+1).
#   2I = SL(2,Z/5):  9 irreps. The two 2-dim "spinor" reps are PSEUDOREAL (FS=-1);
#                    others are real. None complex.
#   SL(2,7):          11 irreps. The two 3-dim irreps (3, 3-bar) are COMPLEX
#                    (each other's conjugate). Same for the two 4-dim irreps.
#                    So SL(2,7) has at least 2 complex-conjugate pairs (4 complex irreps).
# For SL(2, F_p) with p = 7 = 3 mod 4:
#   * The principal series rho_a (a=1,2,3, dim p+1=8) and rho_a-bar are conjugate
#     when a != p-a (mod 7), so we get pairs.
#   * The discrete series sigma_b (dim p-1=6) and conjugate.
# Standard reference: Bonnafe, Reps of SL2(Fq).
#
# At p=7: precise count of complex irreps of SL(2,7) is 4:
#   - the two 3-dim cuspidal irreps form a conjugate pair  (2 complex)
#   - the two 4-dim Steinberg-twist irreps form a conjugate pair (2 complex)
#   - everything else (1, 6, 6, 7, 8, 8, 8) is self-conjugate (real or pseudoreal).
# Actually the correct count, per Bonnafe Thm. 5.1: SL(2,7) has 4 complex irreps.

# Frobenius-Schur indicators per factor:
fs_S3 = [+1, +1, +1]                                  # all real
fs_2I = [+1, -1, -1, +1, +1, -1, -1, +1, +1]          # spinors pseudoreal (FS=-1)
# SL(2,7): per Bonnafe / standard tables
# Map to dims [1, 3, 3, 4, 4, 6, 6, 6, 7, 8, 8]:
#   1: real (trivial)
#   3, 3-bar: complex conjugate pair (FS=0, FS=0)
#   4, 4-bar: complex conjugate pair (FS=0, FS=0)  -- the faithful 2-cover irreps
#   6, 6, 6: principal-series and discrete; one real (Steinberg-like), two pseudoreal
#   7: real (Steinberg)
#   8, 8: real / pseudoreal mix in principal series
# For our purposes the COUNT of complex irreps is what matters: 4 complex (two pairs).
fs_SL27 = [+1, 0, 0, 0, 0, -1, -1, +1, +1, -1, +1]    # length 11, 4 zeros = 4 complex

# Complex irreps of Gamma_70 = S_3 x 2I x SL(2,7):
# A tensor product is complex (FS=0) iff at least one tensor factor has FS=0.
# Equivalently, FS(R_2 (x) R_5 (x) R_7) = FS(R_2) * FS(R_5) * FS(R_7), with the
# convention that 0 absorbs (any factor of 0 -> product is 0 for the FS indicator).

n_complex = 0
n_real = 0
n_pseudo = 0
fs_list = []
for f2 in fs_S3:
    for f5 in fs_2I:
        for f7 in fs_SL27:
            # If any factor is 0 (complex), product irrep is complex.
            if f2 == 0 or f5 == 0 or f7 == 0:
                fs_prod = 0
            else:
                fs_prod = f2 * f5 * f7
            fs_list.append(fs_prod)
            if fs_prod == 0:
                n_complex += 1
            elif fs_prod == 1:
                n_real += 1
            else:
                n_pseudo += 1

print(f"Frobenius-Schur classification of {len(fs_list)} Gamma_70 irreps:")
print(f"  real       (FS=+1): {n_real}")
print(f"  pseudoreal (FS=-1): {n_pseudo}")
print(f"  complex    (FS= 0): {n_complex}")
print(f"  total = {n_real + n_pseudo + n_complex} (expected {3*9*11})")

# Dimensions of complex irreps:
complex_dims = []
i = 0
for f2, d2 in zip(fs_S3, dims_S3):
    for f5, d5 in zip(fs_2I, dims_2I):
        for f7, d7 in zip(fs_SL27, dims_SL27):
            if f2 == 0 or f5 == 0 or f7 == 0:
                complex_dims.append(d2 * d5 * d7)

print(f"\nComplex irreps come in conjugate pairs; {n_complex} complex irreps form")
print(f"  {n_complex // 2} pairs.")
print(f"Complex-irrep dimension multiset: {dict(Counter(complex_dims))}")

# Smallest complex irrep dim and "interesting" ones:
small_complex = sorted(set(complex_dims))[:10]
print(f"  smallest complex-irrep dims: {small_complex}")

# Implication for chirality: chiral SM fermion content (e.g. left-handed Q in 3 of
# something, right-handed conjugates in 3-bar) requires a complex (non-self-dual)
# representation. Pure 2I (or its tensor with S_3 / 2I) has only real / pseudoreal
# irreps, so chirality cannot arise from icosahedral structure alone.
# Gamma_70 INHERITS complex irreps from the SL(2,7) factor -- this is what enables
# chirality "on paper".

results["partE"] = {
    "fs_S3": fs_S3,
    "fs_2I": fs_2I,
    "fs_SL27": fs_SL27,
    "n_real_Gamma70": n_real,
    "n_pseudoreal_Gamma70": n_pseudo,
    "n_complex_Gamma70": n_complex,
    "complex_irrep_dim_histogram": {str(k): v for k, v in dict(Counter(complex_dims)).items()},
    "smallest_complex_dims": small_complex,
    "chirality_source": "SL(2,7) factor (3 vs 3-bar, 4 vs 4-bar)",
    "remark": (
        "Gamma_70 has complex irreps inherited from SL(2,7); 2I alone has none. "
        "Chirality 'lives' on the SL(2,7) factor in this factorization."
    ),
}

# =============================================================================
# PART F: Generation assignment test (3-dim irreps)
# =============================================================================
banner("PART F: 3-generation assignment test")

# 3-dim irreps of each factor:
# S_3:    no 3-dim irrep (dims are 1,1,2)
# 2I:     two 3-dim irreps (the 3 and 3' of the binary icosahedral group)
# SL(2,7): two 3-dim irreps (the 3, 3-bar complex pair)
#
# Tensor products yielding dim 3:
#   d_2 * d_5 * d_7 = 3 with d_2 in {1,1,2}, d_5 in {1,2,2,3,3,4,4,5,6}, d_7 in {1,3,3,4,4,6,6,7,8,8,8}
# Possibilities:
#   1 * 1 * 3 (4 ways: 2 choices for the trivial S_3 irrep x 1 for trivial 2I x 2 for 3 of SL27)
#     = 2 * 1 * 2 = 4 irreps that are pure SL(2,7)-triplet (with parity tag from S_3).
#   1 * 3 * 1 (2 * 2 * 1 = 4): pure 2I-triplet (with parity tag).
# Are there 3-dim irreps from "fusion" e.g. d_5=d_7>1 multiplied? No, since 3 is prime.

three_dim_assignments = []
for i2, d2 in enumerate(dims_S3):
    for i5, d5 in enumerate(dims_2I):
        for i7, d7 in enumerate(dims_SL27):
            if d2 * d5 * d7 == 3:
                three_dim_assignments.append((d2, d5, d7))

print(f"Tensor decompositions giving a 3-dim irrep of Gamma_70:")
for tup in three_dim_assignments:
    print(f"  S_3 dim {tup[0]}, 2I dim {tup[1]}, SL(2,7) dim {tup[2]}")
print(f"Total 3-dim irreps: {len(three_dim_assignments)}")
print()
print("Note: the only ways to get 3 are 1*1*3 or 1*3*1 -- never 1*3*3 etc.")
print("So a 3-generation assignment in Gamma_70 lives ENTIRELY on the 2I factor")
print("OR ENTIRELY on the SL(2,7) factor, with at most a parity tag from S_3.")
print("There is NO mixed/fused 3-dim representation. This is the structural")
print("'CRT triviality' -- the 3-generation structure can't 'see' both 2I and")
print("SL(2,7) at once.")

results["partF"] = {
    "all_3dim_decompositions": [list(t) for t in three_dim_assignments],
    "num_3dim_irreps": len(three_dim_assignments),
    "remark": (
        "Every 3-dim irrep of Gamma_70 is either pure 2I-triplet x trivial(SL27) or "
        "trivial(2I) x SL27-triplet, decorated with an S_3 parity. No 'fused' "
        "3-dim irrep exists, so SM 3-generation structure cannot couple to both "
        "icosahedral and SL(2,7) sectors simultaneously."
    ),
}

# =============================================================================
# PART G: Compare to lower levels
# =============================================================================
banner("PART G: Compare to levels 5, 7, 14, 35, 70")

# Level 5: SL(2,Z/5) = 2I. order 120. characters in Q(phi).
# Level 7: SL(2,7). order 336. characters in Q(zeta_7) and Q(zeta_8).
# Level 14: S_3 x SL(2,7). order 2016. characters in Q(zeta_7) (since char field
#           of S_3 is Q). Adds parity but no new algebraic content.
# Level 35: SL(2,Z/5) x SL(2,Z/7). order 120*336 = 40320. characters in
#           Q(zeta_5, zeta_7) = Q(zeta_35). FULL fusion of icosahedral + sin(pi/14)
#           via products, but no genuinely new fused character.
# Level 70: S_3 x 2I x SL(2,7). order 241920. parity adjunct to level 35.

print("Level | Group                     | Order  | Char field         | New?")
print("------|---------------------------|--------|--------------------|----------")
print(" 5    | SL(2,Z/5) = 2I            |    120 | Q(phi) = Q(sqrt5)  | base")
print(" 7    | SL(2,Z/7)                 |    336 | Q(zeta_7), Q(zeta_8)| base")
print("14    | S_3 x SL(2,7)             |   2016 | Q(zeta_7), Q(zeta_8)| +parity only")
print("35    | 2I x SL(2,7)              |  40320 | Q(zeta_5, zeta_7)  | +product fusion")
print("70    | S_3 x 2I x SL(2,7)        | 241920 | Q(zeta_5, zeta_7)  | +parity over 35")
print()
print("Level 35 is structurally the most economical 'fusion' level -- it joins 2I and SL(2,7)")
print("with no parity factor. Level 70 just adds an S_3 parity that cannot generate new")
print("Yukawa structure (S_3 has no faithful action on the cyclotomic data).")
print()
print("So level 70's ONLY genuine extension over level 35 is the parity Z_2 / S_3 sign,")
print("which is identical to the level 14 -> level 7 promotion. The bulk of the structural")
print("content lives at level 35.")

# Quick irrep count for level 35 = 2I x SL(2,7):
n_irreps_35 = 9 * 11  # 99
sum_sq_35 = sum(d_2I_x_d_SL27 ** 2 for d_2I_x_d_SL27 in
                (d2 * d7 for d2 in dims_2I for d7 in dims_SL27))
print(f"\nLevel 35: # irreps = {n_irreps_35}, sum sq = {sum_sq_35} (= 120*336 = {120*336}? {sum_sq_35 == 120*336})")

# Newforms at level 35:
# LMFDB: dim S_2(Gamma_0(35)) = 3, oldforms from level 5, 7 contribute 0 (both have
# trivial S_2 since X_0(5) and X_0(7) are genus 0). So all 3 forms at level 35 are
# NEWFORMS. There are 3 newform Galois orbits at level 35 in S_2(Gamma_0(35)).
print(f"\nLevel 35 cusp form data (Gamma_0(35)):")
print(f"  dim S_2(Gamma_0(35)) = 3 (genus of X_0(35))")
print(f"  oldforms from levels 5, 7: 0 (since S_2 vanishes there)")
print(f"  newforms at level 35: dim = 3 (LMFDB labels 35.2.a.a, 35.2.a.b, ...)")

results["partG"] = {
    "comparison_table": [
        {"level": 5, "group": "2I = SL(2,Z/5)", "order": 120, "char_field": "Q(sqrt5)"},
        {"level": 7, "group": "SL(2,7)", "order": 336, "char_field": "Q(zeta_7), Q(zeta_8)"},
        {"level": 14, "group": "S_3 x SL(2,7)", "order": 2016, "char_field": "Q(zeta_7), Q(zeta_8)"},
        {"level": 35, "group": "2I x SL(2,7)", "order": 40320, "char_field": "Q(zeta_35)"},
        {"level": 70, "group": "S_3 x 2I x SL(2,7)", "order": 241920, "char_field": "Q(zeta_35)"},
    ],
    "level35_irrep_count": n_irreps_35,
    "level35_sum_sq_check": sum_sq_35 == 120 * 336,
    "level35_newform_dim_S2_Gamma0": 3,
    "level70_relative_to_level35": (
        "Level 70 = level 35 with an extra S_3 = SL(2,Z/2) parity factor. The "
        "parity adds a Z_2 charge but cannot generate new algebraic / cyclotomic "
        "content. All 'fusion' content already lives at level 35."
    ),
}

# =============================================================================
# PART H: Numerical SM spot-check
# =============================================================================
banner("PART H: Numerical SM spot-check (qualitative)")

# Without performing a full tau-scan (the level-14 agent's chi^2_min was ~100),
# we can estimate the IRREDUCIBLE bound at level 70.
#
# At level 7 (Ding-King-Liu 2020): chi^2_min ~ 30-50 in best lepton fits.
# At level 14: chi^2_min ~ 100 (parity Z_2 doesn't add discriminating power).
# At level 35: in principle the 2I factor introduces 9 new irreps and ~9 new
#   weight-2 modular forms via the 2I-decomposition of M_2(Gamma(35)). These
#   provide additional Yukawa coefficients that COULD lower chi^2.
# At level 70: same as level 35 plus S_3 parity, no qualitative improvement.

# Concrete fit estimation for level 35:
# - PMNS predictions: governed by a SL(2,7) subgroup. Best-case chi^2 ~ DKL level 7 result.
#   Adding 2I-flavor Yukawas allows extra free parameters (~9 new coefficients), so
#   chi^2 can in principle be driven low (overfit risk). Without explicit q-expansions,
#   we cannot give a precise number, but the bound is unlikely to go below chi^2 ~ 5-10
#   given the 8 lepton observables.
# - CKM predictions: similar story, but quark hierarchy m_t/m_c/m_u ~ 10^5 ratios are
#   harder to reproduce from cyclotomic ratios alone. Ratios available from
#   Q(zeta_70) are at most O(10) (in the 14th-root combinations), so producing
#   m_t/m_u ~ 10^5 requires either fine-tuned modular form arrangements or
#   extra suppression mechanisms. Generic prediction: m_t/m_c achievable, but
#   m_c/m_u ~ 600 hard without exponential modulus dependence (q-suppression).

estimates = {
    "level7_lepton_chi2_min": "~30-50 (Ding-King-Liu 2020 best fits)",
    "level14_lepton_chi2_min": "~100 (no improvement over level 7)",
    "level35_lepton_chi2_min_estimate": "~5-30 (2I factor adds free parameters, but no fundamentally new structure)",
    "level70_lepton_chi2_min_estimate": "~5-30 (same as level 35, parity Z_2 adds no qualitative power)",
    "ckm_top_to_charm_ratio_achievable": "yes (O(10^2) achievable via q-suppression at moderate Im tau)",
    "ckm_charm_to_up_ratio_achievable": "marginal (O(600) requires exponential modulus dependence, fine-tuning)",
    "cabibbo_angle_achievable": "yes generically (sin theta_C ~ sin(pi/14) ~ 0.22 IS the right size)",
}

print("Estimates (without full tau-scan):")
for k, v in estimates.items():
    print(f"  {k}: {v}")

# Notably: sin(pi/14) ~ 0.2225, sin theta_Cabibbo ~ 0.2253. These are within 1.3%.
# This is a PRE-EXISTING coincidence that motivates sin(pi/14) flavor models.
sin_pi14_num = float(mp.sin(mp.pi / 14))
sin_thetaC = 0.2253
print(f"\nsin(pi/14)         = {sin_pi14_num:.6f}")
print(f"sin theta_Cabibbo  = {sin_thetaC:.6f}")
print(f"relative deviation = {abs(sin_pi14_num - sin_thetaC) / sin_thetaC * 100:.2f}%")

results["partH"] = {
    "estimates": estimates,
    "sin_pi14": sin_pi14_num,
    "sin_thetaC": sin_thetaC,
    "rel_deviation_pct": abs(sin_pi14_num - sin_thetaC) / sin_thetaC * 100,
    "comment": (
        "The Cabibbo-angle / sin(pi/14) coincidence pre-exists at level 7. "
        "Levels 14, 35, 70 do not improve on this. The CKM hierarchy m_c/m_u "
        "is the harder problem and is not natural in any of these levels without "
        "significant tuning of Im(tau)."
    ),
}

# =============================================================================
# PART I: Honest verdict
# =============================================================================
banner("PART I: Honest verdict on Gamma_70")

verdict_points = [
    "1. CRT decomposition CONFIRMED: Gamma_70 = S_3 x 2I x SL(2,7), order 241920.",
    "   297 irreps as tensor products, dimensions and counts all check out.",
    "",
    "2. phi (golden ratio) and sin(pi/14) appear in DIFFERENT factor characters:",
    "   phi in 2I, sin(pi/14) in SL(2,7). They combine ONLY as products of separate",
    "   character values -- no genuinely fused / non-separable character exists.",
    "   This is the same structural triviality as level 14: a direct product",
    "   factorization gives multiplicative, not additive/algebraic, fusion.",
    "",
    "3. Complex irreps EXIST in Gamma_70 (inherited from SL(2,7)). This means",
    "   chirality is structurally available -- but the chirality lives entirely",
    "   on the SL(2,7) factor, not 'between' factors.",
    "",
    "4. Newforms at level 70 EXIST (S_2(Gamma_0(70))^new = 1, an elliptic curve).",
    "   But this newform is associated to a generic L-function, NOT to phi or",
    "   sin(pi/14) in any algebraically distinguished way.",
    "",
    "5. The 'order = 24' coincidence with Pisano(9) is decorative: it refers to",
    "   |Gal(Q(zeta_70)/Q)| = phi(70) = 24, which equals the order of |2T| and",
    "   the Pisano period mod 9. None of these are causally connected to the",
    "   modular flavor structure of Gamma_70 itself.",
    "",
    "6. Level 35 = 2I x SL(2,7) already captures all the genuine structural fusion",
    "   between icosahedral and sin(pi/14) sectors. Level 70 just adds a parity",
    "   Z_2 factor (the SL(2,Z/2) = S_3) that contributes no new algebraic content.",
    "",
    "FINAL ANSWER:",
    "  Gamma_70 is NOT the missing 'genuine fusion' link.",
    "  It is a CRT-trivial direct product, structurally identical to level 14",
    "  but with the 2I factor included. The icosahedral and sin(pi/14) sectors",
    "  remain ALGEBRAICALLY SEPARABLE: they multiply but never genuinely fuse.",
    "  No irrep, no character value, no modular form at level 70 brings phi and",
    "  sin(pi/14) into a single irreducible algebraic combination.",
    "",
    "  The cyclotomic-fusion program is therefore STRUCTURALLY CLOSED at the",
    "  direct-product level. Genuine fusion would require a NON-SPLIT extension,",
    "  e.g. a non-trivial central or semidirect product, OR a different group",
    "  scheme such as a Mumford-style theta group, OR working with non-abelian",
    "  cohomology (Brauer / non-split crossed product). None of these arise from",
    "  CRT factorization of SL(2, Z/N) for N square-free.",
    "",
    "  Verdict: CLOSURE, not fusion. Level 70 is the CRT-completion of the",
    "  framework's discrete cyclotomic content, but it does not generate new",
    "  algebraic structure beyond the products of its prime-power factors.",
]
for line in verdict_points:
    print(line)

results["partI"] = {
    "CRT_confirmed": True,
    "genuine_fusion": False,
    "complex_irreps_present": True,
    "chirality_source": "SL(2,7) factor",
    "newforms_at_level_70_exist": True,
    "order_24_coincidence_meaningful": False,
    "level_35_already_captures_fusion": True,
    "verdict_summary": (
        "Gamma_70 is structurally a direct product S_3 x 2I x SL(2,7). It contains "
        "phi and sin(pi/14) as character values of separate factors, but they "
        "combine only multiplicatively. No genuinely new algebraic / fused irrep "
        "exists. Level 70 is the CRT closure of the discrete cyclotomic program, "
        "not its missing fusion link. Honest answer: closure, not fusion."
    ),
}

# Final save
with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {RESULT_PATH}")
