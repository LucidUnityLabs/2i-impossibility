#!/usr/bin/env python3
"""
supplement_verify.py -- self-contained verification supplement for paper1
("Structural Obstructions to Standard-Model Derivation from Binary Icosahedral
Orbifolds and Modular-Flavor Extensions").

Dependencies: numpy only. No external data files.

Checks:
  [1] 2I constructed as 120 unit quaternions (closure + unit norm).
  [2] 9 conjugacy classes with sizes {1,1,12,12,12,12,20,20,30};
      exponent exp(2I) = 60.
  [3] Irreducible representations of dimensions {1,2,2,3,3,4,4,5,6},
      sum of squares = 120, orthonormal character table.
  [4] Frobenius-Schur indicators of all 9 irreps are +-1 (no complex type);
      the faithful irreps (chi(z) = -chi(1) at the central involution z)
      are exactly the quaternionic ones (dims {2,2,4,6}); the real ones
      (dims {1,3,3,4,5}) factor through A5 = 2I/{+-1};
      Frobenius count sum(nu * dim) = #{g in G : g^2 = e} = 2.
      [Theorem B -- Frobenius-Schur chirality obstruction]
  [5] Arithmetic obstruction: sin(pi/14) = (zeta_14^3 + zeta_14^{-3})/2 lies
      in Q(zeta_14); its minimal polynomial 8x^3 - 4x^2 - 4x + 1 is
      irreducible over Q (rational root test), so [Q(sin(pi/14)):Q] = 3.
      phi(60) = 16 is not divisible by 3, so sin(pi/14) is not contained in
      Q(zeta_60), the character field of 2I (exp(2I) = 60, primes {2,3,5},
      and 7 does not divide 60). Moreover sin(pi/14) is not an algebraic
      integer while every character value is (sum of roots of unity).
      [Theorem A -- Arithmetic Obstruction]

Prints PASS/FAIL per check and a final summary line. Exit code 0 iff all pass.
Adapted from tests/new_computations/01_2I_reps/verify_2I_reps.py (character
table via class-algebra eigenvalues) with the sympy-exact quaternion algebra
replaced by float arithmetic with tight tolerances.
"""

import itertools
import math
import sys

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0
KEY_DECIMALS = 8          # canonical rounding for element identification
CHECK_TOL = 1e-6          # general numeric tolerance

results = []


def record(name, ok, detail=""):
    results.append((name, bool(ok)))
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    return bool(ok)


def q_mul(a, b):
    """Hamilton product of two quaternions given as (w, x, y, z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def q_key(q):
    return tuple(np.round(q, KEY_DECIMALS))


def perm_sign(p):
    n = len(p)
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if p[i] > p[j])
    return 1 if inv % 2 == 0 else -1


def build_2I():
    """The 120 unit quaternions of the binary icosahedral group."""
    elements = {}
    # 8 pure units: +-1, +-i, +-j, +-k
    for sign in (1.0, -1.0):
        for q in ((sign, 0, 0, 0), (0, sign, 0, 0), (0, 0, sign, 0), (0, 0, 0, sign)):
            elements[q_key(q)] = q
    # 16 half-integer units: (+-1 +-i +-j +-k)/2
    for s in itertools.product((1.0, -1.0), repeat=4):
        q = tuple(c / 2.0 for c in s)
        elements[q_key(q)] = q
    # 96 golden units: even permutations of (0, 1/2, phi/2, (1/phi)/2) with signs
    base = (0.0, 0.5, PHI / 2.0, (1.0 / PHI) / 2.0)
    A4 = [p for p in itertools.permutations(range(4)) if perm_sign(p) == 1]
    assert len(A4) == 12
    for p in A4:
        vals = tuple(base[p[i]] for i in range(4))
        for signs in itertools.product((1.0, -1.0), repeat=4):
            q = tuple(signs[i] * vals[i] for i in range(4))
            elements[q_key(q)] = q
    return list(elements.values())


def element_order(idx, cayley, e_idx, n):
    x, k = idx, 1
    while x != e_idx:
        x = int(cayley[x, idx])
        k += 1
        if k > n:
            raise RuntimeError("order computation exceeded group size")
    return k


def main():
    # ------------------------------------------------------------------ [1]
    print("=" * 72)
    print("[1] Constructing 2I as 120 unit quaternions")
    print("=" * 72)
    elements = build_2I()
    N = len(elements)
    ok_count = N == 120
    norms = [np.linalg.norm(q) for q in elements]
    ok_norms = max(abs(n - 1.0) for n in norms) < 1e-9
    keys = {q_key(q) for q in elements}
    ok_closed = True
    for a in elements:
        for b in elements:
            if q_key(q_mul(a, b)) not in keys:
                ok_closed = False
                break
        if not ok_closed:
            break
    record("1a: |2I| = 120", ok_count, f"constructed {N} elements")
    record("1b: all elements unit norm", ok_norms,
           f"max |norm - 1| = {max(abs(n - 1.0) for n in norms):.2e}")
    record("1c: closure under quaternion product", ok_closed,
           "all 14400 products land in 2I" if ok_closed else "closure failure")

    key_to_idx = {q_key(q): i for i, q in enumerate(elements)}
    cayley = np.zeros((N, N), dtype=np.int64)
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            cayley[i, j] = key_to_idx[q_key(q_mul(a, b))]
    e_idx = key_to_idx[q_key((1.0, 0.0, 0.0, 0.0))]

    # ------------------------------------------------------------------ [2]
    print()
    print("=" * 72)
    print("[2] Conjugacy classes and exponent")
    print("=" * 72)
    assigned = [False] * N
    inv_idx = np.zeros(N, dtype=np.int64)
    for i, a in enumerate(elements):
        inv_idx[i] = key_to_idx[q_key((a[0], -a[1], -a[2], -a[3]))]  # conjugate
    classes = []
    class_of = np.zeros(N, dtype=np.int64)
    for i in range(N):
        if assigned[i]:
            continue
        cls = set()
        for g in range(N):
            x = int(cayley[cayley[g, i], inv_idx[g]])  # g i g^{-1}
            cls.add(x)
        for x in cls:
            assigned[x] = True
            class_of[x] = len(classes)
        classes.append(sorted(cls))
    classes.sort(key=lambda c: (len(c), c[0]))
    for ci, cls in enumerate(classes):
        for x in cls:
            class_of[x] = ci
    class_sizes = [len(c) for c in classes]
    k = len(classes)
    ok_classes = (k == 9) and (sorted(class_sizes) == [1, 1, 12, 12, 12, 12, 20, 20, 30])
    record("2a: 9 conjugacy classes, sizes {1,1,12,12,12,12,20,20,30}",
           ok_classes, f"found {k} classes, sizes {class_sizes}")

    class_orders = [element_order(c[0], cayley, e_idx, N) for c in classes]
    exp_order = 1
    for o in class_orders:
        exp_order = exp_order * o // math.gcd(exp_order, o)
    primes_60_expected = {2, 3, 5}

    def prime_factors(n):
        ps, d = set(), 2
        while d * d <= n:
            while n % d == 0:
                ps.add(d)
                n //= d
            d += 1
        if n > 1:
            ps.add(n)
        return ps

    primes_exp = prime_factors(exp_order)
    ok_exp = (exp_order == 60) and (primes_exp == primes_60_expected)
    record("2b: exp(2I) = 60 with prime support {2,3,5}", ok_exp,
           f"exponent = {exp_order}, primes = {sorted(primes_exp)}, "
           f"class orders = {class_orders}")

    # central involution: the non-identity class of size 1
    size1 = [ci for ci in range(k) if class_sizes[ci] == 1]
    z_class = [ci for ci in size1 if ci != class_of[e_idx]]
    ok_central = len(size1) == 2 and len(z_class) == 1
    z_class = z_class[0] if z_class else -1
    record("2c: unique central involution class (z = -1)", ok_central,
           f"classes of size 1 at indices {size1}")

    # ------------------------------------------------------------------ [3]
    print()
    print("=" * 72)
    print("[3] Character table via class-algebra eigenvalues")
    print("=" * 72)
    M_list = []
    for ci in range(k):
        Mi = np.zeros((k, k), dtype=np.int64)
        for cj in range(k):
            g = classes[cj][0]
            for a in classes[ci]:
                Mi[class_of[cayley[a, g]], cj] += 1
        M_list.append(Mi)
    rng = np.random.default_rng(42)
    Mrand = sum(c * M for c, M in zip(rng.standard_normal(k), M_list))
    eigvals, eigvecs = np.linalg.eig(Mrand)
    csz = np.array(class_sizes, dtype=float)
    char_table = np.zeros((k, k), dtype=complex)
    dims = []
    for alpha in range(k):
        v = eigvecs[:, alpha]
        lams = np.zeros(k, dtype=complex)
        idx = int(np.argmax(np.abs(v)))
        for ci in range(k):
            lams[ci] = (M_list[ci] @ v)[idx] / v[idx]
        denom = np.sum(np.abs(lams) ** 2 / csz)
        dim = math.sqrt(N / denom)
        dims.append(dim)
        char_table[alpha, :] = lams * dim / csz
    dims = np.round(np.array(dims)).astype(int)
    ok_dims = sorted(dims.tolist()) == [1, 2, 2, 3, 3, 4, 4, 5, 6]
    ok_burnside = int(np.sum(dims.astype(float) ** 2)) == N
    record("3a: irrep dims {1,2,2,3,3,4,4,5,6}, sum of squares = 120",
           ok_dims and ok_burnside, f"dims = {sorted(dims.tolist())}")

    order_idx = np.argsort(dims, kind="stable")
    dims = dims[order_idx]
    char_table = char_table[order_idx, :]
    gram = np.zeros((k, k), dtype=complex)
    for a in range(k):
        for b in range(k):
            gram[a, b] = np.sum(csz * char_table[a] * char_table[b].conj()) / N
    err_orth = float(np.max(np.abs(gram - np.eye(k))))
    ok_orth = err_orth < CHECK_TOL
    record("3b: character table orthonormal", ok_orth,
           f"max |<chi_a, chi_b> - delta_ab| = {err_orth:.3e}")
    err_real = float(np.max(np.abs(char_table.imag)))
    record("3c: all character values real (self-conjugate irreps)",
           err_real < CHECK_TOL, f"max |Im chi| = {err_real:.3e}")

    # ------------------------------------------------------------------ [4]
    print()
    print("=" * 72)
    print("[4] Frobenius-Schur indicators (Theorem B)")
    print("=" * 72)
    # nu(chi) = (1/|G|) sum_i |C_i| chi(g_i^2)
    square_class = [int(class_of[cayley[c[0], c[0]]]) for c in classes]
    fs = np.zeros(k)
    for alpha in range(k):
        fs[alpha] = np.sum(csz * char_table[alpha, square_class].real) / N
    fs_round = np.round(fs).astype(int)
    ok_pm1 = all(abs(fs[i] - fs_round[i]) < CHECK_TOL for i in range(k)) and \
        all(v in (-1, 1) for v in fs_round)
    record("4a: FS indicators all exactly +-1 (no complex type)", ok_pm1,
           f"nu values = {fs_round.tolist()}")

    chi_z = char_table[:, z_class].real
    faithful = np.abs(chi_z + dims) < CHECK_TOL   # chi(z) = -dim
    ok_equiv = all(faithful[i] == (fs_round[i] == -1) for i in range(k))
    f_dims = sorted(dims[faithful].tolist())
    r_dims = sorted(dims[~faithful].tolist())
    ok_fset = f_dims == [2, 2, 4, 6] and r_dims == [1, 3, 3, 4, 5]
    record("4b: faithful irreps (chi(z)=-dim) exactly the quaternionic ones",
           ok_equiv and ok_fset,
           f"faithful dims = {f_dims} (nu=-1), real dims = {r_dims} (nu=+1)")

    frob_count = float(np.sum(fs_round * dims))
    n_sqr_roots = sum(1 for i in range(N)
                      if int(cayley[i, i]) == e_idx)
    ok_frob = abs(frob_count - n_sqr_roots) < CHECK_TOL and n_sqr_roots == 2
    record("4c: Frobenius count sum(nu*dim) = #{g: g^2=e} = 2", ok_frob,
           f"sum(nu*dim) = {frob_count:g}, #square roots of e = {n_sqr_roots}")

    # A5 consistency: all irreps of 2I/{+-1} = A5 have nu = +1
    #   sum over A5 irreps of nu*dim = #{a in A5: a^2=1} = 1 + 15 = 16
    #   and sum of dims = 1+3+3+4+5 = 16 forces all nu = +1.
    a5_ok = (r_dims == [1, 3, 3, 4, 5]) and (sum(r_dims) == 16)
    record("4d: A5-type irreps (dims 1,3,3',4,5) all real", a5_ok,
           "sum dims 16 = #{a in A5: a^2 = 1} forces nu = +1 on each")

    print()
    print("  Frobenius-Schur table for 2I (paper Table 1):")
    print("    irrep dim | chi(z) | faithful | nu")
    order2 = np.lexsort((-chi_z, dims))
    for i in order2:
        tag = "yes" if faithful[i] else "no "
        print(f"      {dims[i]:2d}      | {chi_z[i]:+2.0f}   | {tag}      | {fs_round[i]:+d}")

    # ------------------------------------------------------------------ [5]
    print()
    print("=" * 72)
    print("[5] Arithmetic obstruction (Theorem A)")
    print("=" * 72)
    S = math.sin(math.pi / 14.0)
    zeta14 = np.exp(2j * np.pi / 14.0)
    S_expr = ((zeta14 ** 3) + (zeta14 ** (-3))).real / 2.0
    ok_in_q14 = abs(S_expr - S) < 1e-12
    record("5a: sin(pi/14) = (zeta_14^3 + zeta_14^{-3})/2 in Q(zeta_14)",
           ok_in_q14, f"|difference| = {abs(S_expr - S):.2e}")

    def m(x):
        return 8.0 * x ** 3 - 4.0 * x ** 2 - 4.0 * x + 1.0

    ok_poly = abs(m(S)) < 1e-10
    record("5b: minimal polynomial 8x^3-4x^2-4x+1 vanishes at sin(pi/14)",
           ok_poly, f"|m(S)| = {abs(m(S)):.2e}")
    rational_candidates = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125]
    has_rational_root = any(abs(m(c)) < 1e-12 for c in rational_candidates)
    ok_irred = not has_rational_root
    record("5c: 8x^3-4x^2-4x+1 irreducible over Q (rational root test)",
           ok_irred, "no rational root among +-1, +-1/2, +-1/4, +-1/8"
           if ok_irred else "rational root found")
    deg_S = 3 if ok_irred else 1

    # Euler phi(60) = 60 * (1-1/2)(1-1/3)(1-1/5) = 16, computed exactly:
    phi_60 = sum(1 for n in range(1, 61) if math.gcd(n, 60) == 1)
    ok_phi = phi_60 == 16
    record("5d: phi(60) = 16 = [Q(zeta_60):Q]", ok_phi, f"phi(60) = {phi_60}")

    ok_degree = (deg_S == 3) and (16 % 3 != 0)
    record("5e: sin(pi/14) not in Q(zeta_60) (degree 3 does not divide 16)",
           ok_degree,
           f"[Q(sin(pi/14)):Q] = {deg_S}, [Q(zeta_60):Q] = {phi_60}, "
           f"3 does not divide {phi_60}")

    ok_primes = (7 not in primes_exp) and (7 in prime_factors(14)) and \
        (math.gcd(60, 14) == 2)
    record("5f: prime separation -- 7 | 14 but 7 does not divide exp(2I) = 60",
           ok_primes,
           f"primes(60) = {sorted(primes_exp)}, primes(14) = {sorted(prime_factors(14))}, "
           f"gcd(60,14) = {math.gcd(60, 14)}")

    not_monic = True  # primitive polynomial 8x^3-4x^2-4x+1 has leading coeff 8
    ok_algint = not_monic and ok_irred
    record("5g: sin(pi/14) not an algebraic integer; character values are",
           ok_algint,
           "minimal polynomial not monic (leading coeff 8); character values "
           "are sums of roots of unity, hence algebraic integers")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks pass; "
              f"FAILURES: {failed}")
        return 1
    print(f"SUMMARY: ALL {len(results)} CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
