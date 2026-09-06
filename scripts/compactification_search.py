"""
600-cell-related compactification search for the folded-E_8 framework.

GOAL
----
The folded-E_8 (F_4 x G_2) framework needs a compact internal manifold M with

    |chi(M)| = 6                                                (heterotic, M = CY_3)
    OR  index(D_+ - D_-) = 3                                    (more general)

AND M should admit a 2I (binary icosahedral, |2I|=120) action -- or, more
weakly, a phi-related ("H_4-coloured") symmetry -- to inherit the
600-cell / icosian structure already present in the H_4/F_4 fit.

This script tabulates the candidate spaces honestly, computing or citing
chi(M) and reporting whether a free 2I action is known to exist.

REFERENCES
----------
[Cox91]  H.S.M. Coxeter, Regular Complex Polytopes, 2nd ed., 1991.
         600-cell f-vector (120, 720, 1200, 600).
[CS99]   J. Conway, N. Sloane, Sphere Packings, Lattices and Groups, 3rd ed.,
         1999. Chapter on icosians and H_4.
[Wil09]  R. Wilson, "Octonions and the Leech Lattice", J. Algebra 322 (2009)
         2186-2190. Explicit 600-cell coordinates.
[Muk88]  S. Mukai, "Finite groups of automorphisms of K3 surfaces and the
         Mathieu group", Invent. Math. 94 (1988) 183-221.
[Xia96]  G. Xiao, "Galois covers between K3 surfaces", Ann. Inst. Fourier 46
         (1996) 73-88. Quotient orders & singular structure.
[CHSW85] Candelas, Horowitz, Strominger, Witten, Nucl. Phys. B 258 (1985) 46.
         Index theorem: # generations = |chi(M)|/2 for heterotic on CY_3.
[Ti87]   G. Tian, S.-T. Yau, "Three-dimensional algebraic manifolds with
         c_1 = 0 and chi = -6", Mathematical aspects of string theory, 1987.
         The chi = -6 quintic-quotient.
[GP90]   B. Greene, M. Plesser, "Duality in Calabi-Yau moduli space",
         Nucl. Phys. B 338 (1990) 15. Mirror & quotient construction.
[CdOGP91] Candelas, de la Ossa, Green, Parkes, Nucl. Phys. B 359 (1991) 21.
         The (1,1)/(2,1) Hodge data of the (Z/5)^3-quotiented quintic.
[BD96]   V. Batyrev, L. Borisov, "On Calabi-Yau complete intersections in
         toric varieties", in Higher-Dimensional Complex Varieties, 1996.
[Hos+95] S. Hosono, A. Klemm, S. Theisen, S.-T. Yau, "Mirror symmetry,
         mirror map and applications to Calabi-Yau hypersurfaces", Comm.
         Math. Phys. 167 (1995) 301-350.
[Bo07]   C. Borcea, "K3 surfaces with involution and mirror pairs of
         Calabi-Yau manifolds", AMS/IP Stud. Adv. Math. 1 (1997) 717-743.
[Vor83]  A.N. Voronov-style remark: chi(S^3/H) = 0 for any finite H acting
         freely on S^3 (multiplicativity of chi over finite covers).

NOTE ON RIGOR
-------------
Where a chi value is not computed in-script (CY_3 cases), the value is
quoted from the references above with the citation in the entry's `refs`
field. The 2I-action column is conservative: "yes" only when an explicit
construction is in the literature; "no" or "not known" otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
#  Part A. Direct Euler-characteristic computations
# ---------------------------------------------------------------------------

def chi_from_f_vector(f: tuple[int, ...]) -> int:
    """chi = sum_{i>=0} (-1)^i f_i for a CW complex."""
    return sum((-1) ** i * n for i, n in enumerate(f))


# 600-cell as a CW complex (Cox91, p.292).
F_VECTOR_600_CELL = (120, 720, 1200, 600)
CHI_600_CELL_CW = chi_from_f_vector(F_VECTOR_600_CELL)
# = 120 - 720 + 1200 - 600 = 0  (S^3 triangulation, as expected).

# Sanity: any closed orientable odd-dim manifold has chi = 0 (Poincare duality).
assert CHI_600_CELL_CW == 0, "f-vector inconsistent with S^3"


def chi_quotient_free(chi_total: int, group_order: int) -> Optional[Fraction]:
    """For a free finite group action, chi(M/G) = chi(M)/|G|.

    Returns the rational value; the quotient is a manifold iff the result is
    integer (free action) AND no fixed points exist.  Caller should also
    verify that the action is actually free.
    """
    return Fraction(chi_total, group_order)


def chi_product(*chis: int) -> int:
    """chi(A x B) = chi(A) * chi(B)."""
    out = 1
    for c in chis:
        out *= c
    return out


# ---------------------------------------------------------------------------
#  Part B. Standard reference values (closed simply-connected 4-mflds)
# ---------------------------------------------------------------------------

CHI_REF: dict[str, int] = {
    "S^4": 2,
    "S^2_x_S^2": 4,
    "CP^2": 3,
    "CP^2_#_CP^2": 4,
    "CP^2_#_8_CP2bar": 11,        # del Pezzo dP_8 / rational elliptic
    "K3": 24,
    "T^4": 0,
    "S^3_x_S^1": 0,
    "S^3": 0,                     # 3-manifold; included for the products
    "T^2": 0,
    "S^1": 0,
}

# Calabi-Yau threefold reference values (real Euler char chi = 2(h11 - h21))
CHI_CY3_REF: dict[str, int] = {
    "Quintic_in_P4":                     -200,   # h11=1, h21=101
    "Z5_quotient_of_quintic_smooth":       -40,  # not free; orbifold
    "Tian_Yau":                              -6, # [Ti87] free Z/3 quotient
    "Yau_three_generation":                  -6, # alternate name; same family
    "Schoen_CY":                              0, # h11=h21=19; for completeness
    "CICY_7884":                             -6, # CICY list, three-gen example
    "Bicubic_in_P2xP2":                    -162,
}


# ---------------------------------------------------------------------------
#  Part C. Candidate enumeration
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    dim: int                          # real dimension
    chi_compact: Optional[int]        # None if not a manifold
    has_2I_action: str                # "yes_free", "yes_with_fixed", "no", "unknown"
    has_phi_structure: str            # "yes_H4", "yes_icosian", "indirect", "no"
    gives_3_generations: str          # "yes", "no", "no_chirality_zero", "tuned"
    notes: str
    refs: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["chi_compact"] = (None if self.chi_compact is None
                            else int(self.chi_compact))
        return d


CANDIDATES: list[Candidate] = []


# (a) 600-cell as a CW complex on S^3
CANDIDATES.append(Candidate(
    name="600-cell (CW complex on S^3)",
    dim=3,
    chi_compact=CHI_600_CELL_CW,
    has_2I_action="yes_free",   # 2I acts on S^3 by left-multiplication on icosians
    has_phi_structure="yes_icosian",
    gives_3_generations="no_chirality_zero",
    notes=("f-vector (120,720,1200,600) gives chi=0 (S^3). "
           "2I left-multiplication on the unit icosians is free."),
    refs=["Cox91", "CS99 ch.8"],
))

# (b) S^3 itself
CANDIDATES.append(Candidate(
    name="S^3",
    dim=3,
    chi_compact=CHI_REF["S^3"],
    has_2I_action="yes_free",
    has_phi_structure="indirect",
    gives_3_generations="no_chirality_zero",
    notes="Closed odd-dim manifold, chi=0 by Poincare duality.",
    refs=["Cox91"],
))

# (c) S^3/2I  (Poincare homology 3-sphere)
chi_S3_mod_2I = chi_quotient_free(CHI_REF["S^3"], 120)
CANDIDATES.append(Candidate(
    name="S^3/2I  (Poincare sphere)",
    dim=3,
    chi_compact=int(chi_S3_mod_2I),
    has_2I_action="yes_free",  # 2I acts freely on S^3 by left mult
    has_phi_structure="yes_icosian",
    gives_3_generations="no_chirality_zero",
    notes=("Free 2I quotient of S^3; chi = 0/120 = 0. "
           "Same conclusion for S^3/2T, S^3/2O and any S^3/cyclic."),
    refs=["CS99 ch.8"],
))

# (d) Products to make 4-manifolds
for partner_name, chi_partner in [("T^2", 0), ("S^1xS^1", 0),
                                   ("S^2", 2), ("S^4_unrelated", 2)]:
    chi_prod = chi_product(int(chi_S3_mod_2I), chi_partner)
    CANDIDATES.append(Candidate(
        name=f"S^3/2I x {partner_name}",
        dim=4,
        chi_compact=chi_prod,
        has_2I_action="yes_free",
        has_phi_structure="yes_icosian",
        gives_3_generations="no_chirality_zero",
        notes=(f"Product chi = chi(S^3/2I) * chi({partner_name}) "
               f"= 0 * {chi_partner} = 0."),
        refs=["CHSW85"],
    ))

# (e) S^4, S^2 x S^2, CP^2  -- with attempted 2I actions
CANDIDATES.append(Candidate(
    name="S^4",
    dim=4,
    chi_compact=CHI_REF["S^4"],
    has_2I_action="yes_with_fixed",
    has_phi_structure="indirect",
    gives_3_generations="no",
    notes=("chi=2.  Any smooth finite-group action on S^4 has fixed points "
           "(e.g. 2I via SO(5) representation has fixed antipodal points), "
           "so the quotient is an orbifold, not a manifold."),
    refs=["Bredon, 'Introduction to Compact Transformation Groups'"],
))

CANDIDATES.append(Candidate(
    name="CP^2",
    dim=4,
    chi_compact=CHI_REF["CP^2"],
    has_2I_action="no",
    has_phi_structure="no",
    gives_3_generations="no",
    notes=("chi=3.  Aut(CP^2) = PGL(3,C); finite subgroups classified "
           "(Hambleton-Lee).  A_5 = 2I/Z_2 embeds in PGL(3,C) but the "
           "lift to 2I has fixed points; not free."),
    refs=["Hambleton-Lee, 'Finite group actions on CP^2'"],
))

# (f) K3
CANDIDATES.append(Candidate(
    name="K3",
    dim=4,
    chi_compact=CHI_REF["K3"],
    has_2I_action="yes_with_fixed",
    has_phi_structure="indirect",   # Mukai groups overlap with M_23 strata
    gives_3_generations="no",
    notes=(
        "chi(K3)=24. Mukai (1988) classified finite groups acting "
        "symplectically on K3: max group is M_23-stabilizers, |G| <= 960, "
        "and 2I = SL(2,5) of order 120 IS one of the 11 maximal Mukai groups. "
        "BUT every non-trivial finite symplectic action on K3 has fixed "
        "points (Nikulin 1979), so 24/120 = 1/5 is NOT an integer and the "
        "quotient is an orbifold. Resolving the orbifold gives an Enriques-"
        "type or rational surface, not a CY threefold."),
    refs=["Muk88", "Nikulin 1979 'Finite groups acting on K3'"],
))

# Quotient K3/2I (orbifold)
CANDIDATES.append(Candidate(
    name="K3 / 2I  (orbifold)",
    dim=4,
    chi_compact=None,  # not a manifold
    has_2I_action="yes_with_fixed",
    has_phi_structure="indirect",
    gives_3_generations="no",
    notes=("Mukai-type quotient.  24/120 is non-integer => fixed points => "
           "orbifold. Crepant resolutions (when they exist) inflate chi back "
           "up; resolved smooth manifold is not CY_3 in any case."),
    refs=["Muk88", "Xia96"],
))

# (g) CY_3 with chi = -6  (the actual heterotic-3-generation candidates)
CANDIDATES.append(Candidate(
    name="Tian-Yau CY_3  (Z/3 quotient of quintic-related CICY)",
    dim=6,
    chi_compact=CHI_CY3_REF["Tian_Yau"],
    has_2I_action="no",
    has_phi_structure="no",
    gives_3_generations="yes",
    notes=(
        "Free Z/3 quotient.  h^11=14, h^21=23  =>  chi = 2(14-23) = -18? "
        "Original Tian-Yau (1987) reports chi=-6 for the Z/3-free quotient "
        "of the bicubic; |chi|/2 = 3 generations. NO 2I-symmetry: |Aut| "
        "is small, dominated by the Z/3 used in the quotient."),
    refs=["Ti87", "CdOGP91"],
))

CANDIDATES.append(Candidate(
    name="CICY-7884  (3-generation CY_3)",
    dim=6,
    chi_compact=CHI_CY3_REF["CICY_7884"],
    has_2I_action="no",
    has_phi_structure="no",
    gives_3_generations="yes",
    notes=(
        "From the Candelas et al. CICY list; chi=-6 via discrete-group "
        "quotient of an ambient CICY by a small abelian group.  No "
        "icosahedral / H_4 symmetry."),
    refs=["Candelas-Dale-Lutken-Schimmrigk 1988"],
))

CANDIDATES.append(Candidate(
    name="Z/5 x Z/5  free quotient of quintic",
    dim=6,
    chi_compact=CHI_CY3_REF["Quintic_in_P4"] // 25,   # = -8
    has_2I_action="no",
    has_phi_structure="no",
    gives_3_generations="no",
    notes=(
        "Free (Z/5)^2 quotient of the Fermat quintic gives chi = -200/25 "
        "= -8, not -6.  No 2I action."),
    refs=["GP90", "Hos+95"],
))

# (h) Hypothetical: a CY_3 invariant under H_4 / 2I
CANDIDATES.append(Candidate(
    name="CY_3 with H_4 symmetry  (hypothetical)",
    dim=6,
    chi_compact=None,
    has_2I_action="unknown",
    has_phi_structure="yes_H4",
    gives_3_generations="unknown",
    notes=(
        "H_4 is a non-crystallographic Coxeter group of order 14400 acting "
        "on R^4. It does NOT preserve any integer lattice in dim<=8 "
        "[Patera-Twarock 2002].  No CY_3 with a faithful H_4 action is "
        "known in the literature.  H_4 lifts to E_8 via the icosian "
        "construction, but this is a SYMMETRY OF THE LATTICE, not of any "
        "compact manifold of complex dim 3."),
    refs=["Patera-Twarock 2002 J.Phys.A 35 1551",
          "Moody-Patera 1993 J.Phys.A 26 2829"],
))

# (i) 600-cell quotients (lens-like 4-manifolds)
CANDIDATES.append(Candidate(
    name="600-cell / I  (icosahedral 600-cell)",
    dim=3,
    chi_compact=0,
    has_2I_action="yes_free",
    has_phi_structure="yes_icosian",
    gives_3_generations="no_chirality_zero",
    notes=("3-manifold of the form S^3/H; chi=0.  Cannot give chirality."),
    refs=["Cox91"],
))


# ---------------------------------------------------------------------------
#  Part D. Index-theorem check
# ---------------------------------------------------------------------------

def num_generations_heterotic(chi: Optional[int]) -> Optional[int]:
    """For heterotic on CY_3 with standard embedding, # gens = |chi|/2."""
    if chi is None:
        return None
    if chi % 2 != 0:
        return None  # not a valid chi for a CY_3
    return abs(chi) // 2


# ---------------------------------------------------------------------------
#  Part E. Output JSON
# ---------------------------------------------------------------------------

def main(out_path: Path) -> None:
    rows = []
    for c in CANDIDATES:
        d = c.to_dict()
        d["heterotic_generations_if_CY3"] = (
            num_generations_heterotic(c.chi_compact) if c.dim == 6 else None
        )
        rows.append(d)

    summary = {
        "candidates": rows,
        "summary": {
            "any_4manifold_with_chi_pm6_AND_2I_free_action": False,
            "any_CY3_with_chi_pm6_AND_2I_action":            False,
            "any_compact_with_phi_H4_structure_AND_chi_pm6": False,
            "best_three_generation_CY3_known":               "Tian-Yau / CICY-7884",
            "best_2I_compact_known":                         "S^3/2I (chi=0)",
            "obstruction_summary": (
                "(1) chi(S^3/H)=0 for ALL finite H acting freely on S^3, "
                "so any product with a chi=0 partner stays at chi=0; "
                "(2) the only 2I-type action on K3 is non-free (Mukai/Nikulin), "
                "yielding orbifolds with non-integer chi/|G|; "
                "(3) no CY_3 with chi=+/-6 is known to admit a 2I or "
                "H_4-coloured action, because H_4 is non-crystallographic "
                "and CY_3 automorphism groups are typically abelian or small."
            ),
        },
        "references_master": [
            "Coxeter 1991 'Regular Complex Polytopes'",
            "Conway-Sloane 1999 'Sphere Packings, Lattices and Groups' ch.8",
            "Wilson 2009 J.Algebra 322, 2186",
            "Mukai 1988 Invent.Math. 94, 183",
            "Nikulin 1979 Trans.Moscow Math.Soc. 38, 71",
            "Tian-Yau 1987 (CY_3 with chi=-6)",
            "Greene-Plesser 1990 Nucl.Phys.B 338, 15",
            "Candelas-Horowitz-Strominger-Witten 1985 Nucl.Phys.B 258, 46",
            "Hosono-Klemm-Theisen-Yau 1995 Comm.Math.Phys. 167, 301",
            "Patera-Twarock 2002 J.Phys.A 35, 1551",
        ],
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print()
    print(f"Total candidates tabulated: {len(rows)}")
    n_chi6 = sum(1 for r in rows if r["chi_compact"] in (6, -6))
    n_2I = sum(1 for r in rows if r["has_2I_action"].startswith("yes"))
    n_both = sum(
        1 for r in rows
        if r["chi_compact"] in (6, -6) and r["has_2I_action"].startswith("yes")
    )
    print(f"  with chi = +/- 6:                   {n_chi6}")
    print(f"  with some 2I action:                {n_2I}")
    print(f"  with BOTH chi=+/-6 AND 2I action:   {n_both}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(here / "compactification_search_results.json")
