# LITERATURE.md — the novelty ledger for the diffusion seed stencils

> **STATUS: LITERATURE AND NOVELTY PASS E5.2 (#43) RUN 2026-09-24 (P1–P6
> below).** The house ledger of the companion wave manuscript, rebuilt for
> diffusion. The claims the manuscript could make are itemised in §1, and
> each item sits in the bucket the record supports. §2 logs what was
> searched, with which tool, and what came back. §3 is the verification
> rule, §4 the standing hazards and §6 the wording the manuscript may use.
> §1d answers plan R1 and decides T3. `paper/references.bib` carries no
> unverified entry (P6).
>
> **Headline.** In one dimension the construction is classical in every
> part:
>
> - the seed chain is the formal-power representation of Sturm–Liouville
>   solutions (K2);
> - stencils of any order computed by integrating ODEs across the
>   coefficient are the exact and truncated schemes of the Samarskii
>   school (K3);
> - equilibrium exactness is Tikhonov–Samarskii's (K1).
>
> The manuscript claims nothing for the 1-D construction. What survived
> the pass as ours (§1b) is two-dimensional:
>
> - stencils on an unchanged scattered node set whose basis is continued
>   through a *given* sub-grid smooth edge by ODEs, in the normal
>   coordinate and in a curve's own coordinates;
> - fourth order at every δ, elliptic and parabolic;
> - the 2016 jump construction as the δ → 0 limit;
> - the head-to-head against the conductance rules and regularised media
>   of diffusion practice, as measured.
>
> The nearest relatives are FLAME (local solutions in difference stencils,
> jumps included) and the LOD line (Galerkin, high order through
> under-resolved coefficients). Each is named in the same paragraph as
> the claim. The ring's flux seeds meet an observation already made for
> elastic waves across imperfect contacts (K9).

**Maintenance rule** (house convention, mirrored from the companion's
ledger). Update this file on **any** new novelty claim, any new comparison
method, and any new regime measured (order, contrast, dimension). Before
anything is called unrecorded, run a directed refuter pass and log it in §2
(date, tool, query, result). Re-run the following before any external
circulation, including the arXiv upload (#52):

- P3's quasi-Trefftz and FLAME queries (both groups posted in 2025–26);
- P2's cited-by sweep of EABE 2017;
- P5's imperfect-interface queries.

**Cite, don't claim.** The manuscript cites §1a items as prior work. It
words §1b items as "no prior instance found", never as "novel" or "first".
Every "not found" below is a keyword-search negative tied to the queries
that produced it (§4).

Verification statuses (rule in §3):

- **[V]**: verified against a fetched primary source, named in §2 or in
  the entry's comment in `paper/references.bib`. Accepted sources are
  the Crossref API record of the DOI, the arXiv API record, a zbMATH
  Open or OpenAlex record, a publisher or repository page, or the PDF
  itself.
- **[S]**: seen in search results or a page summary only. It may be
  discussed here with hedging but may **not** enter the bib.

A metadata [V] does not certify content. Where a paper's *claims* were read
only at abstract level, or not at all, the entry says so.

## 1. The honest claim, itemised

### 1a. Cite, don't claim (trodden ground)

**K1. The constant-flux seed, exact face conductances, and 1-D equilibrium
exactness.** The first seed after the constant, `φ₁ = α_e ∫ dξ/α`, is:

- over one cell, the exact face coefficient `h / ∫ dξ/α` of Tikhonov &
  Samarskii's homogeneous schemes (USSR CMMP 1 (1962) 5–67 [V, metadata;
  content not read, §4]; Samarskii's *Theory of Difference Schemes*, CRC
  2001 [V, metadata]);
- for piecewise-constant α, the harmonic-mean interface conductivity of
  finite-volume heat conduction: Patankar 1980 [V, zbMATH Zbl
  0521.76003], §4.2-3 "The Interface Conductivity", pp. 44–47, eqs.
  4.9–4.10, pinned from the OCR of the 1980 edition (P4). That
  derivation is "based on the steady, no-source, one-dimensional
  situation".

The conservative three-point scheme on these coefficients, the stiff note's
T1-FV (§1.8 item 3), is Pan, Xu & Li's "harmonic average method" (IJNAM 23
(2026) [V; arXiv 2502.09413 read: their eq. 3 is exactly
`β_{i+½} = (h⁻¹ ∫ β⁻¹)⁻¹`]). They prove it second order in 1-D. They also
write that it "has only first order accuracy for general two- or
three-dimensional interface problems except the cases that the interface is
parallel" to an axis. Goode & Appel (USGS WRI 92-4124, 1992 [V; abstract
read on the USGS page]) state the distinction between the nodal and the
exact rule that the stiff note found for itself (§1.8, the correction for
E3.5):

- the harmonic mean "is the exact interblock transmissivity for
  steady-state one-dimensional flow with no recharge if the
  transmissivity is ... uniform over each finite-difference block";
- it "may be inferior to other means if transmissivity varies in a
  continuous or smooth manner between nodes".

The same one-cell equilibrium function is the special shape function of
Babuška–Osborn (SINUM 20 (1983) [V]) and Babuška–Caloz–Osborn (SINUM 31
(1994) [V]), the cell-problem basis of Hou–Wu (JCP 134 (1997) [V]) and the
harmonic coordinates of Owhadi–Zhang (CPAM 60 (2007) [V]).

*Framing:* the 1-D elliptic exactness of the seeds and of T1-FV (stiff
§1.8, §2.5 statement 4) is classical. The manuscript states it as a remark
citing Tikhonov–Samarskii, Patankar and Pan–Xu–Li, and T1-FV's second order
(§2.4) as Pan–Xu–Li's theorem observed. Neither is a finding.

**K2. The 1-D seed chain is the formal-power representation of
Sturm–Liouville solutions.** Kravchenko (Complex Var. Elliptic Equ. 53
(2008) [V; abstract read]) and Kravchenko & Porter (MMAS 33 (2010) [V;
Theorem 1 and Corollary 9 read in arXiv 0811.4488]) write the general
solution of `(p u′)′ + q u = λ r u` as power series in λ. The coefficients
are recursive integrals `X^(n)`, `X̃^(n)` from a point `x₀`, with the weights
`1/(u₀² p)` and `u₀² r` alternating. Their Corollary 9 credits the case
`q ≡ 0` to Weyl.

With `q = 0`, `u₀ = r = 1`, `p = α` and `x₀ = x_e`, the seeds of stiff §1.2
are `φ_k = k! α_e^⌈k/2⌉ X^(k)` for odd k and `k! α_e^⌈k/2⌉ X̃^(k)` for even
k. This is an identity, and the λ-series coefficients are the
time-polynomial profiles, with λ standing for ∂ₜ. The E5.2 pass checked it
against the repo's marcher to 12 significant digits for k ≤ 5 through the
MATLAB medium's δ = 0.01 edge. `tests/heat1d/test_stiff.py::
test_the_seeds_are_the_formal_powers_of_the_sturm_liouville_operator` pins
it at 1e-9.

Related work:

- For constant α the time-polynomial solutions are the heat polynomials
  (Rosenbloom & Widder, Trans. AMS 92 (1959) [V, metadata]; Widder, *The
  Heat Equation*, 1975 [V, zbMATH]).
- For `u_xx − q u = u_t` they are the transmuted heat polynomials of
  Kravchenko, Otero & Torba (Adv. Math. Phys. 2017 [V; abstract read]).
- Complete families for parabolic equations with analytic coefficients
  are Colton's (SIAM J. Math. Anal. 6 (1975) [V; abstract read; not
  entered]).

*Framing obligation:* §3 of the manuscript says that in 1-D the seeds are
these formal powers anchored at the evaluation node, with the citation. The
seeds' 2-D chain (straight and tangential, stiff §3.2, §3.10) is not a
Sturm–Liouville object and is not covered by K2.

**K3. Stencils of any order from ODEs integrated across the coefficient
(1-D).** Tikhonov & Samarskii's "truncated schemes of rank m" expand the
exact three-point scheme in powers of h with multiple-integral
coefficients. This is per the introduction of Makarov, Mayko & Ryabichev,
Ukr. Mat. Zh. 75 (2023) [V], read in full by P3; the 1962 papers themselves
were not read.

Samarskii & Makarov (Sov. Math. Dokl. 41 (1990) [V, zbMATH Zbl 0717.65052;
the review read]) "construct a three-point difference scheme of an
arbitrary order of accuracy" "in the class of piecewise smooth functions
k(x), q(x) and f(x)". To define it at a node, "it is necessary to solve four
auxiliary Cauchy problems", each "in one step by a one-step method".
Gavrilyuk, Hermann, Makarov & Kutniv (ISNM 159, Birkhäuser 2011 [V,
Crossref + zbMATH Zbl 1226.65066; the review read]) combine the exact
schemes "with the modern IVP-solvers".

Other stencils exact on a local non-polynomial space:

- Exact schemes: Vizvari et al. (Adv. Differ. Equ. 2020:497 [V]).
- HODIE: Lynch & Rice (Math. Comp. 34 (1980) [V; abstract read]), whose
  coefficients are "obtained 'locally' by solving a small linear system
  for each group of stencil points in order to make the approximation
  exact on a linear space S".
- L-splines: Schultz–Varga (Numer. Math. 10 (1967) [V]).
- Fitted operators: Allen–Southwell 1955, Il'in 1969, Scharfetter–Gummel
  1969 and the survey of Roos–Stynes–Tobiska 2008 [all V]. These are for
  a small diffusion parameter. P3's QB4 found no fitted-operator paper
  for an interior layer in α.

At a jump, three-point schemes "of any prescribed order of accuracy" for
`−(p u′)′ + q u = f` with discontinuous p are Gartland's (IMA JNA 9 (1989)
[V; abstract read]). Taylor data carried across a step to make the weights
is Sujecki's, in photonics (Opt. Lett. 35 (2010) [V; abstract read]). That
is the 2016 translated basis in 1-D, found by a different community (P1,
finding 3).

*Framing:* high-order 1-D stencils made by integrating ODEs through a
piecewise-smooth coefficient, exact at equilibrium, with the jump as a
special case, are all prior. The five-point FD4 stencil exact on the
anchored formal powers, applied to the parabolic problem, is a variant of
them. The manuscript claims nothing for the 1-D construction and presents
§3 as the specialisation of the companion's construction, whose ingredients
these are.

**K4. Local solutions of the equation in difference stencils (2-D).**

- **FLAME** (Tsukerman, JCP 211 (2006) [V, metadata] and JCP 229 (2010)
  [V; arXiv 0906.1388 abstract and introduction read]) "replaces the
  Taylor polynomials with much more accurate 'Trefftz' approximations by
  local solutions of the underlying differential equation":
  - in 2-D and 3-D;
  - with basis functions that "satisfy the underlying differential
    equation, along with the interface boundary conditions";
  - since 2010, on irregular stencils by least squares (a modification
    it credits to Pinheiro & Webb);
  - for piecewise-constant media (particles, photonic crystals) in
    electro- and magnetostatics and wave scattering, all steady.
- **The tailored finite point method** (Huang, NHM 4 (2009) [V; abstract
  read]) is exact in 1-D "when the coefficients are piecewise linear
  functions".
- **Its multiscale form** (Han & Zhang, CMS 10 (2012) [V; full text read by
  P3]) builds a five-point 2-D stencil from local cell problems solved
  numerically, at second order.
- **The multiscale finite-volume method** (Jenny, Lee & Tchelepi, JCP 187
  (2003) [V; abstract read]) builds "transmissibilities that capture the
  local properties of the differential operator".

*Framing obligation:* local solutions in difference stencils, interfaces
and irregular stencils included, are FLAME's. None of these continues a
basis through a smooth sub-grid layer, and none is parabolic or of fourth
order through an unresolved coefficient. The manuscript names FLAME in the
same paragraph as O1.

**K5. Galerkin bases from local solves, and high order through
under-resolved coefficients.**

- Babuška–Caloz–Osborn 1994 [V; abstract read]: special shape functions
  for "unidirectional composites", with "the same accuracy as usual
  methods have for problems with smooth coefficients". These are the
  closest in spirit to the flat seeds.
- Hou–Wu 1997.
- Chu, Graham & Hou (Math. Comp. 79 (2010) [V; abstract read]): "subgrid
  problems for the basis functions ... without resolving the interfaces",
  sharp interfaces, O(h) in energy.
- Owhadi–Zhang 2007 (harmonic coordinates).
- The localized orthogonal decomposition (LOD): Målqvist & Peterseim
  (Math. Comp. 83 (2014) [V; abstract read]).
- Two papers reach high order with no regularity assumption on the
  coefficient:
  - Maier (SINUM 59 (2021) [V; abstract read]): "high-order convergence
    rates ... does not rely on additional regularity assumptions on the
    domain, the diffusion coefficient";
  - Hauck & Peterseim (Math. Comp. 92 (2023) [V; abstract read]):
    "uniform algebraic approximation rates ... even for under-resolved
    rough coefficients".

*Framing obligation:* high-order approximation through a coefficient the
mesh does not resolve exists, in Galerkin form with local fine-scale
solves. The claim of O1 is scoped to stencils on an unchanged node set,
with a basis continued through a given edge. It may not read as "first
high-order method through sub-grid coefficients".

**K6. Trefftz and quasi-Trefftz spaces.** The polynomial quasi-Trefftz
spaces build local solutions "in the sense of Taylor polynomials":

- elliptic: Imbert-Gérard, Moiola, Perinati & Stocker, IMA JNA 45 (2025)
  [V; abstract read];
- general scalar linear equations "with smooth variable coefficients":
  Imbert-Gérard, arXiv:2505.18480 [V as preprint];
- the heat equation `∂ₜu − ∇·(κ∇u) = 0`, in a space–time LDG: Gómez,
  Perinati & Stocker, JSC 106 (2026) [V; abstract and sect. 5.3 read by
  P3].

They are all DG, with coefficients smooth within each element (P3 QB2). The
companion (Martin2026seeds, own) shows why the Taylor route stops at a
radius of order δ.

*Framing:* as in the companion's K3. The seeds are what a Taylor-built
local solution cannot be through a sub-grid edge. None of the 16
quasi-Trefftz or 55 Trefftz-titled arXiv records (P3) treats under-resolved
or steep coefficients.

**K7. Chebyshev systems, and a local truncation error one order short.**
Well-posedness is the companion's argument (own), resting on Pólya 1922,
Karlin–Studden 1966, Coppel 1971 and Mühlbach 1973 [all V, as re-fetched in
P6].

The stiff note's "third order locally on the seeded rows, fourth globally"
(§1.6) is an instance of what Beale & Layton (CAMCoS 1 (2006) [V; abstract
read]) prove for interface corrections: "the solution can be computed with
uniform O(h²) accuracy even if the truncation error is O(h) at the
interface". *Framing:* cite them at §3's truncation remark. The seeded
rows' lost order is the same phenomenon, and the manuscript does not claim
it.

**K8. Interface methods for a jump, elliptic and parabolic.**

- Classical:
  - Babuška 1970 (FEM with discontinuous coefficients);
  - the immersed interface method (LeVeque & Li 1994; Li & Mayo 1994, ADI
    for the heat equation; the Li–Ito SIAM monograph 2006; Wiegmann &
    Bube 2000);
  - ghost fluid / boundary condition capturing (Liu, Fedkiw & Kang 2000);
  - matched interface and boundary (Zhou, Zhao, Feig & Wei 2006);
  - parabolic FEM (Chen & Zou 1998);
  - unfitted Nitsche / CutFEM (Hansbo & Hansbo 2002; Burman et al. 2015);
  - a matched ADI for parabolic interfaces (Li & Zhao 2017).
- Fourth order and above, 2016–2026:
  - correction functions (Marques, Nave & Rosales 2017);
  - augmented MIB (Feng & Zhao 2020);
  - kernel-free boundary integrals (Xie, Li & Ying 2023);
  - Cartesian finite volumes to sixth order (Thacher, Johansen & Martin
    2023);
  - hybrid FD to sixth order (Feng, Han & Minev 2024).

  All [V]; abstracts read for the post-2016 ones (P2).
- RBF, meshfree and particle methods at material interfaces:
  - own: the dissertation; EABE 2017;
  - Divo & Kassab 2007 (localized RBF, conjugate heat transfer);
  - Ahmad & Siraj-ul-Islam 2018 (meshless, parabolic interfaces);
  - Bartwal et al. 2023 (clouds restricted to one side);
  - Kraus, Kuhnert, Meister & Suchde 2023 (GFDM);
  - Cheng, Ju & Zhang, JSC 108 (2026) (high-order RBF-FD, anisotropic
    elliptic interfaces; [V, metadata only]).

  All [V].
- The 38 works OpenAlex lists as citing EABE 2017 (P2): every
  interface-relevant one handles a sharp interface.
- The plan's "Reutskiy, Bayona et al." are EABE's refs [20]–[21]: RBF-FD
  for elliptic problems with hard boundaries or inhomogeneous media, not
  interface methods (P1, finding 2).

*Framing:* all take the jump as given. Pan–Xu–Li note the harmonic mean's
first order at a general interface in 2-D and 3-D (K1).

**K9. Thin resistive layers and imperfect interfaces.** A layer whose
conductivity scales with its thickness tends to a resistive interface
`[u] = R q` with continuous flux, which is EABE eq. 40's ring, where
`s w = 1`:

- Sanchez-Palencia, J. Math. Pures Appl. 53 (1974), "... couches minces de
  grande résistivité" [V, zbMATH; title only].
- Perrussel & Poignard, AMC 221 (2013) [V; INRIA RR-7163 read in part by
  P5]: "the thin layer is highly insulating in the sense that
  σm = δξSm", with the order-0 condition `ξSm [u₀] = σe ∂ₙu₀`.
- Hashin, J. Appl. Phys. 89 (2001) [V; abstract read]: "imperfect
  interface conditions involve interface discontinuities of potential and
  normal component of flux".
- The name: Swartz & Pohl's review, RMP 61 (1989) [V; abstract read]. Its
  Kapitza resistance is interfacial phonon physics, not a thin layer, so
  the manuscript says "a contact resistance" and cites the review only
  for the term.

Strong-form schemes for such contacts report their order:

- Lombard & Piraux 2003 and 2006 (SISC [V]);
- Cao & Yuan, AIMS Math. 8 (2023) [V; full text read by P5]: fourth order
  in 1-D;
- Leguèbe, Poignard & Weynans 2015, second order;
- Guittet et al. 2015 (VIM), second order.

*The ring's flux seeds (plan R8, stiff §3.11) meet an observation already
made for elastic waves.* Lombard & Piraux 2006 found the loss for 2-D
elastic waves across spring-mass contacts. arXiv physics/0508018v2 §4.7 and
§5.2 were read, and the quotes checked against it:

- "we need k = 3 to ensure second-order accuracy";
- "for k = 2, the convergence orders fall below 1.4. (we recall that k = 2
  suffices to ensure second-order accuracy for perfect contacts in 2-D
  contexts ..., and for imperfect contacts in 1-D contexts ...)".

Their 2003 paper shows the mechanism in 1-D: the far side's matrix entries
lose a power of Δx under spring-mass conditions (eqs. 4.25–4.26, HAL
hal-00004812, read by P5), without an order lost there. In Chiavassa &
Lombard (CiCP 13 (2013) [V from the arXiv journal_ref; not entered]) no
order is lost, because the jump is proportional to a primary unknown.

*Framing obligation:* the manuscript cites Lombard–Piraux 2003/2006 where
it introduces the five degree-5 flux seeds. It may say they are the scalar
diffusion, scattered-node form of that observation, with the remedy of
adding only the flux-carrying functions of one degree higher. It may not
present "the flux is needed one degree higher" as a new general fact.

**K10. The standing alternative: change the coefficient, keep the scheme.**

- Harmonic or arithmetic cell means:
  - Patankar 1980;
  - Ewing, Iliev & Lazarov (SISC 23 (2001) [V; abstract read]): "the
    known schemes which use arithmetic and harmonic averaging of the
    discontinuous diffusion coefficient";
  - Goode & Appel 1992;
  - Pan–Xu–Li 2026.
- Special averaging for layers finer than the grid:
  - Moskow, Druskin, Habashy, Lee & Davydycheva (SINUM 36 (1999) [V;
    abstract read]): a "special cell averaging [that] ... does not
    require the grid to be small compared to the layering";
  - Davydycheva, Druskin & Habashy, Geophysics 68 (2003) [V];
  - used in marine CSEM by Mittet (Geophysics 75 (2010) [V; full text
    read by P4]).
- Regularised coefficients:
  - Tornberg & Engquist (JSC 19 (2003) [V, metadata]);
  - the same authors' MAA 13 (2006) paper [V; first page read], which
    reports that "the modified higher order discretization yields a
    second order error component originating from the discontinuities,
    and a fourth order error from the smooth regions". This is the wave
    precedent for the stiff note's finding that nodal treatments cap the
    fourth-order operator at second order once the edge is resolved
    (§2.5 statement 3).
- On node clouds:
  - Kraus et al. 2023 (full text of arXiv 2204.05191 read by P2) "smooth
    the diffusivity η by using weights s_ij = exp(−3‖x_j − x_i‖²/h_i²)
    in a convex combination ... over the neighborhood", with harmonic
    means at the Voronoi faces;
  - Kraus, Kuhnert & Suchde (CAMWA 194 (2025) [V; full text read by P3])
    use pairwise Pythagorean means and see "first-order convergence" at
    an interface;
  - SPH takes the pairwise harmonic mean (Cleary & Monaghan, JCP 148
    (1999) [V, metadata], as quoted from Monaghan's 2005 review, §7.1,
    read by P2).

*(i) The 2-D comparator's source.* No instance was found of a harmonic or
arithmetic mean of α over a disc of radius h/2 or h about each node (P2
QA4), the stiff note's E4.9 rule. The manuscript calls it "the natural
scattered-node analogue of the harmonic-mean conductance". It cites
Patankar for the 1-D rule, and Kraus et al. 2023 and Cleary & Monaghan for
averaging α over a node cloud or pair. No nodal harmonic mean under a
high-order non-conservative `Dx A Dx` was found either (P4 QC2); the
nearest is the volume harmonic averaging of seismic staggered grids (the
companion's K1). The 1-D T1 is called the like-for-like changed-medium
comparator and cites nothing, as the E3.5 breadcrumb on #43 proposed.

*(ii) T3, the band-limited coefficient, was not found in use for diffusion*
(P4 QC4):

- **Waves only.** The seismic line band-limits density and compliance for
  waves: Mittet, Geophysics 82 (2017) [V; abstract read]; Koene, Wittsten
  & Robertsson, GJI 229 (2022) [V; arXiv 2104.08206 read in full].
- **Proposed, not demonstrated.** Mittet's 2017 abstract and his SEG 2018
  abstract propose it for Maxwell's equations and marine CSEM, with no EM
  example.
- **Mentioned, not adopted.** libEMM (Yang, CPC 288 (2023) [V; full text
  of arXiv 2304.00233 read]) says the band-limited step function "can be
  very accurate also" and that it "is not adopted in libEMM".
- **Elsewhere.** No heat-conduction, Darcy or FFT-homogenisation use of a
  spectrally band-limited coefficient turned up.

Decision in §1d.

*(iii) The head-to-head was run* (stiff note §2.4–2.5 and §4.9, statement
10 of §5.3).

1-D, with the edge unresolved (`h ≥ 4δ`, δ = 0 included), in ‖e‖₂/‖u‖₂,
the ranking is seeds < T1-FV < T1 (two cells) < naive ≳ T1 (one cell) ≈
T2 < T0 (m = 1) < T0 (m = 2). At 200 nodes and δ = 0.0025 on the MATLAB
ramp problem the errors are:

| seeds | T1-FV | T1, two cells | naive / T1, one cell / T2 | T0, m = 1 | T0, m = 2 |
| --- | --- | --- | --- | --- | --- |
| 1.6e-9 | 8.0e-6 | 8.9e-5 | 7.9e-4 / 4.0e-4 / 4.6e-4 | 2.2e-3 | 5.0e-3 |

- T1-FV is second order at every δ.
- The nodal treatments cap the scheme at second order once `h ≲ δ/4`.
- On eq. 75, T1-FV is ahead of the seeds at δ = 0 below 801 nodes and at
  δ = 0.0025 at 101 nodes, where the seeds' sinusoid line is still
  pre-asymptotic.

2-D, on scattered nodes, the treatments are the harmonic and arithmetic
disc means at radii h/2 and h, and T0 with m = 1, 2:

- none beats plain sampling by more than 1.75× in the RMS norm (in the max
  norm up to 3.1× on the coarsest sets, 1.55× from 5000 nodes on);
- the radius-h means are *worse* than sampling at the jump from 5000 nodes
  on, the harmonic one 1.6–3.4× and the arithmetic one 1.3–3.0× (#49: this
  had read 1.6–3.4× for both, the notes' figure before E5.3's number check);
- the seeds' lead over the best of them grows from 2.3 to 4.7 orders over
  1250–40,000 nodes.

No conservative scattered-node scheme was built.

Scope: one scheme per dimension, one contrast per case (9 : 1 in 1-D, 5 : 1
on cases 1–2), one tanh profile, our implementation of each treatment. §6b
words what that licenses.

### 1b. Ours as scoped (survived the pass; framing obligations noted)

**O1. Seeds for the diffusion operator on an unchanged scattered node set,
continued through a given sub-grid smooth edge by ODEs.** The construction:

- RBF-FD rows (Gaussians plus seeds, 30 nodes, degree 4, inside a 42 / 5
  bulk);
- the polynomial part replaced by the t = 0 profiles of time-polynomial
  solutions of `u_t = ∇·(α∇u)`;
- the profiles marched as ODE chains in the normal coordinate of a
  straight feature (stiff §3.2–3.4) and in a curve's own coordinates
  where α or the edge varies along it (the tangential chain, §3.10, O2);
- every row within 20 δ of an edge seeded, with no δ threshold and no
  switch-off.

What it measured:

- fourth order at every δ, elliptic and parabolic: fits 4.2–4.5 over
  1250–40,000 nodes at δ = 0.04 … 0.0025, with the widths within
  1.2–2.5× of one another (§5.3 statement 4);
- the 2016 jump construction (E2.3's rows) reproduced at δ = 0: spans to
  4.9e-14, rows to 7.6e-12 (statement 4).

**No prior instance found** (P2: the QA1 sweep and QA3 cited-by; P3: QB1,
QB3; P4: QC5). Nearest relatives, each different in mechanism:

- FLAME (K4): local analytic solutions in piecewise-constant media,
  steady, no smooth sub-grid layer;
- the multiscale TFPM (K4): local cell problems, second order;
- Babuška–Caloz–Osborn and the LOD line (K5): Galerkin;
- quasi-Trefftz (K6): Taylor, DG;
- the companion (own): the wave equation, with route (a) on curved edges.

*Obligations:*

- name K3–K6 in the same paragraph;
- word the claim as "no prior instance found", with this ledger as the
  reference for the search;
- say that the 1-D form is classical (K2, K3);
- say that the construction is the companion's, specialised to one field
  (stiff §1.2).

**O2. The tangential chain.** The seeds of a curved or tangentially varying
edge are continued in the foot curve's own coordinates. The variation of α
and of the metric along the curve is carried by 75 coupled levels (stiff
§3.10, §4.7). This is fourth order where a frozen normal profile, route (a),
the companion's curved-edge treatment, is O(1)-inconsistent (§4.6). No
search targeted it beyond O1's queries. It is claimed as part of O1's
construction, not separately.

**O3. The thin resistive ring with seeds.** On EABE eq. 40's ring:

- the seeds are exact on the matched radial profile to rounding at every
  s (1.3–2.1e-15 from s = 10³ to 10¹¹);
- their per-stencil blocks condition independently of s, where E2.3's
  continuity construction loses digits like s (§5.3 statement 9);
- Fig. 19's twin is fourth order at every s to 40,000 nodes;
- through a smooth ring whose width and edges are both below h, the seeds
  are fourth order on the far field.

The flux seeds carry K9's framing obligation. EABE's own ring (2017) already
treated a layer thinner than the node spacing as two jumps (P1, finding 1),
so the claim is the *smooth* ring and the conditioning, not a sub-grid
layer as such.

**O4. The measurements.**

- The δ-parametrised knee of the naive product through a smooth edge:
  - in 1-D, 100–200× on the MATLAB problem and 800–1000× on eq. 75, at
    h ≈ δ (§2.5 statement 1);
  - on scattered nodes, 34–38× deep, over `4δ ≳ h ≳ 0.75δ` (§5.3
    statement 2).
- The δ = 0 construction (the 2016 method):
  - it sits on the `c δ` floor, `c = (a − b) ln(a/b)/(2ab)`, while
    `h ≳ 2δ`;
  - it grows to O(1) once the grid resolves the edge, so using it
    requires knowing δ (statements 1 and 3).

The loss of order of standard schemes at a jump is documented (K8, K10).
The knee and the construction's floor, as functions of δ/h, are ours as
scoped: one contrast per case, one tanh profile, node set 0 with fixed rows
beside each curve, ≤ 40,000 nodes for the sweeps (§5.5).

**O5. Remarks.** The following are methodological remarks. No claim rests
on them, and none was searched:

- the 1-D elliptic degeneracy (K1's classical exactness, stated as a
  remark, plan R2);
- the flux on the pair beside the edge as a sharper indicator than the
  RMS (§5.3 statement 2);
- E4.12's diagonal rule for the ring's tail rows (§4.10, a proxy).

### 1c. Unswept (no claim may cite this ledger for support)

- **Own exclusions, not literature gaps (plan §3, stiff §5.5):** three
  dimensions; corners and triple junctions (E2.10); anisotropic (tensor)
  α; α depending on u; time-dependent media; edges meeting a Dirichlet
  boundary; profiles other than tanh; contrasts other than those run.
- **Russian- and Ukrainian-language literature beyond the zbMATH reviews
  and one full text** (Makarov et al. 2023): the Russian originals of
  Tikhonov–Samarskii (1961) and Samarskii–Makarov (Differ. Uravn. 26
  (1990), Zbl 0711.65060); Samarskii, Lazarov & Makarov's book on schemes
  for generalized solutions (1987/1989); Makarov, Makarov & Prikazchikov
  (1979, [S]).
- **Neutron-diffusion "analytic nodal" / "analytic coarse-mesh finite
  difference" methods:** García-Herranz et al., Nucl. Sci. Eng. 144
  (2003), [V, abstract read by P3, not entered]. Only this one paper was
  seen; the line was not swept.
- **Reservoir and groundwater upscaling of thin barriers** (transmissibility
  of faults and shale layers) beyond Goode–Appel, the multiscale FV and
  the EM averaging of Moskow/Davydycheva.
- **Discontinuous-coefficient heat conduction in DG, lattice-Boltzmann and
  peridynamics**, and computational electromagnetics beyond FLAME and CSEM
  (conformal FDTD, thin-sheet FDTD).
- **The boundary analogue of K9** (a Robin or Neumann closure needing one
  more order than a Dirichlet one): Svärd & Nordström, JCP 218 (2006), and
  Papac, Gibou & Ratsch, JCP 229 (2010), were seen as metadata only (P5).
- **Chinese-language literature;** Scholar-style citation graphs beyond
  OpenAlex's cited-by for EABE 2017.

### 1d. Decisions of this pass: R1 answered, T3 decided

**R1 (plan §5), "novelty is thinner than in the wave case": confirmed, and
thinner than the plan thought in 1-D.** The plan's defensible claim was the
combination: fourth order and higher on scattered nodes in 2-D through a
sub-grid smooth layer, elliptic and parabolic, with the jump construction
as its δ → 0 limit, measured against the harmonic-mean practice. It
survives, scoped as O1–O4.

- **In 1-D there is no construction claim.** The chain is Kravchenko's
  formal powers (K2). High-order three-point stencils from Cauchy problems
  through a piecewise-smooth coefficient are Samarskii–Makarov's and
  Gavrilyuk et al.'s (K3). Exactness at equilibrium is Tikhonov–Samarskii's
  (K1), and T1-FV's second order is Pan–Xu–Li's theorem. The 1-D sections
  of the manuscript derive the construction (the specialisation of the
  companion's) and carry measurements, O4 and K10 (iii), not novelty.
- **In 2-D the claim is O1–O3, scoped to stencils on an unchanged node
  set.** High order through under-resolved coefficients exists in Galerkin
  form (K5, the LOD line), and local solutions in difference stencils exist
  for piecewise-constant media (K4, FLAME). What the pass did not find is a
  stencil basis continued through a *smooth* edge thinner than the node
  spacing, parabolic or elliptic, reducing to a jump construction as
  δ → 0.
- **The ring's flux seeds are not a new general fact** (K9, Lombard–Piraux).
- **The comparison is against diffusion's own practice** (K10), the
  conductance rules included. It licenses no statement beyond the
  implementations run (§6b).

**T3 (plan §3.4): dropped, not built.** The pass found the band-limited
coefficient in use for waves only. It is proposed, but not demonstrated,
for the diffusive Maxwell equations of CSEM, where the codes found average
the conductivity instead (K10 (ii)). Plan §3.4 keeps T3 "if the literature
pass finds it used, otherwise drop it and say so". The manuscript's
comparator paragraph says so in one sentence, citing Mittet 2017 and Koene
et al. 2022.

A hazard for anyone who builds it later (P4, an inference, not measured):
the seismic analogue band-limits the parameter that multiplies the field in
the first-order system. For `∇·(α∇u)` that is 1/α, not α, and its Gibbs
overshoot can make 1/α negative at a large enough contrast.

## 2. Refuter-pass log

All fetches 2026-09-24. Tools:

- the Crossref REST API (`api.crossref.org/works/<doi>`,
  `query.bibliographic`);
- the arXiv API (`export.arxiv.org/api/query`);
- OpenAlex (`api.openalex.org`: search, `title_and_abstract.search`,
  `cites:` filters, and abstracts rebuilt from `abstract_inverted_index`);
- the zbMATH Open API;
- archive.org (advancedsearch and full-text search);
- the Open Library, HAL and DataCite APIs;
- Claude Code's WebSearch and WebFetch;
- `curl` and `pdftotext` on the PDFs in `papers/` and on the open-access
  PDFs fetched during the pass.

Four searches (P2–P5) ran in parallel as subagents with the same brief and
rules: the queries verbatim, every [V] from a record fetched in the pass,
and the content level stated. The lead read P1 and ran P6, and re-checked
every load-bearing quote below against a record or text it fetched itself.
Query strings are verbatim. arXiv and OpenAlex filter counts are the APIs'
result counts; Crossref's `query.bibliographic` totals are relevance-ranked
and meaningless, so for those the top 3–15 hits were read.

### P1 — own work

- **Read:**
  - the dissertation's title page, ch. 1 §1.2 (the heat-interface history,
    PDF 10–16), ch. 4's opening (PDF 75–76), ch. 6 (PDF 123–124) and its
    47-entry bibliography (PDF 125–128);
  - EABE 2017's abstract and introduction (PDF 1–4) and its 29 references
    (PDF 47–49).
- **Finding 1.** The 2016–2017 heat work:
  - handles smoothly variable α on either side of a jump by Taylor
    multiplication matrices (dissertation §4.1.1, EABE eq. 14–15);
  - treats "a region between two interfaces ... much smaller than the
    distance between adjacent discrete data nodes" as two jumps (EABE
    abstract: "thin, nearly insulating layers").

  Neither treats an edge that is smooth but thinner than the spacing. The
  seeds appear in no own work before the companion (2026).
- **Finding 2.** The 2016 map of heat-interface methods is:
  - FEM: Babuška 1970; Bramble–King 1996; Chen–Zou 1998; Lin–Yang–Zhang
    2015;
  - IIM: LeVeque–Li 1994; Li–Mayo 1994; Li–Shen 1999; Wiegmann–Bube 2000;
    Linnick–Fasel 2005; Ito–Li–Kyei 2005;
  - MIB Galerkin: Xia–Zhan–Wei 2014;
  - difference potentials: Epshteyn–Medvinsky 2015;
  - HOC: Mittal et al. 2016;
  - Beale–Layton 2006 (the dissertation's ref. [42]);
  - meshless: Yu–Chen 2011 (radial point interpolation with material
    interface conditions); Wang–Qin–Kang 2005; Reutskiy 2016; Ooi et al.
    2015.

  EABE's refs [20]–[21], Reutskiy 2016 and Bayona et al. 2017, are cited
  there for "challenging boundary conditions in elliptic problems", which
  is the plan's "Reutskiy, Bayona et al." bucket. None of the exact-scheme
  (K3), formal-power (K2), FLAME or TFPM (K4) lines is cited in either
  document.
- **Finding 3.** The dissertation's ch. 6 says "to the best of our
  knowledge, it is the only current method that makes use of a single
  piecewise polynomial basis near one or more curved interface(s), where
  polynomial data is explicitly 'translated' into other polynomial data as
  each curved interface is crossed". This pass found the 1-D form of that
  translation in Sujecki 2010 (photonics) and Gartland 1989. It found local
  interface-satisfying bases in FD stencils in FLAME (2006). None of them
  uses curved interfaces, scattered nodes and the parabolic problem
  together, so the 2016 claim's scope holds. Written today, though, it
  would have to cite them.
- **Crossref records** for the 2016 map's DOIs were fetched (Babuška 1970,
  Bramble–King 1996, Chen–Zou 1998, Wiegmann–Bube 2000, Beale–Layton 2006,
  Yu–Chen 2011, Reutskiy 2016, Ito–Li–Kyei 2005, Xia–Zhan–Wei 2014, Peskin
  1972, Mayo 1984), all matching the dissertation's citations.

### P2 — elliptic and parabolic interface methods, 2016–2026; RBF and meshfree

Cited-by: OpenAlex `filter=cites:W2604169584` (EABE 2017) returned **38**.
The interface-relevant ones are:

- Asif et al. 2026; Tóth 2026 (Kapitza-type imperfect interfaces, [S]);
- Krowiak–Podgórski 2024 ×2; Krowiak–Filipowska 2019;
- Bartwal et al. 2023;
- Gholampour et al. 2020–2022;
- Davydov–Safarpoor 2021;
- Ahmad–Siraj-ul-Islam–Larsson 2020; Ahmad–Siraj-ul-Islam 2018.

All handle sharp interfaces: domain decomposition, one-sided clouds,
modified bases, jump conditions. Cheng–Ju–Zhang 2026 does not appear in the
list (OpenAlex may lack its references).

QA1, the vocabulary sweep for a sub-grid smooth transition:

- arXiv:
  - `abs:"under-resolved interface"` 2;
  - `abs:"unresolved interface" AND abs:diffusion` 0;
  - `abs:"sub-grid" AND abs:"interface" AND abs:(elliptic OR diffusion OR Poisson)` 2;
  - `abs:"subgrid" AND abs:"discontinuous coefficient"` 0;
  - `abs:"thin transition layer" AND abs:(diffusion OR conduction OR elliptic)` 2;
  - `abs:"smooth interface" AND abs:"heat conduction"` 0;
  - `abs:"steep gradient" AND abs:coefficient AND abs:(elliptic OR diffusion) AND abs:"finite difference"` 0;
  - `abs:"graded interface" AND abs:(diffusion OR conduction OR elliptic)` 10 (physics);
  - `abs:"sub-cell" AND abs:"interface" AND abs:"elliptic"` 1;
  - `abs:"thin layer" AND abs:"heat conduction" AND abs:"finite difference"` 0;
  - `abs:"regularized" AND abs:"discontinuous coefficients" AND abs:"interface width"` 0.
- OpenAlex `title_and_abstract.search`:
  - `"under-resolved interface" AND (elliptic OR diffusion OR Poisson OR "heat conduction")` 3 (phase field);
  - `"subgrid interface" OR "sub-grid interface"` 23 (Huh & Sethian 2008, a sharp interface);
  - `"smooth interface" AND ("discontinuous coefficient" OR "variable coefficient") AND (elliptic OR Poisson)` 8 ("smooth" describes the curve);
  - `"steep coefficient" OR "steep gradient coefficient" OR "rapidly varying coefficient"` 94 (homogenization);
  - `("thinner than the mesh" OR "thinner than the grid" OR "smaller than the mesh size" OR "smaller than the grid spacing" OR "below the grid resolution") AND (conductivity OR "diffusion coefficient" OR "material property")` 9, none;
  - `("interface of finite thickness" OR "finite-thickness interface" OR "thick interface") AND ("heat conduction" OR elliptic OR diffusion) AND numerical` 4, none;
  - `("graded interlayer" OR "functionally graded interlayer" OR "graded interphase" OR "graded interface") AND ("heat conduction" OR thermal OR diffusion) AND (meshless OR "finite difference" OR "boundary element" OR "radial basis" OR "finite element")` 30 (materials engineering);
  - `"diffuse interface" AND "interface problem" AND (elliptic OR Poisson OR "heat equation")` 4 (Schlottbom 2016, a domain boundary).
- Crossref, the LOD line: `Maier high-order approach elliptic multiscale problems general unstructured coefficients` found Maier 2021 and Dong–Hauck–Maier 2023; `Malqvist Peterseim localization of elliptic multiscale problems` found Hauck–Peterseim 2023.

QA2 was answered by title lookups (Crossref and arXiv) for the classics and
the post-2016 high-order methods of K8. Corrections found: MIB 2006's third
author is Feig, the correction-function paper's third author is Rosales, and
the kernel-free boundary integral method starts with Ying & Henriquez 2007.

QA3, RBF and meshfree interface work:

- arXiv:
  - `abs:"RBF-FD" AND abs:interface` 1;
  - `abs:"radial basis" AND abs:"interface problems"` 0;
  - `abs:meshless AND abs:"interface problems" AND abs:elliptic` 0;
  - `abs:"heat conduction" AND abs:"radial basis" AND abs:interface` 0;
  - `abs:"RBF-FD" AND abs:"elliptic interface"` 1;
  - `abs:"radial basis" AND abs:"interior layer"` 0;
  - `abs:"generalized finite difference" AND abs:"discontinuous" AND abs:"diffusion"` 2 (Kraus et al.).
- OpenAlex author filters: Reutskiy, 84 works, no layered-media paper; Bayona, 28 works, no interface-coefficient work.

QA4, averaging α on nodes: OpenAlex `title_and_abstract.search`

- `(meshless OR meshfree OR "radial basis" OR "point cloud" OR "generalized finite difference") AND ("harmonic mean" OR "harmonic average" OR "harmonic averaging")` 30 (Zhan et al. 2022; the rest lidar);
- `"harmonic mean" AND "smoothed particle hydrodynamics" AND conduction` 3 (Ng et al. 2019);
- `("Voronoi" AND "harmonic mean" AND (conductivity OR "diffusion coefficient") AND (interface OR discontinuous))` 0.

The full texts of arXiv 2204.05191, 2305.01320, 2112.03123 and 2001.01597
and Monaghan's 2005 review were grepped.

**Verdict.**

- No FD, RBF-FD, meshfree or FV scheme was found that treats a smooth
  transition narrower than the spacing as its own object and aims at high
  order through it.
- High order through under-resolved coefficients exists in Galerkin form
  (LOD; K5).
- Every post-2017 RBF or meshfree interface paper found takes a sharp
  interface.
- The disc means have no source (K10 (i)).
- Three OpenAlex queries failed (HTTP 429/503).

### P3 — operator-adapted bases, exact and fitted schemes, Trefftz for the heat equation

- **Crossref** `query.bibliographic`, the top hits read:
  - `Exact and Truncated Difference Schemes for Boundary Value ODEs` (the book and its chapters);
  - `Makarov Gavrilyuk Kutniv three-point difference schemes arbitrary order`;
  - `tailored finite point method interface problem`;
  - `A class of difference schemes with flexible local approximation`;
  - `Trefftz difference schemes flexible local approximation heterogeneous media`;
  - `Kravchenko A representation for solutions of the Sturm-Liouville equation`;
  - `space-time Trefftz discontinuous Galerkin heat equation` (found Gómez–Perinati–Stocker 2026);
  - `multiscale finite difference method heterogeneous coefficients local solutions stencil`;
  - `RBF-FD augmented with non-polynomial functions problem-adapted basis`;
  - `exponentially fitted difference scheme discontinuous diffusion coefficient interior layer`, `fitted operator method interface problem discontinuous coefficient high order` and `steep diffusion coefficient interior layer finite difference uniformly convergent rapidly varying coefficient`: every hit had a small diffusion parameter, none a layer in α;
  - `Lynch Rice ... HODIE`;
  - `unresolved thin layer high contrast diffusion coefficient finite difference stencil smooth transition narrower than grid spacing` (noise).
- **zbMATH:**
  - `ti:Exact and Truncated Difference Schemes for Boundary Value ODEs` 1;
  - `au:Makarov, V* & ti:exact difference` 20;
  - `au:Kutniv & ti:three-point` 26;
  - `ti:truncated difference schemes` 12;
  - `ti:exact difference scheme* & ti:parabolic` 5 (Łapińska-Chrzczonowicz & Matus 2008: exact only for separable structures);
  - `ti:fitted & ti:discontinuous coefficient*` 0;
  - `au:Widder & ti:heat equation` 12.
- **arXiv:**
  - `abs:"quasi-Trefftz"` 16, all read by title;
  - `ti:Trefftz` 55, the 2025+ titles read;
  - `abs:Trefftz AND abs:"heat equation"` 0;
  - `abs:Trefftz AND (abs:parabolic OR abs:diffusion)` 8;
  - `abs:"heat polynomials"` 9;
  - `abs:"heat polynomials" AND (abs:"variable coefficient" OR abs:"variable coefficients")` 0;
  - `abs:"exact difference scheme"` 1;
  - `abs:"truncated difference scheme"` 0;
  - `abs:"sub-grid" AND abs:"diffusion coefficient" AND abs:"finite difference"` 0;
  - `abs:"tailored finite point"` 7;
  - `au:Tsukerman AND abs:Trefftz` 9.
- **The lead's own arXiv queries** for the SPPS preprints: `ti:spectral AND ti:parameter AND ti:power AND ti:series` 14, found arXiv 0811.4488 (Kravchenko & Porter). Theorem 1, eqs. 6–8 and Corollary 9 were read in its PDF, and the identity of K2 was checked numerically (the test named there).
- **Full texts:**
  - Tsukerman 2010 (arXiv 0906.1388);
  - the MsTFPM paper (intlpress);
  - Makarov, Mayko & Ryabichev 2023 (UMJ, in Ukrainian);
  - Gómez–Perinati–Stocker (arXiv 2411.14819).

**Verdict.**

- The exact and truncated scheme line covers the 1-D steady
  boundary-value problem `(k u′)′ − q u = −f`: piecewise-smooth k,
  three-point stencils, any order, coefficients from multiple integrals
  or Cauchy problems per cell (K3). Its parabolic case is exact only for
  special structures.
- FLAME, TFPM and MsFV put local solutions into stencils (K4).
- The Galerkin bases and the quasi-Trefftz spaces are as in K5–K6.
- Not found: a five-point or RBF-FD stencil made exact on the
  time-polynomial solutions of the variable-coefficient heat equation, or
  any stencil for a smooth layer thinner than h built by marching ODEs
  in a normal coordinate, in 2-D or for a parabolic problem.
- Open: whether Samarskii–Makarov need the discontinuities of k at nodes
  (the review does not say; §4).

### P4 — coefficient treatments in diffusion

- **Patankar** (QC1):
  - Crossref: doi 10.1201/9781482234213 (the CRC 2018 reissue; ch. 4 is
    doi …-4, pp. 41–77).
  - zbMATH `au:Patankar ti:Numerical heat transfer` gave Zbl 0521.76003.
  - archive.org advancedsearch
    `title:(numerical heat transfer fluid flow) AND creator:(patankar)`
    gave 1 hit, item `numericalheattra0000pata`, access-restricted.
  - Full-text search within that item for "interface conductivity",
    "harmonic mean", "abrupt changes", "(4.9)", "steady one-dimensional
    situation" and others, each 1 or 0, returned the OCR snippets:
    - contents "4.2-3 The Interface Conductivity 44";
    - index "Harmonic mean of conductivities, 45";
    - "The recommended interface-conductivity formula (4.9) is based on
      the steady, no-source, one-dimensional situation";
    - "Equations (4.10) show that k_e is the harmonic mean of k_P and
      k_E".

    The typeset form of eq. 4.9 is unconfirmed.
- **FV averaging** (QC2):
  - Crossref title lookups: Ewing–Iliev–Lazarov; Crumpton–Shaw–Ware (JCP
    116 (1995), metadata only); Eymard–Gallouët–Herbin; Forsyth–Sammon;
    Weiser–Wheeler; Moskow et al.; Goode–Appel; Kadioglu et al. (INL
    2008, "if the boundary layers are finely resolved, then the harmonic
    and arithmetic averaging techniques are identical in the truncation
    error sense", [V, OSTI; not entered]).
  - WebSearch
    `"harmonic mean" conductivity nodes "fourth-order" finite difference variable coefficient "non-conservative" discontinuous diffusion`
    found no source for a nodal harmonic mean under a high-order
    `Dx A Dx`.
  - arXiv `abs:"harmonic average" AND abs:"coefficient" AND abs:"order"`
    4, including Pan–Xu–Li (arXiv 2502.09413, read).
- **Regularised coefficients** (QC3):
  - Crossref: Li–Lowengrub–Rätz–Voigt 2009 (diffuse domain; domain
    boundaries, not material edges; not entered); Lervåg–Lowengrub 2015;
    Burger–Elvetun–Schlottbom 2017; Schlottbom 2016.
  - Tornberg–Engquist 2003 and 2006.
- **T3** (QC4):
  - Crossref:
    - `band-limited conductivity model controlled-source electromagnetic finite-difference`;
    - `anti-aliased conductivity electromagnetic finite difference interface`;
    - Mittet 2010, 2017, 2021; Mittet SEG 2018; Yang–Mittet 2023;
      Moczo et al. 2023; Kristek et al. 2025 (seismic; wavenumber-limited
      medium).
  - arXiv:
    - `abs:"band-limited" AND abs:"conductivity" AND abs:"finite-difference"` 0;
    - `abs:"anti-aliased" AND (abs:"conductivity" OR abs:"permeability" OR abs:"diffusivity")` 13 (image processing);
    - `abs:"band-limited" AND (abs:"diffusion coefficient" OR abs:"permeability" OR abs:"thermal conductivity")` 1, irrelevant;
    - `abs:"Heaviside" AND abs:"band-limited" AND abs:"interface"` 0.
  - OpenAlex: works citing Mittet 2017 (35), Mittet 2018 (3), Mittet 2021
    (10) and Koene 2022 (10), read by title.
  - Full texts: Mittet 2010, Mittet SEG 2018, Koene et al. (arXiv
    2104.08206), libEMM (arXiv 2304.00233), Yang–Mittet (arXiv v1).
- **Task vocabulary** (QC5), Crossref:
  - `heat conduction thin functionally graded interlayer between dissimilar materials numerical`;
  - `steep conductivity gradient thin transition layer heat conduction high-order numerical scheme`;
  - `subgrid conductivity variation finite difference diffusion coefficient narrower than grid spacing`;
  - `unresolved thin layer variable conductivity numerical diffusion scheme`;
  - `interphase thermal conduction composite numerical graded interphase thickness smaller than mesh`.

  They found FGM meshless papers, which resolve the gradation (Sladek et
  al. 2003; Liu–Ming 2018, "material properties are defined at the
  node"), and Hashin 2001.

**Verdict.** QC1: pinned. QC2: the exact face conductance is Pan–Xu–Li's
harmonic average method. The nodal and exact rules are distinguished by
Goode–Appel, and no source was found for the nodal harmonic mean under a
non-conservative high-order operator. QC3: Tornberg–Engquist for
regularisation, with the diffuse-domain papers left out (boundaries).
QC4: T3 not found in use for diffusion (§1d). QC5: no paper treats a
smooth transition narrower than the grid as distinct from a jump or a
resolved variation.

### P5 — thin layers, imperfect interfaces, contact resistance

- **Crossref** `query.bibliographic`, one per candidate:
  - Swartz–Pohl; Pham Huy–Sanchez-Palencia; Bövik; Benveniste;
    Hashin; Benveniste–Miloh; Geymonat–Krasucki–Lenci; Sanchez-Palencia
    1980; Kapitza 1941;
  - Guittet et al.; Cisternino–Weynans; Lombard–Piraux 2003 and 2006;
    Leguèbe–Poignard–Weynans; Kolahdouz–Salac;
    Perrussel–Poignard; Schmidt–Tordeux; Haddar–Joly–Nguyen;
  - by subject:
    - `elliptic interface problems with imperfect contact`;
    - `immersed interface method imperfect interface jump condition proportional to flux`;
    - `matched interface and boundary method imperfect interface`;
    - `ghost fluid method Robin jump condition interface Poisson imperfect`;
    - `kernel-free boundary integral method imperfect interface`;
    - `implicit jump condition interface problem finite difference high order`;
    - `Robin-type interface condition elliptic high order accuracy immersed`.
- **arXiv:**
  - `abs:"imperfect interface" AND (abs:"finite difference" OR abs:"immersed interface" OR abs:"finite element" OR abs:"boundary integral")` 4;
  - `abs:"imperfect contact" AND abs:interface` 3;
  - `abs:"Kapitza resistance" AND (abs:numerical OR abs:method)` 9;
  - `abs:"Robin interface"` 8;
  - `abs:"Robin-type interface"` 9;
  - `abs:"spring-mass" AND abs:interface` 12;
  - `abs:"imperfect interface" AND abs:order` 2;
  - `abs:"imperfect interface" AND abs:"thin layer"` 0;
  - `abs:"RBF-FD" AND abs:interface AND (abs:jump OR abs:discontinuous)` 0;
  - `abs:"thin layer" AND abs:"radial basis"` 0.
- **OpenAlex:**
  - `"imperfect contact" interface "fourth-order"` 83;
  - `"implicit jump condition"` 3;
  - `"Kapitza" "immersed interface"` 8;
  - `"thermal contact resistance" "meshless"` 30, none relevant.
- **WebSearch** (three queries), including
  `"imperfect interface" OR "imperfect contact" finite difference "order reduction" OR "loses one order" OR "loss of one order" jump flux interface`.
- **Full texts read in part:**
  - Lombard–Piraux 2003 (HAL hal-00004812) and 2006 (arXiv
    physics/0508018v2; the §4.7 and §5.2 quotes re-checked by the lead);
  - Chiavassa–Lombard 2013 (arXiv 1109.3281);
  - Cao–Yuan 2023;
  - Leguèbe et al. (INRIA RR-8302);
  - Guittet et al. 2015, 2017;
  - Perrussel–Poignard (INRIA RR-7163);
  - Kolahdouz–Salac (arXiv 1311.0824);
  - Attanayake–Chou–Deng (arXiv 2204.07665).

**Verdict.**

- The resistive-interface limit is classical (K9).
- Strong-form schemes for `[u] = R q` exist to fourth order in 1-D and
  about second order in 2-D. Galerkin methods reach optimal order with
  the condition imposed weakly (Jo & Kwak 2018; Attanayake et al. [V, not
  entered]).
- The R8 observation is **partly prior**: Lombard–Piraux 2006 for 2-D
  elastic waves (K9).
- Not found: the effect for the scalar diffusion operator in 2-D, on
  scattered nodes or with RBF-FD; the framing of a layer thinner than h
  crossed by ODE-continued functions; the targeted remedy of five
  flux-carrying functions of one degree higher.
- Kapitza 1941 (*J. Phys. USSR*) could not be verified; the *Phys. Rev.*
  60 (1941) letter was [V] and is not entered.

### P6 — bibliography verification

- **Entries.** `paper/references.bib` holds 94 entries:
  - the four E5.1 seeds, re-verified;
  - 29 carried from the companion's bib, re-fetched rather than copied;
  - 61 new from P1–P5.
- **Each has a dated `VERIFIED 2026-09-24` note** naming its fetch:
  - Crossref for the 86 DOIs, compared field by field by a scratch
    script;
  - zbMATH Open for Patankar 1980, Samarskii–Makarov 1990, Widder 1975,
    Karlin–Studden 1966 and Sanchez-Palencia 1974;
  - the arXiv API for Imbert-Gérard 2025;
  - the dissertation's PDF plus OpenAlex W2595797237 for Martin 2016;
  - the companion's own `main.tex` for Martin2026seeds.
- **The script's remaining mismatches are Crossref quirks,** each resolved
  in the entry's comment:
  - article numbers in place of pages;
  - `&amp;` in container titles;
  - a stray affiliation "author" on Huang 2009;
  - "De G. Allen";
  - no authors on Roos–Stynes–Tobiska and Goode–Appel (filled from the
    USGS page);
  - issued years that are the online date (Hauck–Peterseim, Koene et
    al., Cao–Yuan).
- **Corrections made from the records:**
  - Tornberg & Engquist 2006: Crossref lists the authors in the wrong
    order; the paper's first page (read) has Tornberg first.
  - Vizvari et al.: the names carry no accents in the record.
  - Kravchenko–Otero–Torba: pages 1–5 per the record.
  - MIB 2006's co-author Feig.
  - Rosales on Marques–Nave–Rosales.
  - Li–Mayo 1994: the volume, 48, is in the DOI only and is left out.
- **Gates.** `make_arxiv.unverified_entries` passes on the file, and
  `tests/test_literature.py` pins that. Every `\cite` key of §6 exists in
  the file (the same test). The bib is safe for pdfTeX.
- **Unreachable:**
  - Springer, ScienceDirect, De Gruyter and ACM article pages (403 or a
    login redirect; Crossref used throughout);
  - archive.org's page images of Patankar (access-restricted);
  - mathnet.ru, not tried this time (the companion found it unreachable).

## 3. Verification-status rule

- **[V]** requires a fetched primary source: the Crossref API record of the
  DOI (publisher-deposited metadata), the arXiv API record, a zbMATH Open
  or OpenAlex record, the publisher's or repository's page, or the paper's
  own PDF. The fetch is named in §2 or in the entry's `VERIFIED` comment in
  `paper/references.bib`.
- **[S]** is search-result or page-summary evidence only. It may be
  discussed here and in prose with hedging. It may **not** enter
  `paper/references.bib`, and no §1 verdict may rest on an [S] item alone.
- A metadata [V] is not a content [V]. Where a claim about what a paper
  *does* rests on its abstract, a review or a first page, §1–§2 say so.
  Where nothing was read, they say "metadata only". The manuscript quotes
  content only from papers read at source.
- Preprints ([V] via the arXiv API) are cited as preprints and weighted
  accordingly.

## 4. Standing hazards

- **Vocabulary splits the field.** The three lines closest to the
  construction live in three communities and never say "interface
  stencil" or "RBF":
  - the Samarskii school's exact and truncated schemes ("homogeneous
    difference schemes", "Cauchy problems", "piecewise smooth
    coefficients");
  - Kravchenko's spectral-parameter power series ("formal powers",
    "transmutation");
  - Tsukerman's FLAME ("Trefftz", "flexible local approximation").

  Sujecki's step translation is in photonics. None cites the others or
  the 2016 work, and the 2016 work cites none of them (P1, finding 2).
  More may exist under "analytic nodal", "tailored", "exact difference
  scheme", "generalized solutions" or "template functions". Sweep those
  first-class before circulation.
- **Absence ≠ absence of practice.** Every "no prior instance found" is
  tied to the listed queries. Phrase every such statement as "no prior
  instance found", never "novel".
- **Content unread at source:**
  - Tikhonov–Samarskii 1962 and Samarskii 2001: the attribution of the
    exact face coefficient to them follows Makarov et al. 2023's
    introduction and the companion.
  - Samarskii–Makarov 1990 and Gavrilyuk et al. 2011: zbMATH reviews
    only. Whether the jumps of k must sit on nodes is unconfirmed.
  - Sanchez-Palencia 1974: title only.
  - Tsukerman 2006: metadata only; the method is quoted from 2010.
  - Cleary–Monaghan 1999: quoted via Monaghan 2005.
  - Cheng–Ju–Zhang 2026: metadata only.
  - Patankar's eq. 4.9 in typeset form.
  - The Lombard–Piraux quotes come from preprints, not the SISC text.

  Read before quoting specifics beyond what §1a records.
- **Fast-moving neighbours.** Re-run before the arXiv upload:
  - the quasi-Trefftz group, which has a heat-equation paper out (2026)
    and posts often; if a quasi-Trefftz or FLAME paper for
    *under-resolved* or steep coefficients appears, O1's bucket must be
    re-examined;
  - the LOD line (K5), which already reaches "under-resolved rough
    coefficients".
- **Galerkin versus strong form.** K5's and K9's Galerkin results (LOD's
  rates, optimal order with Robin-type jumps) mean every O-item must say
  "stencils" or "strong form". "High order through a sub-grid layer"
  alone is refuted.
- **The companion is unsubmitted.** `Martin2026seeds` says "Manuscript"
  until its arXiv identifier exists; #52 fills it before this manuscript
  is packaged.
- **Re-runs send only placeholders.** The bibliographic APIs ask for a
  contact in a `mailto` parameter or the User-Agent ("polite pool"). Every
  re-run of these queries puts a placeholder there, never a personal
  address, and each search brief says so.
- **Bibliography labels.** UTF-8 accents in single-author surnames
  within amsalpha's label prefix (Pólya, Mühlbach) give non-ASCII labels. `make_arxiv.py` checks that they render; #51 decides the
  style, as in the companion.

## 5. Cross-references

- `paper/references.bib`: verified entries only; each carries a dated
  `VERIFIED` comment naming its fetch.
- `docs/stiff-diffusion.md` §1.10 and §3.9: the reference lists this
  ledger re-verified; the note is canonical for every number quoted in
  §1a K10 (iii) and §1b.
- `docs/plan.md` §3.4 (T3) and §5 (R1, R8): answered here, with pointers
  back.
- `docs/paper-index.md`: page maps for the dissertation and EABE 2017 read
  in P1.
- Issues:
  - #7, the manuscript epic;
  - #43, this pass;
  - #45, which places §6a;
  - #46, which cites K1–K3 and K7 in §3;
  - #47 and #49, the comparator paragraphs (K10);
  - #48, the ring's flux seeds (K9);
  - #50, §6b in the conclusions;
  - #51, which re-checks the wording against this section;
  - #52, the companion's identifier and the pre-circulation re-runs.

## 6. Manuscript wording

The paragraph and sentences below are the only novelty language the
manuscript may use without re-opening this ledger. #45 places (a) in §1
(`\subsection{Relation to prior work}`, `sec:priorwork`); #45 and #50 fix
(b) in the abstract, the contributions and the conclusions. Every `\cite`
key exists in `paper/references.bib` and is [V]
(`tests/test_literature.py`). `\cite` commands to [V] keys may be inserted
into these sentences where the manuscript places them; the words may not
change (#45, Brad, 2026-09-24: §1 cites Lombard–Piraux inside §6b's
flux-seed sentence, as K9 asks).

### 6a. Relation to prior work (§1 of the manuscript)

```latex
In one dimension the construction is classical in every part, and we
claim nothing for it. The first seed, the constant-flux function
$\alpha_e \int d\xi/\alpha$, is over one cell the exact face conductance of
Tikhonov and Samarskii's homogeneous schemes
\cite{TikhonovSamarskii1962,Samarskii2001} and, for a piecewise-constant
coefficient, the harmonic-mean interface conductivity of finite-volume heat
conduction \cite{Patankar1980}; the conservative scheme built on it, exact
at the source-free equilibrium, is second order \cite{PanXuLi2026}. The whole chain is the
representation of Sturm--Liouville solutions by recursive integrals, as
power series in the spectral parameter \cite{Kravchenko2008,KravchenkoPorter2010},
taken from the anchor; for a constant coefficient its time-polynomial
solutions are the heat polynomials \cite{RosenbloomWidder1959,Widder1975}.
Three-point schemes of any order for $(k u')'$ with a piecewise-smooth $k$,
whose coefficients come from Cauchy problems integrated across each cell,
are the exact and truncated difference schemes
\cite{SamarskiiMakarov1990,GavrilyukEtAl2011}; three-point schemes of any
order exist across a jump too \cite{Gartland1989}, one of them built by
carrying Taylor data across the step \cite{Sujecki2010}.
Stencils exact on a local space other than the polynomials go back to
fitted operators and to HODIE
\cite{AllenSouthwell1955,Ilin1969,LynchRice1980,RoosStynesTobiska2008}. In
two dimensions, local solutions of the equation that satisfy the interface
conditions are the basis of the Trefftz difference schemes of FLAME, on
irregular stencils too \cite{Tsukerman2006,Tsukerman2010}, and of the
tailored finite point methods \cite{Huang2009,HanZhang2012}; Galerkin
methods build their bases from local solves
\cite{BabuskaCalozOsborn1994,HouWu1997,ChuGrahamHou2010,OwhadiZhang2007,MalqvistPeterseim2014}
and reach high order through coefficients the mesh does not resolve
\cite{Maier2021,HauckPeterseim2023}, and the quasi-Trefftz spaces, one of
them for the heat equation, build polynomial local solutions by Taylor
expansion \cite{ImbertGerardMoiolaPerinatiStocker2025,GomezPerinatiStocker2026},
which cannot reach through an edge thinner than the stencil
\cite{Martin2026seeds}. For a jump, the interface-aware RBF-FD stencils this
work generalises \cite{Martin2016,MartinFornberg2017heat} stand beside the
immersed interface, ghost-fluid and matched-interface methods and unfitted
finite elements
\cite{LeVequeLi1994,LiIto2006,LiuFedkiwKang2000,ZhouZhaoFeigWei2006,HansboHansbo2002,BurmanEtAl2015},
now of fourth and sixth order \cite{FengZhao2020,FengHanMinev2024}, all of
which take the jump as given. A resistive layer thinner than the grid is an
imperfect interface in the limit
\cite{SanchezPalencia1974,PerrusselPoignard2013,Hashin2001}, and
strong-form schemes across such a contact need its jump conditions one
derivative higher in two dimensions \cite{LombardPiraux2003,LombardPiraux2006};
the flux seeds of Section~\ref{sec:ring} are that observation for the
diffusion operator. The standing alternative changes the coefficient rather
than the stencil: harmonic or arithmetic means over a cell
\cite{Patankar1980,EwingIlievLazarov2001,GoodeAppel1992}, averaging for
layering finer than the grid \cite{MoskowEtAl1999,DavydychevaDruskinHabashy2003},
or a regularised coefficient \cite{TornbergEngquist2003,TornbergEngquist2006}.
Those methods take the cell or the width from the grid; here the width is a
property of the medium, below the node spacing, and the aim is the scheme's
full order through it. We found no prior instance of stencils on an
unchanged scattered node set whose basis is continued through a sub-grid
smooth edge by ordinary differential equations, nor of such stencils
reducing to a jump construction as $\delta \to 0$; the search is logged in
the repository's \texttt{LITERATURE.md}. We ran the cell means, a widened
edge and, in one dimension, the conservative scheme with exact face
conductances, as our implementation of each on one problem per dimension:
through an edge the grid does not resolve none is better than second order,
and on the scattered nodes none improves on sampling the coefficient by more
than a factor of $1.75$ in the root-mean-square error. We claim nothing beyond
that measurement.
```

(About 500 words, longer than §1 of the manuscript can carry. #45 cuts it
to 250–350 words. In order, it may:

1. drop the HODIE and fitted-operator sentence, which §3 cites anyway;
2. drop the list of jump methods down to the IIM and CutFEM keys plus the
   post-2016 two;
3. move the resistive-layer sentence to §5.4 (`sec:ring`), keeping one
   clause in §1.

It keeps verbatim the first sentence, the FLAME and LOD clauses, and the
last four sentences, which carry the obligations of §1a K2–K5, K10 and §1b
O1.)

*Placed by #45 (2026-09-24), with the three cuts and no other.* They save
less than the note above expected: the paragraph in §1 is about 480 words
(476 without the `\cite` keys), not 250–350. Reaching 350 would also cut
the face-conductance (K1) and formal-powers (K2) sentences, the only
support §1 then gives "classical in every part" and the abstract's
formal-powers clause beyond K3. Brad accepted the longer paragraph over
that; the 250–350 target is retired. It stays one paragraph, since O1 asks
for K3–K6 beside the claim.

### 6b. Claim sentences (abstract, contributions, conclusions)

Bucket §1b, worded as "no prior instance found":

- *Abstract-safe.* "The seeds are the profiles of solutions polynomial in
  time, continued through the edge by ordinary differential equations
  along the normal or along the edge's own coordinates; in one dimension
  they are the classical formal powers of the Sturm--Liouville operator,
  and as $\delta \to 0$ they reduce to the interface-aware stencils of a
  jump. On scattered nodes the resulting RBF-FD stencils are fourth order
  at every $\delta$ tested, elliptic and parabolic, with no switch between
  regimes."
- *Contributions / conclusions.* "To our knowledge no earlier scheme builds
  the basis of stencils on an unchanged scattered node set through a smooth
  material edge thinner than the node spacing by marching ordinary
  differential equations; its ingredients (the exact face conductance, the
  formal powers, bases from local solutions of the equation) are each
  classical, and in one dimension so is the construction."
- *Required caveat wherever the comparison with coefficient treatments is
  a headline.* "The comparison is against coefficient sampling on the same
  nodes and against our implementation of the cell harmonic and arithmetic
  means, a widened edge and, in one dimension, the conservative scheme with
  exact face conductances, on one contrast and one test problem per
  dimension. Through an edge the grid does not resolve none of them is
  better than second order; the conservative scheme is exact at the
  one-dimensional equilibrium and, on the sinusoidal medium at
  $\delta = 0$, ahead of the seeds below 801 nodes, where the seeds' line
  is pre-asymptotic;
  on the scattered nodes none improves on sampling by more than a factor of
  1.75 in the RMS norm, and the seeds lie 2.3 to 4.7 orders below the best
  of them over 1250 to 40,000 nodes."
- *Required wherever the flux seeds are named.* "An imperfect contact costs
  strong-form schemes a derivative in the jump conditions, as Lombard and
  Piraux observed for elastic waves; the flux seeds add the functions of
  one degree higher that carry the flux, and nothing else."

Not to be used:

- "novel", "new method", "first";
- "the first high-order method through sub-grid coefficients", refuted by
  K5;
- "outperforms existing interface methods";
- "beats harmonic averaging / finite volumes" (only our implementations,
  and T1-FV wins at 1-D equilibrium and on coarse eq. 75 grids);
- "arbitrary contrast", "any order" (only degree 4 / FD4 was run);
- "the flux must be one degree higher" as a general fact (K9);
- "Kapitza resistance" for the ring (a contact resistance; K9).
