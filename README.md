# Unfolding Supercell Bands & DOS (DFTK.jl)

**Repository:** Unfolding technique implemented with [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl))

**Author:** @physarief78

**Summary**

This README documents a low-memory, chunked pipeline to perform a supercell self-consistent-field (SCF) calculation with DFTK, compute chunked eigenvalues for DOS, compute band structure along a primitive-cell path using the *supercell* wavefunctions, and *unfold* the supercell bands onto the primitive Brillouin zone (BZ). The implementation is intentionally memory-conservative (waves and large arrays are quickly dropped and garbage-collected) and chunked so it runs under limited RAM budgets.

Included example results (place these images in `figures/` in the repo):

* `src_code/results/Si_Supercell_NType_Unfolded.png` — N-type supercell unfolded bands + DOS
* `src_code/results/Si_Supercell_PType_Unfolded.png` — P-type supercell unfolded bands + DOS
* `src_code/results/Si_Supercell_Unfolded.png` — pristine 2×2×2 supercell unfolded bands + DOS

---

## 1. DFTK Workflow (high level)

1. **Build primitive and supercell models** — define lattice, atomic positions, pseudopotentials, create `model_DFT` for both primitive cell and supercell geometry.
2. **SCF on supercell** — run `self_consistent_field` on the supercell to obtain the converged density $\rho_{SC}$ and Fermi energy $\varepsilon_F$. Immediately save only the density and Fermi energy to disk and drop the wavefunction-heavy objects from memory.
3. **Chunked eigenvalue extraction for DOS** — iterate over each supercell k-point (or small groups) and call `compute_bands(..., ρ=ρ_{SC})` to compute eigenvalues at that k-point. Collect eigenvalues only (no persistent ψ storage). Use DFTK smearing to compute a DOS curve.
4. **Chunked band-path computation** — build a path in the *primitive* BZ (e.g. L → Γ → X → K → Γ), map these primitive path k-points to the supercell reciprocal coordinates, and compute bands for those k-points in small chunks. For each supercell band at each supercell k-point, compute an **unfolding spectral weight** onto primitive k-points using the supercell plane-wave coefficients.
5. **Plot** — scatter unfolded bands (energy vs primitive-path coordinate) where color (and/or marker area) encodes the spectral weight; plot the DOS beside it.

Note: the code is optimized for low-memory operation: it processes k-points in chunks, drops heavy objects, and forces `GC.gc()` to reduce peak memory.

---

## 2. Mathematical formulation

This section translates the implementation logic into compact mathematical statements so you can verify and reproduce the approach independently.

### 2.1 Lattices and reciprocal lattices

Let the **primitive direct lattice** be given by the 3×3 matrix of column vectors $a^{(p)} = [\mathbf{a}_1^{(p)},\, \mathbf{a}_2^{(p)},\, \mathbf{a}_3^{(p)}]$. The corresponding **primitive reciprocal lattice** is

$$
\bigl[\mathbf{b}*i^{(p)}\\bigr]_{i=1}^3 \quad \text{with}\quad B^{(p)} = 2\pi (a^{(p)})^{-T},
$$

so that so that $\mathbf{b}_{i}^{(p)}.

For the supercell (constructed by integer multiples along direct axes), the supercell direct lattice matrix is $a^{(s)}$ and its reciprocal lattice is

$$
B^{(s)} = 2\pi,(a^{(s)})^{-T}.
$$

The coordinates used in DFTK are consistent with these definitions.

### 2.2 Bloch wave expansion (plane-wave representation)

For a supercell calculation at supercell Bloch vector $\mathbf{K}$ the single-particle eigenstate (band index (n)) is expressed in plane waves as

$$
\Psi_{n\mathbf{K}}^{(s)}(\mathbf{r}) = \frac{1}{\sqrt{\Omega_s}}\sum_{\mathbf{G}^{(s)}} C_{n \mathbf{K}}(\mathbf{G}^{(s)}) e^{i(\mathbf{K}+\mathbf{G}^{(s)})\cdot\mathbf{r}},
$$

where $\mathbf{G}^{(s)}$ are reciprocal lattice vectors of the supercell, $C_{n \mathbf{K}}(\mathbf{G}^{(s)})$ are complex plane-wave coefficients returned by DFTK, and $\Omega_s$ is the supercell volume.

### 2.3 Primitive k mapping (unfolding condition)

We want to discover whether a supercell state $\Psi_{n\mathbf{K}}^{(s)}$ contributes spectral weight to a particular **primitive** Bloch vector $\mathbf{k}$ (a k-point on the primitive path). Because primitive and supercell reciprocal lattices are commensurate, any supercell plane wave label (\mathbf{K}+\mathbf{G}^{(s)}) may be expressed in the primitive reciprocal basis. If there exists a supercell reciprocal vector $\mathbf{G}^{(s)}$ such that

$$
\mathbf{k} = \mathbf{K} + \mathbf{G}^{(s)} \quad(\text{mod } B^{(p)}),
$$

then that plane-wave component of the supercell eigenstate maps onto the primitive k-point $\mathbf{k}$.

In coordinates used by the code we compute for each supercell G-vector a *difference vector* in Cartesian form

$$
\mathbf{q} = \mathbf{K}_{\text{cart}} + B^{(s)} \mathbf{G}^{(s)},
$$

and compare it against the primitive target $\mathbf{k}_{\text{cart}}$. The condition is that

$$
\Delta = (a^{(p)})^{T}(\mathbf{q} - \mathbf{k}_{\text{cart}}) / (2\pi)
$$

should be an integer triplet (within a small numerical tolerance). Code checks `all(x -> abs(x - round(x)) < tol, Δ)` with `tol ≈ 1e-3`.

### 2.4 Spectral weight (unfolding weight)

When a plane-wave component labeled by $\mathbf{G}^{(s)}$ satisfies the mapping condition to primitive $\mathbf{k}$, that component contributes its squared amplitude to the spectral weight of primitive $\mathbf{k}$. The **spectral weight** for band $n$ at supercell k-point $\mathbf{K}$ onto primitive k-point $\mathbf{k}$ is

$$
W_{n\mathbf{K}}(\mathbf{k}) = \sum_{\mathbf{G}^{(s)}:\ \mathbf{K}+\mathbf{G}^{(s)}\equiv\mathbf{k}  (\text{mod } B^{(p)})} \bigl|C_{n,\mathbf{K}}(\mathbf{G}^{(s)})\bigr|^2.
$$

The code builds the weights by summing `abs2(coeffs[iG, ib])` for all matching G-vectors. After summing, a small cutoff (e.g., `weights > 0.01`) is applied to remove negligibly-weighted contributions in the plotted unfolded bands.

The weights satisfy a normalization property across primitive k-points (for a complete decomposition), i.e. the sum of the weights across all primitive k-points for a given supercell $n,\mathbf{K}$ equals the band occupancy (approximately 1 for a single-particle normalized state, modulo numerical issues and truncation):

$$
\sum_{\mathbf{k}} W_{n\mathbf{K}}(\mathbf{k}) \approx 1.
$$

### 2.5 Unfolded DOS

The density of states unfolded onto the primitive BZ or just the **total DOS** can be computed by smearing eigenvalues. Let (g_\sigma(E)) be a smearing kernel (e.g., Gaussian) with width (\sigma). The DOS per unit energy (states/eV) is computed as

$$
\mathrm{DOS}(E) = \frac{1}{N_{k}} \sum_{\mathbf{K},n} g_\sigma\bigl(E - E_{n\mathbf{K}}\bigr)
$$

where the code uses DFTK's `smearing` function and samples all supercell k-points. If one wants an *unfolded* DOS resolved per primitive k-point, insert the weight:

$$
\mathrm{DOS}*{\text{unfolded}}(\mathbf{k},E) = \frac{1}{N*{K}}\sum_{\mathbf{K},n} W_{n\mathbf{K}}(\mathbf{k}),g_\sigma\bigl(E-E_{n\mathbf{K}}\bigr).
$$

In the example code the DOS plotted on the side is the total DOS (summed over supercell k-points) and is converted from Hartree energies to electronvolts using

$$
1~\mathrm{Ha} = 27.2114~\mathrm{eV}.
$$

### 2.6 Units and conversions

* DFTK returns energies in Hartree (Ha). The code uses `E_eV = (E_Ha - ε_F) * 27.2114` to shift by the Fermi energy and express energies relative to (E_F) in eV.
* Reciprocal transformations use the `reciprocal_lattice` from `basis.model` (DFTK).

---

## 3. Implementation notes (mapping to code)

* `build_primitive_si()` and `build_supercell_si()` construct primitive and supercell lattices and atomic lists.
* Running SCF on the supercell yields `scfres.ρ` (saved to a `.jld2` file) and `scfres.εF`.
* For memory efficiency, `compute_bands(basis_sc_scf, kgrid_obj; n_bands=320, ρ=ρ_sc)` is called for a single k-point or small k-point chunk. Only `res.eigenvalues` (and `res.ψ` temporarily for unfolding) are used — and cleared after the chunk is processed.
* Unfolding logic lives in the nested loop where the code constructs `q_cart = K_sc_cart + recip_lattice_sc * G_int` and checks that `n_check = (lattice_prim' * diff) / 2π` is close to integer.
* We accumulate `plot_k`, `plot_E`, `plot_W` arrays that are plotted with `CairoMakie` as a scatter plot where `color = weights`.

---

## 4. How to run (quick start)

1. Install Julia (v1.9+ recommended) and add packages used in the script: `DFTK`, `PseudoPotentialData`, `CairoMakie`, `JLD2`, `FFTW`, `StaticArrays`, etc.

2. Put the script `si_unfold_dos_fully_chunked.jl` and the `figures/` images into a git repository.

3. Run the script with multiple threads to speed up embarrassingly-parallel parts:

```bash
julia -t 4 si_unfold_dos_fully_chunked.jl
```

Adjust `-t` to the number of threads available. If memory is tight, reduce `CHUNK_SIZE`.

**Parameters to tune in the script**

* `E_CUT` — plane-wave energy cutoff (Ha). Increase for convergence.
* `KGRID_PRIM`, `KGRID_SC` — k-point grids for primitive and supercell.
* `n_bands` passed to `compute_bands` — must be large enough to include all bands of interest.
* `CHUNK_SIZE` — number of path k-points computed per chunk (smaller reduces memory peak at cost of more calls).
* `tol` for integer-check in mapping — default `1e-3` in the code.

---

## 5. Interpreting results (example)

Place the example images into `figures/` and inspect them. The left panel shows unfolded bands (scatter) along the primitive path L → Γ → X → K → Γ. The color/marker intensity represents the **spectral weight**: bright/dark points indicate strong primitive character in that supercell eigenstate at that primitive k. The right panel shows the *total DOS* computed from the supercell eigenvalues. The horizontal dashed line marks the Fermi energy (0 eV on the energy axis).

---

## 6. Tips & caveats

* The unfolding quality depends on commensurability between the supercell and primitive cell. If supercell is not an integer repetition, unfolding becomes ambiguous.
* The method is plane-wave based: all mapping relies on a consistent plane-wave indexing between primitive and supercell reciprocals.
* Numerical tolerances (for integer matching) must be set carefully — too loose and false matches appear, too strict and genuine matches are lost.
* The `weights > 0.01` plotting cutoff is arbitrary; choose a threshold appropriate to your system and basis size.

---

## 7. Files & expected layout

```
README.md  (this file)
si_unfold_dos_fully_chunked.jl
figures/
  ├─ Si_Supercell_NType_Unfolded.png
  ├─ Si_Supercell_PType_Unfolded.png
  └─ Si_Supercell_Unfolded.png
```
