#!/usr/bin/env julia
# si_unfold_doped_chunked.jl
#
# AUTOMATED PIPELINE (DOPED SUPERCELL + Low Memory):
# 1. GENERATE: 2x2x2 Supercell with 1 Boron atom at the center (P-Type).
# 2. SCF: Computes Density (Drops heavy wavefunctions immediately).
# 3. DOS: Computes Eigenvalues iteratively.
# 4. BANDS: Computes Path iteratively.
# 5. PLOT: Unfolds and plots side-by-side.

using LinearAlgebra
using DFTK
using PseudoPotentialData
using CairoMakie
using JLD2
using Printf
using Statistics
using FFTW
using StaticArrays
using Dates

# --- Global Settings ---
setup_threading()
const t_start = time()

n_threads = Threads.nthreads()
println("Running on $n_threads threads.")

const E_CUT = 20.0       # Hartree
const KGRID_PRIM = [4, 4, 4]
const KGRID_SC   = [2, 2, 2] 
const TOL_SCF    = 1e-5
const CHUNK_SIZE = 4     # Low memory chunking

# ==============================================================================
# 1. Structure Definitions
# ==============================================================================
function build_primitive_si()
    # Reference lattice (must remain Pure Si for unfolding math to work)
    a_ang = 5.43
    a_bohr = a_ang * 1.8897259886
    lattice = (a_bohr / 2) * [[0 1 1]; [1 0 1]; [1 1 0]]
    positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    atoms = [ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")) for _ in positions]
    return lattice, atoms, positions
end

function build_doped_supercell_si()
    # 1. Define Geometry
    a_ang = 5.43
    a_bohr = a_ang * 1.8897259886
    nx, ny, nz = 2, 2, 2
    
    # 2. Generate FCC Nodes (Fractional relative to Supercell)
    # The logic here mimics the "generate_doped_silicon_supercell" provided
    # but outputs fractional coordinates for DFTK.
    
    fcc_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    basis_shift = [0.25, 0.25, 0.25]
    cubic_nodes = vcat(fcc_nodes, [p .+ basis_shift for p in fcc_nodes])
    
    positions = Vector{Vector{Float64}}()
    
    # Generate all positions in the 2x2x2 grid
    for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1
        shift = [i, j, k]
        for atom_pos in cubic_nodes
            # Convert to supercell fractional coordinates
            new_pos = (atom_pos .+ shift) ./ [nx, ny, nz]
            push!(positions, new_pos)
        end
    end

    # 3. DOPING LOGIC (Center Replacement)
    # The center of the supercell in fractional coordinates is [0.5, 0.5, 0.5]
    center_point = [0.5, 0.5, 0.5]
    min_dist = Inf
    dopant_index = 0

    # Find atom closest to center
    # Note: We calculate distance in fractional space which works because cell is cubic
    for (i, p) in enumerate(positions)
        d = norm(p - center_point)
        if d < min_dist
            min_dist = d
            dopant_index = i
        end
    end
    
    println("    Supercell Construction:")
    println("    - Total Atoms: $(length(positions))")
    println("    - Doping: Replacing Atom $dopant_index at $(positions[dopant_index]) with Boron.")

    # 4. Assign Elements (Si and B)
    # We use the same PseudoFamily. It will look for Si and B potentials automatically.
    psp_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    
    atoms = Vector{ElementPsp}(undef, length(positions))
    for i in 1:length(positions)
        if i == dopant_index
            atoms[i] = ElementPsp(:B, psp_family)  # Dopant
        else
            atoms[i] = ElementPsp(:Si, psp_family) # Host
        end
    end

    lattice = diagm([nx * a_bohr, ny * a_bohr, nz * a_bohr])
    return lattice, atoms, positions
end

# ==============================================================================
# 2. Primitive Lattice (Reference)
# ==============================================================================
lattice_prim, _, _ = build_primitive_si()

# ==============================================================================
# 3. Doped Supercell SCF
# ==============================================================================
supercell_file = "doped_supercell_scf.jld2"

basis_sc_scf = nothing
ρ_sc = nothing
εF_sc = 0.0

if isfile(supercell_file)
    println("\n[1/4] Found cached Doped Supercell SCF. Loading...")
    dat = JLD2.load(supercell_file)
    l, a, p = build_doped_supercell_si()
    model = model_DFT(l, a, p; functionals=PBE(), temperature=1e-3)
    
    global basis_sc_scf = PlaneWaveBasis(model; Ecut=E_CUT, kgrid=KGRID_SC)
    global ρ_sc = dat["rho"]
    global εF_sc = dat["epsilonF"]
else
    println("\n[1/4] Running Doped Supercell SCF (Si63 B1)...")
    l, a, p = build_doped_supercell_si()
    model = model_DFT(l, a, p; functionals=PBE(), temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut=E_CUT, kgrid=KGRID_SC)
    
    # Run SCF
    nbandsalg = AdaptiveBands(model; n_bands_converge=320)
    scfres = self_consistent_field(basis; tol=TOL_SCF, nbandsalg=nbandsalg)
    
    global basis_sc_scf = basis
    global ρ_sc = scfres.ρ
    global εF_sc = scfres.εF
    
    # Save & Drop Memory
    JLD2.save(supercell_file, "rho", ρ_sc, "epsilonF", εF_sc)
    println("SCF done. Dropping heavy wavefunctions...")
    scfres = nothing 
    GC.gc()
end

# ==============================================================================
# 4. Chunked Calculation of SCF Eigenvalues (for DOS)
# ==============================================================================
println("\n[2/4] Computing SCF Eigenvalues for DOS (Chunked)...")

n_kpoints_scf = length(basis_sc_scf.kpoints)
eigenvalues_sc = Vector{Vector{Float64}}(undef, n_kpoints_scf)

# Iterate 1-by-1
for (ik, kpt) in enumerate(basis_sc_scf.kpoints)
    k_coord = kpt.coordinate 
    kgrid_obj = DFTK.ExplicitKpoints([k_coord])
    
    res = compute_bands(basis_sc_scf, kgrid_obj; n_bands=320, ρ=ρ_sc)
    eigenvalues_sc[ik] = res.eigenvalues[1]
    
    res = nothing
    GC.gc()
    if ik % 1 == 0; @printf "    Processed SCF k-point %d / %d\r" ik n_kpoints_scf; end
end
println("\n    Done computing SCF eigenvalues.")

# --- DOS Computation ---
println("    Calculating DOS curves...")
dos_energies_Ha = range(minimum(minimum.(eigenvalues_sc)) - 0.05, 
                        maximum(maximum.(eigenvalues_sc)) + 0.05, length=1000)

smearing_func = basis_sc_scf.model.smearing
temp = basis_sc_scf.model.temperature

dos_values = [sum(compute_dos(e, basis_sc_scf, eigenvalues_sc; smearing=smearing_func, temperature=temp)) 
              for e in dos_energies_Ha]

dos_energies_eV = (dos_energies_Ha .- εF_sc) .* 27.2114

# ==============================================================================
# 5. Band Structure Calculation (Chunked & Unfolding)
# ==============================================================================
println("\n[3/4] Computing Bands Path along L -> G -> X -> K -> G...")

# A. Define Primitive Path
points = [
    ([0.5, 0.5, 0.5], "L"),
    ([0.0, 0.0, 0.0], "Γ"),
    ([0.5, 0.0, 0.5], "X"),
    ([0.375, 0.375, 0.75], "K"),
    ([0.0, 0.0, 0.0], "Γ")
]

n_interp = 30
kpath_prim_frac = Vector{Vector{Float64}}()
k_axis = Float64[]
x_ticks = Float64[]
x_labels = String[]

current_dist = 0.0
for i in 1:length(points)-1
    start_k, lab_s = points[i]
    stop_k, lab_e  = points[i+1]
    push!(x_ticks, current_dist); push!(x_labels, lab_s)
    
    dist_seg = norm(stop_k - start_k)
    for t in range(0, 1, length=n_interp)
        if t < 1.0 || i == length(points)-1
            push!(kpath_prim_frac, start_k + t*(stop_k - start_k))
            push!(k_axis, current_dist + t*dist_seg)
        end
    end
    global current_dist += dist_seg
end
push!(x_ticks, current_dist); push!(x_labels, points[end][2])

# B. Map Path to Supercell
B_prim = 2π * inv(lattice_prim)' 
kpath_cart = [B_prim * k for k in kpath_prim_frac]

recip_lattice_sc = basis_sc_scf.model.recip_lattice
kpath_sc_frac = [recip_lattice_sc \ k_cart for k_cart in kpath_cart]
kpath_sc_static = [SVector{3, Float64}(k) for k in kpath_sc_frac]

# C. Chunked Calculation
println("    Total Path k-points: $(length(kpath_sc_static))")

plot_k = Float64[]
plot_E = Float64[]
plot_W = Float64[]

total_k = length(kpath_sc_static)

for chunk_start in 1:CHUNK_SIZE:total_k
    chunk_end = min(chunk_start + CHUNK_SIZE - 1, total_k)
    current_range = chunk_start:chunk_end
    
    @printf "    Processing chunk %d -> %d (%.1f%%)...\n" chunk_start chunk_end (chunk_end/total_k*100)
    
    kpath_chunk = kpath_sc_static[current_range]
    kprim_chunk = kpath_cart[current_range]
    kaxis_chunk = k_axis[current_range]
    
    # Calculate bands for chunk
    kgrid_obj = DFTK.ExplicitKpoints(kpath_chunk)
    band_data = compute_bands(basis_sc_scf, kgrid_obj; n_bands=320, ρ=ρ_sc)
    basis_bands = band_data.basis
    
    for (idx_in_chunk, k_prim_cart) in enumerate(kprim_chunk)
        Kpoint_sc = basis_bands.kpoints[idx_in_chunk]
        coeffs = band_data.ψ[idx_in_chunk]     
        evals  = band_data.eigenvalues[idx_in_chunk]
        
        K_sc_cart = recip_lattice_sc * Kpoint_sc.coordinate
        G_sc_vecs = G_vectors(basis_bands, Kpoint_sc)
        
        local weights = zeros(Float64, length(evals))
        
        for (iG, G_int) in enumerate(G_sc_vecs)
            q_cart = K_sc_cart + recip_lattice_sc * G_int
            diff = q_cart - k_prim_cart
            n_check = (lattice_prim' * diff) / 2π
            
            if all(x -> abs(x - round(x)) < 1e-3, n_check)
                for ib in 1:length(evals)
                    weights[ib] += abs2(coeffs[iG, ib])
                end
            end
        end
        
        for ib in 1:length(evals)
            if weights[ib] > 0.01
                push!(plot_k, kaxis_chunk[idx_in_chunk])
                push!(plot_E, (evals[ib] - εF_sc) * 27.2114) # Ha -> eV
                push!(plot_W, weights[ib])
            end
        end
    end
    
    band_data = nothing
    basis_bands = nothing
    GC.gc()
end

# ==============================================================================
# 6. Plotting
# ==============================================================================
println("\n[4/4] Generating Combined Plot...")

fig = Figure(size = (1200, 600))

# -- Panel 1: Band Structure --
ax_bands = Axis(fig[1, 1], 
    title = "Unfolded Bands: Doped Si (Si63 B1)",
    ylabel = "Energy (eV - Ef)",
    xlabel = "Wavevector",
    xticks = (x_ticks, x_labels)
)

sc = scatter!(ax_bands, plot_k, plot_E, 
    color = plot_W, 
    colormap = :turbo,
    markersize = 4,
    colorrange = (0, 1),
    transparency = true
)

hlines!(ax_bands, [0.0], color=:gray, linestyle=:dash)
vlines!(ax_bands, x_ticks, color=:gray, linewidth=0.5)
ylims!(ax_bands, -13, 8) 

# -- Panel 2: DOS --
ax_dos = Axis(fig[1, 2], 
    title = "Total DOS (P-Type)",
    xlabel = "DOS (states/eV)",
    ylabel = "Energy (eV - Ef)", 
)

lines!(ax_dos, dos_values, dos_energies_eV, color=:firebrick2, linewidth=2, label="Doped")
hlines!(ax_dos, [0.0], color=:gray, linestyle=:dash)

linkyaxes!(ax_bands, ax_dos)
hideydecorations!(ax_dos, grid=false)
ylims!(ax_dos, -13, 8) 
xlims!(ax_dos, 0, nothing)

Colorbar(fig[1, 3], sc, label="Spectral Weight (Si Character)")
colsize!(fig.layout, 1, Relative(2/3))
colsize!(fig.layout, 2, Relative(1/3))

save("Si_Doped_Unfolded.png", fig)
println("Done. Saved to 'Si_Doped_Unfolded.png'.")

t_end = time()
t_total = t_end - t_start
println("-------------------------------------------------------")
@printf "Total Wall Clock Time: %.2f seconds (%.2f minutes)\n" t_total (t_total/60)
println("-------------------------------------------------------")