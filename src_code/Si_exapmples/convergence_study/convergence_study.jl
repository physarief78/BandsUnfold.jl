#!/usr/bin/env julia
# si_sc_convergence_final.jl
#
# CONVERGENCE STUDY PIPELINE (Memory Optimized):
# System: Silicon 2x2x2 Supercell (64 Atoms)
# 1. E_cut Convergence (10 -> 22 Ha)
# 2. K-grid Convergence (1 -> 3)
# 3. Aggressive Garbage Collection to stay under 20GB

using LinearAlgebra
using DFTK
using PseudoPotentialData
using CairoMakie
using Printf
using Statistics

# --- Global Settings ---
setup_threading()
const t_start = time()

n_threads = Threads.nthreads()
println("Running on $n_threads threads.")

# ==============================================================================
# 1. Structure Definitions
# ==============================================================================
function build_supercell_si()
    a_ang = 5.43
    a_bohr = a_ang * 1.8897259886
    nx, ny, nz = 2, 2, 2
    fcc_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    basis_shift = [0.25, 0.25, 0.25]
    cubic_nodes = vcat(fcc_nodes, [p .+ basis_shift for p in fcc_nodes])
    positions = Vector{Vector{Float64}}()
    for i in 0:nx-1, j in 0:ny-1, k in 0:nz-1
        shift = [i, j, k]
        for atom_pos in cubic_nodes
            new_pos = (atom_pos .+ shift) ./ [nx, ny, nz]
            push!(positions, new_pos)
        end
    end
    lattice = diagm([nx * a_bohr, ny * a_bohr, nz * a_bohr])
    atoms = [ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")) for _ in positions]
    return lattice, atoms, positions
end

lattice, atoms, positions = build_supercell_si()
println("System built: Si 2x2x2 Supercell ($(length(positions)) atoms)")

# ==============================================================================
# 2. Convergence Parameters
# ==============================================================================

# Range for Energy Cutoff Study (Hartree) - Max 22
ecut_range = [10, 12, 14, 16, 18, 20, 22]

# Range for K-Point Grid Study - Max [3,3,3]
kgrid_range = [1, 2, 3] 

# Fixed parameters for the opposing study
const FIXED_KGRID = [2, 2, 2] 
const FIXED_ECUT  = 20.0      
const SCF_TOL     = 1e-5

# Storage for results
results_ecut = Float64[]
results_kgrid = Float64[]

# ==============================================================================
# 3. Study 1: Energy Cutoff Convergence
# ==============================================================================
println("\n[1/2] Starting E_cut Convergence Study (Fixed k=$FIXED_KGRID)...")
println("      E_cut (Ha) | Total Energy (Ha)     | Time (s)")
println("      ---------------------------------------------")

for E in ecut_range
    t_iter_start = time()
    
    # 1. Build Model & Basis
    model = model_DFT(lattice, atoms, positions; functionals=PBE(), temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut=E, kgrid=FIXED_KGRID)
    
    # 2. Run SCF
    scfres = self_consistent_field(basis; tol=SCF_TOL, mixing=LdosMixing(), callback=identity)
    
    # 3. Store Results
    etot = scfres.energies.total
    push!(results_ecut, etot)
    
    t_iter_end = time()
    dt = t_iter_end - t_iter_start
    
    @printf "      %2d         | %.8f       | %.2f\n" E etot dt
    
    # 4. AGGRESSIVE MEMORY CLEANUP
    scfres = nothing
    basis = nothing
    model = nothing
    GC.gc() 
end

# ==============================================================================
# 4. Study 2: K-Point Grid Convergence
# ==============================================================================
println("\n[2/2] Starting K-Grid Convergence Study (Fixed E_cut=$FIXED_ECUT)...")
println("      K-Grid     | Total Energy (Ha)     | Time (s)")
println("      ---------------------------------------------")

for k in kgrid_range
    t_iter_start = time()
    
    k_array = [k, k, k]
    
    # 1. Build Model & Basis
    model = model_DFT(lattice, atoms, positions; functionals=PBE(), temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut=FIXED_ECUT, kgrid=k_array)
    
    # 2. Run SCF
    scfres = self_consistent_field(basis; tol=SCF_TOL, mixing=LdosMixing(), callback=identity)
    
    # 3. Store Results
    etot = scfres.energies.total
    push!(results_kgrid, etot)
    
    t_iter_end = time()
    dt = t_iter_end - t_iter_start
    
    @printf "      [%d,%d,%d]    | %.8f       | %.2f\n" k k k etot dt
    
    # 4. AGGRESSIVE MEMORY CLEANUP
    scfres = nothing
    basis = nothing
    model = nothing
    GC.gc()
end

# ==============================================================================
# 5. Plotting Results (FIXED)
# ==============================================================================
println("\nGenerating Plots...")

fig = Figure(size = (1000, 500))

# --- Calculate Delta E (per atom, in meV) ---
n_atoms = length(positions)
ref_E_ecut = results_ecut[end]
delta_E_ecut = [abs(e - ref_E_ecut) * 27211.4 / n_atoms for e in results_ecut] # Ha -> meV

ref_E_kgrid = results_kgrid[end]
delta_E_kgrid = [abs(e - ref_E_kgrid) * 27211.4 / n_atoms for e in results_kgrid] # Ha -> meV

# CRITICAL FIX: Replace exactly 0.0 with NaN so the log scale doesn't panic
replace!(delta_E_ecut, 0.0 => NaN)
replace!(delta_E_kgrid, 0.0 => NaN)

# --- Panel 1: E_cut ---
ax1 = Axis(fig[1, 1], 
    title = "E_cut Convergence (Si 64 Atom SC)",
    xlabel = "Kinetic Energy Cutoff (Ha)",
    ylabel = "|E - E_final| (meV/atom)",
    yscale = log10
)

# Plot lines and markers separately
lines!(ax1, ecut_range, delta_E_ecut, color=:blue, linewidth=2)
scatter!(ax1, ecut_range, delta_E_ecut, color=:blue, markersize=10)

# Add reference tolerance line
hlines!(ax1, [1.0], color=:red, linestyle=:dash, label="1 meV/atom")
axislegend(ax1)

# --- Panel 2: K-grid ---
ax2 = Axis(fig[1, 2], 
    title = "K-Grid Convergence (Si 64 Atom SC)",
    xlabel = "N (for NxNxN grid)",
    ylabel = "|E - E_final| (meV/atom)",
    yscale = log10,
    xticks = kgrid_range
)

lines!(ax2, kgrid_range, delta_E_kgrid, color=:orange, linewidth=2)
scatter!(ax2, kgrid_range, delta_E_kgrid, color=:orange, markersize=10, marker=:rect)

hlines!(ax2, [1.0], color=:red, linestyle=:dash)

save("Si_SC_Convergence_Study.png", fig)
println("Done. Saved to 'Si_SC_Convergence_Study.png'.")