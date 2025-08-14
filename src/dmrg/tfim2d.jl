using ITensors
using ITensorMPS
using DataFrames
using CSV

function finite_scale_tfim(N, kappa)
    """
    Compute properties of a 2D TFIM with OBC using DMRG.
    
    Args:
    N (Int): Linear size of the 2D lattice (N x N)
    kappa (Float64): Strength of the transverse field
    
    Returns:
    Tuple: (energy, avg_magz, avg_magx, magz_squared, Czz)
    """
    # Make an array of N^2 Index objects with spin 1/2 sites
    sites = siteinds("S=1/2", N*N)
    
    J = 1.0 # spin coupling
    
    # Paulis
    sigx = [0 1; 1 0]
    sigy = [0 -im; im 0]
    sigz = [1 0; 0 -1]

    # Define the Hamiltonian
    os = OpSum()
    
    # ZZ terms (nearest neighbors in 2D)
    for i in 1:N
        for j in 1:N
            site = (i-1)*N + j
            if j < N  # horizontal bonds
                os += -J, sigz, site, sigz, site+1
            end
            if i < N  # vertical bonds
                os += -J, sigz, site, sigz, site+N
            end
        end
    end
    
    # Transverse field terms
    for j in 1:N*N
        os += -kappa, sigx, j
    end
    
    H = MPO(os, sites)
    
    # DMRG parameters
    nsweeps = 12
    maxdim = [10,50,100,200,400,800,1200,1600,2000,2400,2800,3000]
    cutoff = [1E-15]
    
    # Initial state
    psi0 = randomMPS(sites, 2)
    
    # Run DMRG
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    
    # Compute observables
    magz = expect(psi, sigz)
    avg_magz = sum(magz) / (N*N)
    
    magx = expect(psi, sigx)
    avg_magx = sum(magx) / (N*N)


    Czz = correlation_matrix(psi, sigz, sigz)
    magz_squared = sum(Czz) / (N*N)^2
    
    Cxx = correlation_matrix(psi, sigx, sigx)
    magx_squared = sum(Cxx) / (N*N)^2


    return energy, magz, avg_magz, magx, avg_magx, magz_squared, Czz, magx_squared, Cxx
end

# Main execution
let
    df = DataFrame("N" => Int[], "kappa" => Float64[], "energy" => Float64[], 
                   "magz" => Vector{Float64}[], "avg_magz" => Float64[], "magx" => Vector{Float64}[], "avg_magx" => Float64[], 
                   "magz_squared" => Float64[], "Czz" => Matrix{Float64}[],
                   "magx_squared" => Float64[], "Cxx" => Matrix{Float64}[])
    
    system_sizes = [4,5,6,7,8,9,10]  # Reduced sizes for 2D
    kappa_range = [3.04438]
    
    for N in system_sizes
	
	open("out_dmrg.txt","a") do io
   	   println("N = ",N)
	end

        for kappa in kappa_range
	
	   open("out_dmrg.txt","a") do io
	      println("kappa = ", kappa)
	   end
	   
	   energy, magz, avg_magz, magx, avg_magx, magz_squared, Czz, magx_squared, Cxx = finite_scale_tfim(N, kappa)
           # Add data to dataframe
           push!(df, (N, kappa, energy, magz, avg_magz, magx, avg_magx, magz_squared, Czz, magx_squared, Cxx))
        
   	end
    end
    
    CSV.write("system_size_scaling_k=critical.csv", df)
end
