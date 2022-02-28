##

# I02: problem set 2, exercise 2
# Clara Kyung and Arjun Gopinath

##

# Preparing environment ########################################################
using Pkg
# Pkg.activate(".") # Create new environment in this folder 

# Pkg.add(["Distributions", "Random"])
# Pkg.add("StatsBase")
# Pkg.add("StatsFuns")
# Pkg.add(["Plots","StatsPlots"])
# Pkg.add(["DataFrames","DataFramesMeta","Chain"]) # don't need?
# Pkg.add("Optim")
# Pkg.add("CSV")

# Pkg.instantiate() 

using LinearAlgebra
using Distributions
using Random
using StatsBase
using StatsFuns
using Plots
using StatsPlots
using DataFrames
using Optim
using CSV

##  

# Set parameters ###############################################################

Random.seed!(819613)

T = 200 # number of markets
K = 30  # number of potential entrants

# parameters in the θ vector
true_α = 1
true_β = 2
true_δ = 6 
true_γ = 3 
true_ρ = 0.8

sd_η = 1 # N(0,1)
sd_ϵ = 1 # N(0,1)
sd_x = 1 # lognormal(0,1)
sd_z = 2 # N(0,2)

##  

# Generate and simulate data ###################################################

# function to get equilibrium entrant ids and nstar
function f_equil(T, K, X, Z, η, ϵ, α, β, δ, γ, ρ)

    # heterogeneous component of profits 
    phi = α.*Z .+ ρ.*η .+ sqrt(1-ρ^2).*ϵ
    # phi = true_α.*Z

    # create sorted kk matrix based on order of phi
    # this is a matrix of entrant IDs
    kk_sort = Array{Int64}(undef, K, T)
    Threads.@threads for t in 1:T 
        kk_sort[:,t] = sortperm(phi[:,t], rev=true)
    end # for

    # number of potential entrants in each mkt
    N = K    

    # compute profits for each entrant i in mkt t for each possible n 
    piN = zeros(K, T, N)
    for i in 1:K, t in 1:T, n in 1:N
        piN[i,t,n] = γ .+ β.*X[i,t] .- δ*log(n) .+ phi[i,t]
    end # for

    # number of firms with positive profits for each market and each possible n
    positive_piN = sum((piN .> 0), dims=1)

    # equilibrium number of firms in each market
    nstar = zeros(Int64, 1, T) 
    for t in 1:T, n in 1:N 
        if positive_piN[1,t,n] >= n 
            nstar[1,t] = n
        end # if 
    end # if

    # get ids of entrants in each market. 
    entrants = zeros(Int64, K, T) 
    Threads.@threads for t in 1:T 
        for i in 1:nstar[1,t]
            entrants[kk_sort[i,t],t] = 1
        end # for 
    end # for

    return entrants,nstar
end # f_equil

# function to generate data 
function f_gendata(T, K, α, β, δ, γ, ρ, sd_η=1, sd_ϵ=1, sd_x=1, sd_z=2)
    
    X = repeat(transpose(rand(LogNormal(0,sd_x), T)), K, 1) # mkt-specific
    Z = rand(Normal(0,sd_z), K, T)  # firm-mkt specific 
    η = repeat(transpose(rand(Normal(0,sd_η), T)), K, 1) # common w/i mkt
    ϵ = rand(Normal(0,sd_ϵ), K, T) # firm-mkt specific

    entrants,nstar = f_equil(T, K, X, Z, η, ϵ, α, β, δ, γ, ρ)

    return X,Z,entrants,nstar
end # f_gendata

# function to simulate data - will use in the GMM obj fn
function f_simdata(T, K, X, Z, α, β, δ, γ, ρ; S::Int64=500)

    # arrays to store simulated pr and nstar
    pr_sim = Array{Float64}(undef, K, T, S) 
    nstar_sim = Array{Float64}(undef, 1, T, S) # float, since we will take avg. 

    # take S draws of eta and epsilon 
    eta = repeat(transpose(rand(Normal(0,sd_η), T)), K, 1, S) # common w/i mkt
    eps = rand(Normal(0,sd_ϵ), K, T, S) # firm-mkt specific 

    Threads.@threads for s in 1:S 
        η = eta[:,:,s]
        ϵ = eps[:,:,s]
        entrants,nstar = f_equil(T, K, X, Z, η, ϵ, α, β, δ, γ, ρ)
        pr_sim[:,:,s] = entrants
        nstar_sim[:,:,s] = nstar
    end # for

    # take average 
    pr_sim = (1/S).*sum(pr_sim, dims=3)[:,:,1] # K × T 
    nstar_sim = (1/S).*sum(nstar_sim, dims=3)[:,:,1] # 1 × T 

    return pr_sim, nstar_sim
end # f_simdata

# generate data
@time X,Z,entrants_obs,nstar_obs = 
    f_gendata(T, K, true_α, true_β, true_δ, true_γ, true_ρ)
# takes 0.003495 seconds

# simulate data (just testing it out)
@time pr_sim,nstar_sim = 
    f_simdata(T, K, X, Z, true_α, true_β, true_δ, true_γ, true_ρ; S=500)
# takes 0.539815 seconds

## 

# GMM estimation ###############################################################

# function to compute objective 
function f_gmmobj(T, K, X, Z, α, β, δ, γ, ρ, entrants_obs, nstar_obs)
    
    if ρ < 0
        ρ = 0
    elseif ρ > 1 
        ρ = 1 
    end # if-elseif

    pr_sim,nstar_sim = f_simdata(T, K, X, Z, α, β, δ, γ, ρ)
    
    ξ = entrants_obs - pr_sim # K × T 
    u = nstar_obs - nstar_sim # 1 × T

    # note X,Z are both K × T, but X is constant w/i col (mkt)
    # construct moments (expectations of ξX,uX,ξZ,uZ,ξ,u, averaged across mkts)
    mom1 = (1/(K*T)).*sum(ξ .* X)  
    mom2 = (1/(T)).*sum(u .* X[1,:]') 
    mom3 = (1/(K*T)).*sum(ξ .* Z) 
    mom4 = (1/(K*T)).*sum(repeat(u,K) .* Z) 
    mom5 = (1/(K*T)).*sum(ξ) 
    mom6 = (1/(T)).*sum(u) 

    mom = vcat(mom1, mom2, mom3, mom4, mom5, mom6) # 6 × 1
    W_inv = W_inv = Matrix{Float64}(I,6,6)

    # construct objective - make sure it's a scalar
    obj = (mom' * W_inv * mom)[1,1]

    return obj
end # f_gmmobj

# let's see how long it takes to run once
@time f_gmmobj(T, K, X, Z, true_α, true_β, true_δ, true_γ, true_ρ, 
    entrants_obs, nstar_obs)
# takes 0.562045 seconds

f(param) = f_gmmobj(T, K, X, Z, 
            param[1], param[2], param[3], param[4], param[5], 
            entrants_obs, nstar_obs)

myOptions = Optim.Options(x_tol = 5e-4 , f_tol = 5e-4, iterations = 300)

# just testing it out 
param_init = [1.1, 1.9, 5.8, 2.9, 0.7]
results = optimize(f, param_init, myOptions)
θ_hat = Optim.minimizer(results)

# hm, max. number of iterations is always reached.
# 300 iterations takes 426 seconds 

## 

# Repeat estimation for different initial values ###############################

param_init = [1.1 2.1 6.2 3.1 0.85; 
              0.9 1.9 5.8 2.9 0.75;
              1.5 2.5 7.0 4.0 0.95;
              0.5 1.5 5.0 2.0 0.65]
n_init = size(param_init, 1) # number of rows (sets of initial conditions)
n_hist = 20 # number of observations to put in each histogram 

# dim=1: each row is a different set of initial conditions
# dim=2: each column is a parameter
# dim=3: repeating estimation several times to get 
param_hat = Array{Float64}(undef, n_init, 5, n_hist)

# try running one set of initial values at a time instead of looping
# (bc the loop takes too long to run)
# first set of initial values 
@time Threads.@threads for h in 1:n_hist
    results = optimize(f, param_init[1,:], myOptions)
    param_hat[1,:,h] = Optim.minimizer(results)
end # for 
# 29962 seconds???? 

df_1 = DataFrame(param_hat[1,:,:]', [:alpha,:beta,:delta,:gamma,:rho])
CSV.write("output/est_param_1.csv", df_1)

# there must be a more efficient way of making these plots...
plot_1_α = plot(param_hat[1,1,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false)
plot!(param_hat[1,1,:], seriestype = :density, legend = false, xlabel = "α")

plot_1_β = plot(param_hat[1,2,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false)  
plot!(param_hat[1,2,:], seriestype = :density, legend = false, xlabel = "β") 

plot_1_δ = plot(param_hat[1,3,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[1,3,:], seriestype = :density, legend = false, xlabel = "δ")  

plot_1_γ = plot(param_hat[1,4,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[1,4,:], seriestype = :density, legend = false, xlabel = "γ") 

plot_1_ρ = plot(param_hat[1,5,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[1,5,:], seriestype = :density, legend = false, xlabel = "ρ")  

param_plot_1 = plot(plot_1_α, plot_1_β, plot_1_δ, plot_1_γ, plot_1_ρ, 
                    layout = 5, 
                    plot_title = "Initial conditions (α,β,δ,γ,ρ)=(1.1, 2.1, 6.2, 3.1, 0.85)", 
                    plot_titlefontsize = 12, size = (600,400))
savefig(param_plot_1, "output/param_plot_1.pdf")

# second set of initial values
@time Threads.@threads for h in 1:n_hist
    results = optimize(f, param_init[2,:], myOptions)
    param_hat[2,:,h] = Optim.minimizer(results)
end # for 
# 8787 seconds - more reasonable

df_2 = DataFrame(param_hat[2,:,:]', [:alpha,:beta,:delta,:gamma,:rho])
CSV.write("output/est_param_2.csv", df_2)

plot_2_α = plot(param_hat[2,1,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false)
plot!(param_hat[2,1,:], seriestype = :density, legend = false, xlabel = "α")

plot_2_β = plot(param_hat[2,2,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false)  
plot!(param_hat[2,2,:], seriestype = :density, legend = false, xlabel = "β") 

plot_2_δ = plot(param_hat[2,3,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[2,3,:], seriestype = :density, legend = false, xlabel = "δ")  

plot_2_γ = plot(param_hat[2,4,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[2,4,:], seriestype = :density, legend = false, xlabel = "γ") 

plot_2_ρ = plot(param_hat[2,5,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[2,5,:], seriestype = :density, legend = false, xlabel = "ρ")  

param_plot_2 = plot(plot_2_α, plot_2_β, plot_2_δ, plot_2_γ, plot_2_ρ, 
                    layout = 5, 
                    plot_title = "Initial conditions (α,β,δ,γ,ρ)=(0.9, 1.9, 5.8, 2.9, 0.75)", 
                    plot_titlefontsize = 12, size = (600,400))
savefig(param_plot_2, "output/param_plot_2.pdf")

# third set of initial values 
@time Threads.@threads for h in 1:n_hist
    results = optimize(f, param_init[3,:], myOptions)
    param_hat[3,:,h] = Optim.minimizer(results)
end # for 
# 9425 seconds

df_3 = DataFrame(param_hat[3,:,:]', [:alpha,:beta,:delta,:gamma,:rho])
CSV.write("output/est_param_3.csv", df_3)

plot_3_α = plot(param_hat[3,1,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false)
plot!(param_hat[3,1,:], seriestype = :density, legend = false, xlabel = "α")

plot_3_β = plot(param_hat[3,2,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false)  
plot!(param_hat[3,2,:], seriestype = :density, legend = false, xlabel = "β") 

plot_3_δ = plot(param_hat[3,3,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[3,3,:], seriestype = :density, legend = false, xlabel = "δ")  

plot_3_γ = plot(param_hat[3,4,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[3,4,:], seriestype = :density, legend = false, xlabel = "γ") 

plot_3_ρ = plot(param_hat[3,5,:], seriestype = :histogram, normalize = true, 
                bins = 8, legend = false) 
plot!(param_hat[3,5,:], seriestype = :density, legend = false, xlabel = "ρ")  

param_plot_3 = plot(plot_3_α, plot_3_β, plot_3_δ, plot_3_γ, plot_3_ρ, 
                    layout = 5, 
                    plot_title = "Initial conditions (α,β,δ,γ,ρ)=(1.5, 2.5, 7.0, 4.0, 0.95)", 
                    plot_titlefontsize = 12, size = (600,400))
savefig(param_plot_3, "output/param_plot_3.pdf")

# fourth set of initial values - didn't include in pset
@time Threads.@threads for h in 1:n_hist
    results = optimize(f, param_init[4,:], myOptions)
    param_hat[4,:,h] = Optim.minimizer(results)
end # for 

## 

# GMM objectives over a grid around true parameter #############################

# around α
f(param) = f_gmmobj(T, K, X, Z, 
            param, true_β, true_δ, true_γ, true_ρ, 
            entrants_obs, nstar_obs)
α_grid = range(-5, stop=5, step=0.25)
gmmobj_α = Array{Float64}(undef,length(α_grid))

@time for g in 1:length(α_grid)
    gmmobj_α[g] = f(α_grid[g])
end # for

plot_α = plot(α_grid, gmmobj_α, title = "GMM Objective as a Function of α", 
              xlabel = "α", ylabel = "Value of GMM objective", legend=false) # yes, it works!! 
savefig(plot_α, "output/gmm_alpha_plot.pdf") 

# around β 
f(param) = f_gmmobj(T, K, X, Z, 
            true_α, param, true_δ, true_γ, true_ρ, 
            entrants_obs, nstar_obs)
β_grid = range(-5, stop=5, step=0.25)
gmmobj_β = Array{Float64}(undef,length(β_grid))
for g in 1:length(β_grid)
    gmmobj_β[g] = f(β_grid[g])
end # for

plot_β = plot(β_grid, gmmobj_β, title = "GMM Objective as a Function of β", 
              xlabel = "β", ylabel = "Value of GMM objective", legend=false)
savefig(plot_β, "output/gmm_beta_plot.pdf") 

# around δ
f(param) = f_gmmobj(T, K, X, Z, 
            true_α, true_β, param, true_γ, true_ρ, 
            entrants_obs, nstar_obs)
δ_grid = range(2, stop=10, step=0.25)
gmmobj_δ = Array{Float64}(undef,length(δ_grid))
for g in 1:length(δ_grid)
    gmmobj_δ[g] = f(δ_grid[g])
end # for

plot_δ = plot(δ_grid, gmmobj_δ, title = "GMM Objective as a Function of δ", 
              xlabel = "δ", ylabel = "Value of GMM objective", legend=false)
savefig(plot_δ, "output/gmm_delta_plot.pdf") 

# around γ
f(param) = f_gmmobj(T, K, X, Z, 
            true_α, true_β, true_δ, param, true_ρ, 
            entrants_obs, nstar_obs)
γ_grid = range(-1, stop=7, step=0.25)
gmmobj_γ = Array{Float64}(undef,length(γ_grid))
for g in 1:length(γ_grid)
    gmmobj_γ[g] = f(γ_grid[g])
end # for

plot_γ = plot(γ_grid, gmmobj_γ, title = "GMM Objective as a Function of γ", 
              xlabel = "γ", ylabel = "Value of GMM objective", legend=false)
savefig(plot_γ, "output/gmm_gamma_plot.pdf") 

# around ρ
f(param) = f_gmmobj(T, K, X, Z, 
            true_α, true_β, true_δ, true_γ, param, 
            entrants_obs, nstar_obs)
ρ_grid = range(0, stop=1, step=0.05)
gmmobj_ρ = Array{Float64}(undef,length(ρ_grid))
for g in 1:length(ρ_grid)
    gmmobj_ρ[g] = f(ρ_grid[g])
end # for

plot_ρ = plot(ρ_grid, gmmobj_ρ, title = "GMM Objective as a Function of ρ", 
              xlabel = "ρ", ylabel = "Value of GMM objective", legend=false)
savefig(plot_ρ, "output/gmm_rho_plot.pdf") 
