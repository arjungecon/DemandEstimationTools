using LinearAlgebra
using Distributions
using ForwardDiff
using CSV
using DataFrames




function GInvertShares( X::Matrix{<:Real}, s::Vector{<:Real}, I::Int64,
                        σ::Vector{<:Real}, ζ::Matrix{<:Real}, J::Int64)

    delPrev = zeros(J)
    sHat = log.(PredictShares( delPrev, X, numConsumer, σ, ζ))
    sTrue = log.(s)
    diff = 1e5
    #This doesn't need to be so tight
    while( diff > 1e-8)
        delNew = delPrev + sTrue - sHat
        diff = maximum( abs.( delNew - delPrev))
        
        delPrev = delNew
        sHat = log.(PredictShares( delPrev, X, numConsumer, σ, ζ))
    end

    #Now we do a Newton Step. Need the log-jacobian.
    #Since its the log jacobian we divide by PredictShares() which is exp( sHat)
    sJac = BuildXiJac( X, delPrev, I, σ, ζ, J ) ./ exp.( sHat )
    sHat = log.(PredictShares( delPrev, X, numConsumer, σ, ζ))
    while( diff > 1e-14)
        delNew = delPrev - sJac \ (sHat - sTrue)
        diff = maximum( abs.( sHat - sTrue))
        #println( diff)
        delPrev = delNew
        sJac = BuildXiJac( X, delPrev, I, σ, ζ, J ) ./ exp.( sHat )
        sHat = log.(PredictShares( delPrev, X, numConsumer, σ, ζ))
    end

    return delPrev
end




function GetDelta( X::Matrix{<:Real}, s::Vector{<:Real}, 
                numConsumer::Int64, σ::Vector{<:Real}, ζ::Matrix{<:Real},
                   J::Vector{Int64}, tMap)
    # This function just uses tMap (which maps from market to rows in the CSV)
    # To get delta for every market, and then vertically concatenates it.
    xiVec = Vector{Real}(undef,N);
    for t in 1:T
        #println(t)
        xMat = X[tMap[t],:]
        sSel = s[tMap[t]]
        
        xiVec[tMap[t]] = GInvertShares( xMat, sSel, numConsumer, σ, ζ, J[t])
    end
    return xiVec
end

function GetBeta( X::Matrix{<:Real}, Z::Matrix{<:Real}, W::Matrix{<:Real}, δ::Vector{<:Real})
    LHS = Hermitian(X'*Z*W*Z'*X)
    F = cholesky( LHS )

    #Weird type conflict here where RHS would be Vector{Any} because delta <: Real
    #Static cast fixes it. 
    RHS = convert( Vector{Real}, X' * Z * W * Z' * delta)

    #Cholesky at work here
    β = F.U \ (F.L \ RHS)
    return β
end



function Objective( X::Matrix{<:Real}, Z::Matrix{<:Real}, s::Vector{<:Real},
                    numConsumer::Int64, σ::Vector{<:Real}, ζ::Matrix{<:Real},
                    J::Vector{Int64}, tMap, W::Matrix{<:Real})

    delta = GetDelta( X, s, numConsumer, σ, ζ, J, tMap)

    β = GetBeta( X, Z, W, delta)
    

    xi = delta - X*β
    ortho = Z'*xi

    return ortho'*W*ortho 
    
end



function MarketSpecificJac(X::Matrix{<:Real}, ξ::Vector{<:Real},
                           β::Vector{<:Real}, I::Int64,
                           σ::Vector{<:Real}, ζ::Matrix{<:Real}, J::Int64)
    # A = mean( IndBuildJac( IndPredictShares( X, ξ, β + σ .* ζ[i,:]), J )*X
    #           for i in 1:I )
    B = mean( IndBuildJac( IndPredictShares( X, ξ, β + σ .* ζ[i,:]), J )*X*diagm(ζ[i,:])
              for i in 1:I )[:,[2,3]]
    C = BuildXiJac(X, ξ, β, I, σ, ζ, J)

    return -C \ B
end


function Gradient( X::Matrix{<:Real}, Z::Matrix{<:Real}, s::Vector{<:Real}, 
                    numConsumer::Int64, σ::Vector{<:Real}, ζ::Matrix{<:Real},
                    J::Vector{Int64}, tMap, W::Matrix{<:Real})

    delta = GetDelta( X, s, numConsumer, σ, ζ, J, tMap)
    β = GetBeta( X, Z, W, delta)

    xi = delta - X*β
    
    Jθ = vcat( [MarketSpecificJac( X[tMap[t],:], xi[tMap[t]], β, numConsumer, σ, ζ, J[t]) for t in 1:T]... );

    return 2*(Z'*Jθ)'W*(Z'*xi)
    
   
end




function IndPredictShares( δ::Vector{<:Real} )
    vMax = maximum(δ)

    num = exp.(δ .- vMax)
    return num ./ (exp(-vMax) + sum(num))
end

function PredictShares( δ::Vector{<:Real},  X::Matrix{<:Real}, I::Int64,
                        σ::Vector{<:Real}, ζ::Matrix{<:Real})
    return mean( IndPredictShares( δ +  X * (σ .* ζ[i,:])) for i in 1:I)
end


function PredictShares( X::Matrix{<:Real}, ξ::Vector{<:Real},  β::Vector{<:Real}, I::Int64,
                        σ::Vector{<:Real}, ζ::Matrix{<:Real})
    return mean( IndPredictShares( X, ξ, β + σ .* ζ[i,:]) for i in 1:I)
end


function BuildPJac( X::Matrix{<:Real}, ξ::Vector{<:Real},  β::Vector{<:Real}, I::Int64,
                   σ::Vector{<:Real}, ζ::Matrix{<:Real}, J::Int64)
    return mean( IndBuildJac( IndPredictShares( X, ξ, β + σ .* ζ[i,:]), J ).*(β[2] + σ[2]*ζ[i,2])
                  for i in 1:I)
end

function BuildXiJac( X::Matrix{<:Real}, ξ::Vector{<:Real},  β::Vector{<:Real}, I::Int64,
                   σ::Vector{<:Real}, ζ::Matrix{<:Real}, J::Int64)
    return mean( IndBuildJac( IndPredictShares( X, ξ, β + σ .* ζ[i,:]), J )
                  for i in 1:I)
end

function BuildXiJac( X::Matrix{<:Real}, δ::Vector{<:Real}, I::Int64,
                   σ::Vector{<:Real}, ζ::Matrix{<:Real}, J::Int64)
    return mean( IndBuildJac( IndPredictShares( δ +  X * (σ .* ζ[i,:])), J )
                  for i in 1:I)
end


function IndPredictShares( X::Matrix{<:Real}, ξ::Vector{<:Real},  β::Vector{<:Real} )

    δ = X*β + ξ
    vMax = maximum(δ)

    num = exp.(δ .- vMax)

    return num ./ (exp(-vMax) + sum(num))
end

function IndBuildJac( s::Vector{<:Real}, J::Int64 )
    Jac = Matrix{Real}(undef,J,J)#zeros(J,J)
    Jac .= 0.0
    for j in 1:J
        for k in 1:J
            if j == k
                Jac[j,j] = s[j]*(1-s[j])
            else
                Jac[j,k] = -s[j]*s[k]
            end
        end
    end
    return Jac
end



bigCSV = CSV.File( "psetOne.csv" )  |> DataFrame

T = length(unique( bigCSV[:,1]))
K = 6
numConsumer = 25
ζ = rand( Normal(), numConsumer, K)

X = convert(Matrix{Float64}, bigCSV[:,2:(K+1)] );
Z = convert(Matrix{Float64}, bigCSV[:,8:11]);
s = convert(Vector{Float64}, bigCSV[:,end]);



N = size(bigCSV,1)

# tMap[t] returns all rows in bigCSV that have market = t
tMap = [findall( x->bigCSV[x,1]==t, 1:N ) for t in 1:T]

# This gives the size of each market
J = [size(tMap[t],1) for t in 1:T]

W = inv(newZ'*newZ)


f(x) = Objective( X, newZ, bigCSV[:,end], numConsumer,
                  [0.0, x[1], x[2], 0.0, 0.0, 0.0],
                  ζ, J, tMap, W)



g(x) = Gradient( X, newZ, bigCSV[:,end], numConsumer,
                  [0.0, x[1], x[2], 0.0, 0.0, 0.0],
                 ζ, J, tMap, W)

# From here onwards is just KNITRO stuff. For more info check the
# examples in the KNITRO.jl github.
function callbackEvalF(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    evalResult.obj[1] = f(x)
    

    return 0
end

function callbackEvalG!(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x

    grad = g(x)
    # Evaluate gradient of nonlinear objective
    for i in 1:length(grad)
        evalResult.objGrad[i] = grad[i]
    end
    
    return 0
end



objGoal = KTR_OBJGOAL_MINIMIZE
n = 2
x_L = repeat([-KTR_INFBOUND], n)
x_U = repeat([KTR_INFBOUND], n)

kc = KNITRO.KN_new()
KNITRO.KN_add_vars(kc, n)

KNITRO.KN_set_var_lobnds(kc, x_L)
KNITRO.KN_set_var_upbnds(kc, x_U)
KNITRO.KN_set_var_primal_init_values(kc, zeros(2) )
KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE)


cb = KNITRO.KN_add_objective_callback(kc, callbackEvalF)

KNITRO.KN_set_cb_grad(kc, cb, callbackEvalG!)

nStatus = KNITRO.KN_solve(kc)
nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)


