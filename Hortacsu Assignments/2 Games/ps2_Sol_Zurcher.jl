using LinearAlgebra
using Distributions



function logsum( v1,v2 )
    v0 = max( v1, v2)
    return v0 + log( exp( v1 - v0) + exp(v2 - v0))

    #return log( exp( v1) + exp(v2))
end


## Notation Here:
## cOne is the linear component of cost,
## cTwo is the quadratic term
## RC is the replacement cost
## tranProbNoReplace is the Transition matrix
## β is the discount factor
## states is a vector of the states (Corresponding to indices)
## EV is a vector containing the continuation values of not replacing the engine at some mileage
## nX is the size of states

## Data is stored as:
## mileCounter which is a vector of the mileage
## keeps which is a vector ∈ {0,1} on whether or not the bus is replaced


 


function Γ(β, EV, nX, RC, cOne, cTwo, tranProbNoReplace, states )

    c(miles) = -cOne*miles - cTwo*miles*miles

    costs = c.( states )
    return tranProbNoReplace*logsum.( costs + β*EV,
                   (c(0.0) - RC ) .+ β*EV[1] )
end



function DoContractionFast( EV, nX, RC, cOne, cTwo, tranProbNoReplace, β, states)
    c(miles) = -cOne*miles - cTwo*miles*miles
    EVNew = Vector{Real}(undef, nX)#copy( EV);
    normMe = 999.0
    while( normMe > 1e-2 )
        EVNew = Γ( β, EV, nX, RC, cOne, cTwo,
                   tranProbNoReplace,  states)
        normMeNew = maximum( abs.( EV - EVNew))
        if( abs( normMeNew / normMe - β ) < 1e-2)
            EV = EVNew
            break
        end
        #EV = copy(EVNew)
        EV = EVNew
        println(normMeNew / normMe)
        normMe = normMeNew
    end
    
    println("Give em the King-Werner")
    #normMe = maximum( abs.( EV - EVNew))
    #println( normMe)
    its = 0
    Xn = copy(EV)
    Yn = copy(EV)
    while( normMe > 1e-10 )

        EVNew = Γ( β, EV, nX, RC, cOne, cTwo,
                   tranProbNoReplace,  states)

        EVWerner = .5*Xn + .5*Yn
        
        pVec = 1.0 ./ (1.0 .+ exp.( -c.(states) .- RC - β*EVWerner .+
                                    c(0.0) .+ β*EVWerner[1]))

        A = β*diagm( pVec );
        A[:,1] += β*(1.0 .- pVec);

        ΓPrime = tranProbNoReplace * A;

        Xn = EV - ( I - ΓPrime) \ ( EV - EVNew)
        normMe = maximum( abs.( EV - Xn))
        println( normMe)

        #EV = copy(Xn)

        if normMe <= 1e-10
            EV = copy(Xn)
            break
        end
                
        its += 1

        EVNew = Γ( β, Xn, nX, RC, cOne, cTwo,
                   tranProbNoReplace,  states)

        Yn = Xn - ( I - ΓPrime) \ ( Xn - EVNew)
        normMe = maximum( abs.( Xn - Yn))
        
        println( normMe)
        EV = copy(Yn)
    end

    pVec = 1.0 ./ (1.0 .+ exp.( -c.(states) .- RC - β*EV .+
                                c(0.0) .+ β*EV[1]))

    A = β*diagm( pVec );
    A[:,1] += β*(1.0 .- pVec);

    ΓPrime = tranProbNoReplace * A;

    return EV,ΓPrime
        

    # if i % 100 == 0
    #     println(i)
    # end
end


function Likelihood( mileCounter, keeps, cOne, cTwo, RC, tranProbNoReplace, β, nX, states)
    EV = Vector{Real}(undef,nX);

    EV .= 0.0;
    
    EV,ΓPrime = DoContractionFast( EV, nX, RC, cOne, cTwo*1e-4, tranProbNoReplace, β, states);

    c(miles) = -cOne*miles - cTwo*miles*miles*1e-4
    pVec = 1.0 ./ (1.0 .+ exp.( -c.(states) .- RC - β*EV .+ c(0.0) .+ β*EV[1]))

    logLik = sum( log.( pVec[mileCounter[keeps]])) +
        sum(log.( 1.0 .- pVec[mileCounter[keeps .== 0]]))
       
    
    return -logLik
end

function Gradient( grad, mileCounter, keeps, cOne, cTwo, RC, tranProbNoReplace, β, nX, states)
    
    EV = Vector{Real}(undef,nX);

    EV .= 0.0;
    
    EV,ΓPrime = DoContractionFast( EV, nX, RC, cOne, cTwo*1e-4, tranProbNoReplace, β, states);

    c(miles) = -cOne*miles - cTwo*miles*miles*1e-4
    pVec = 1.0 ./ (1.0 .+ exp.( -c.(states) .- RC - β*EV .+ c(0.0) .+ β*EV[1]))

    # The gradient gets pretty bad. 
    A = -pVec .* states;
    B = -pVec .* states .* states .* 1e-4;
    C = -(1.0 .- pVec)

    ∂EV = (I - ΓPrime) \ tranProbNoReplace*hcat( A, B, C)

    # This allocates memory per state, not so well done.  Maybe a more
    # sophisticated way is to do this with some matrix multiplcation.
    Grad(state) = -pVec[state]*(1.0 - pVec[state])*[state - β*∂EV[state,1] + β*∂EV[1,1],
                                                   state*state*1e-4 - ∂EV[state,2] +
                                                   β*∂EV[1,2],
                                                   - β*∂EV[state,3] + β*∂EV[1,3] - 1.0]

    grad = -(sum( Grad.( mileCounter[keeps] ) ./ pVec[mileCounter[keeps]] ) - sum(Grad.( mileCounter[keeps .== 0] ) ./ (1.0 .- pVec[mileCounter[keeps .== 0]]) ))
    return grad
end


