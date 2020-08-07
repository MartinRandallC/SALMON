using Distributions
using Random

using HMMBase
using HDPHMM
using RTTHMM

import RTTHMM.Priors
import StatsBase: mad

"""
Predict the mixture components associated to observations.
"""
function predict(mixture::MixtureModel, observations)
    map(idx -> idx[2], argmax(hcat([logpdf.(d, observations) for d in components(mixture)]...), dims=2))[:,1]
end

"""
Serialize a Gaussian mixture model.
"""
function to_dict(mixture::HDPHMM.MixtureModel)
    return Dict(
        :weights   => mixture.prior.p,
        :means     => [x.μ for x in mixture.components],
        :variances => [x.σ^2 for x in mixture.components]
    )
end

"""
Serialize an HMM.
"""
function to_dict(hmm::HMM)
    return Dict(
        :transmat => permutedims(hmm.π), # Important to transpose to read in the right order in Python
        :states => [(:GaussianMixture, to_dict(d)) for d in hmm.D]
    )
end

"""
Re-estimate a mixture model using a robust estimator for
the mean and the variance of the normal distribution.
"""
function refit_mixture_robust(mixture, observations)
    comps = Normal[]
    prior = Float64[]

    observations = collect(skipmissing(observations))

    states = predict(mixture, observations)

    for state in unique(states)
        obs = observations[states .== state]
        
        std = mad(obs, normalize=true)/0.67449
        
        if std == 0.0
            println("WARN: std = 0, length(obs) = $(length(obs))")
            std = 1e-3
        end
        
        #c = fit_mle(Normal, obs)
        c = Normal(median(obs), std)
        p = length(obs) / length(observations)
    
        push!(comps, c)
        push!(prior, p)
    end

    # TODO: Print kl distance
    # https://ieeexplore.ieee.org/abstract/document/4218101
    # Approximating the Kullback Leibler Divergence Between Gaussian Mixture Models

    return MixtureModel(comps, prior)
end

"""
Convert the sampler state to an HMM and re-estimate the transition matrix.
"""
function to_hmm(obs, seq::Vector{Int64}, model::BlockedSamplerState; refit_states=false)
    mapping, transmat = compute_transition_matrix(seq)
    
    # Re-index
    seq = [mapping[x] for x in seq]
    states = Dict()
    for (old_idx, new_idx) in mapping
        states[new_idx] = model.obs_model.mixtures[old_idx][1]
    end

    new_states = Distribution{Univariate}[]
    for new_idx in 1:size(transmat)[1]
        mixture = states[new_idx]
        state_obs = obs[seq .== new_idx]
        if refit_states && (length(state_obs) > 0)
            mixture = refit_mixture_robust(mixture, state_obs)
        end
        push!(new_states, mixture)
    end
 
    return HMM(transmat, new_states)
end

function fit_model(observations; iterations=300, seed=2019, verbose=true)
    Random.seed!(seed)
    init_state = Priors.initial_state(Priors.GaussianDPMM(), observations)
    seqs, comps, states = HDPHMM.run(init_state, observations, iterations, verbose = verbose)

    sample_idx = RTTHMM.select_sample(seqs)
    seq = seqs[sample_idx,:]
    model = states[sample_idx]

    return seq, model
end