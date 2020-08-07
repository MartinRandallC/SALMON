import numpy as np

from numba import jit


ϵ = np.finfo(float).eps


@jit(nopython=True)
def jit_normal_pdf(μ, σ2, x):
    norm = 1 / np.sqrt(2 * np.pi * σ2)
    return norm * np.exp(-((x - μ) ** 2 / (2 * σ2)))


@jit(nopython=True)
def jit_gmm_pdf(πs, μs, σ2s, x):
    pdf = 0.0
    for π, μ, σ2 in zip(πs, μs, σ2s):
        pdf += π * jit_normal_pdf(μ, σ2, x)
    return pdf


@jit(nopython=True)
def jit_normal_rvs(μ, σ2):
    "Fast only for a single sample (100x faster than scipy)"
    u1, u2 = 0.0, 0.0
    while u1 < ϵ or u2 < ϵ:
        u1 = np.random.rand()
        u2 = np.random.rand()
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z1 * np.sqrt(σ2) + μ


@jit(nopython=True)
def jit_hmm_filter(transmat, belief, likelihoods):
    belief = likelihoods * (transmat.T @ belief)
    return belief / np.sum(belief)


@jit(nopython=True)
def jit_hmm_predict(transmat, belief):
    return transmat.T @ belief
