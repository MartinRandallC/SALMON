import numpy as np

from numba import jit

from scipy.stats import norm
from scipy.special import logsumexp

from .jit import *


class GaussianMixture:
    "Stateless Gaussian mixture model."

    def __init__(self, weights, means, variances):
        assert len(weights) == len(means) == len(variances)
        self.weights = np.array(weights)
        self.means = np.array(means)
        self.variances = np.array(variances)
        self.mean = np.dot(weights, means)

    @property
    def n_components(self):
        return len(self.distributions)

    def pdf(self, x):
        "Likelihood of observation `x`."
        return jit_gmm_pdf(self.weights, self.means, self.variances, x)

    def rvs(self):
        "Sample the GMM."
        z = np.random.choice(self.n_components, p=self.weights)
        return jit_normal_rvs(self.means[z], self.variances[z])

    @classmethod
    def from_dict(cls, obj):
        "Deserialize the model."
        return cls(**obj)

    def to_dict(self):
        "Returns a serializable representation of the model."
        return {
            "weights": list(self.weights),
            "means": list(self.means),
            "variances": list(self.variances),
        }


class HMM:
    "Stateless hidden Markov model with arbitrary emissions."

    def __init__(self, transmat, states):
        assert transmat.shape[0] == transmat.shape[1] == len(states)
        self.transmat = transmat
        self.states = states

    @property
    def n_states(self):
        return len(self.states)

    @property
    def stationnary_dist(self):
        "Stationnary distribution of the underlying Markov chain."
        vals, vecs = np.linalg.eig(self.transmat.T)
        return vecs[:, vals == 1].ravel()

    def pdfs(self, x):
        "Likelihood of observation `x` in each state of the HMM."
        return np.array([state.pdf(x) for state in self.states])

    def sample(self, T):
        "Generate a trajectory of the HMM for `T` time steps."
        seq = np.zeros(T, np.int)
        obs = np.zeros(T)

        seq[0] = np.random.choice(self.n_states)
        obs[0] = self.states[seq[0]].rvs()

        for t in range(1, T):
            seq[t] = np.random.choice(self.n_states, p=self.transmat[seq[t - 1], :])
            obs[t] = self.states[seq[t]].rvs()

        return seq, obs

    @classmethod
    def from_dict(cls, obj):
        "Deserialize the model."
        transmat = np.array(obj["transmat"])
        states = [globals()[cls].from_dict(o) for cls, o in obj["states"]]
        return cls(transmat, states)

    def to_dict(self):
        "Returns a serializable representation of the model."
        return {
            "transmat": self.transmat.tolist(),
            "states": [
                (type(state).__name__, state.to_dict()) for state in self.states
            ],
        }


class HMMFilter:
    "Stateful HMM filter."

    def __init__(self, model, belief=None):
        if belief is None:
            belief = model.stationnary_dist

        if len(belief) == 0:
            # print('Failed to compute stationnary distribution. Will use uniform initial distribution.')
            belief = np.ones(model.n_states) / model.n_states

        self.model = model
        self.belief = belief

    def update(self, x):
        "Update the filter with observation `x`."
        self.belief = jit_hmm_filter(
            self.model.transmat, self.belief, self.model.pdfs(x)
        )

    def predict(self):
        "Update the filter by predicting."
        self.belief = jit_hmm_predict(self.model.transmat, self.belief)
