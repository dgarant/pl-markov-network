import numpy as np
import scipy as sp
from scipy import misc as spmisc
from scipy import optimize as spoptim
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

def mle_objective_and_grad(x1, x2, y1, y2, weights):
    # Objective function (negative log-likelihood) and 
    # gradient for MRF maximum likelihood estimation
    
    # negative LL : [sum_{i in data} weights * potentials(i)] - [1/|data| log sum_{j in joint configs} exp(weights * potentials(j))]
    # negative LL grad: [sum_{j in joint configs} P(j|weights) * potentials(j)] - [1/|data| sum_{i in data} potentials(i)]
    #   intuitively, the gradient represents the discrepancy between the potentials we would expect given the parameters, 
    #       and the potentials we saw in the data

    numerator = log_prob_numerator(x1, x2, y1, y2, weights)
    x1p, x2p, y1p, y2p = flatmesh([0, 1], [0, 1], [0, 1], [0, 1])
    pterms = log_prob_numerator(x1p, x2p, y1p, y2p, weights)
    partition = spmisc.logsumexp(pterms)

    grad = (np.sum(np.exp(pterms) / np.exp(pterms).sum() * potentials(x1p, x2p, y1p, y2p), axis=1) - 
            np.sum((1.0 / x1.shape[0]) * potentials(x1, x2, y1, y2), axis=1))

    return (np.sum(x1.shape[0] * partition) - np.sum(numerator), grad)

def mple_objective_and_grad(x1, x2, y1, y2, weights):
    # Objective function (negative log-pseudo-likelihood) and gradient for 
    # MRF maximum pseudo-likelihood estimation
    npoints = x1.shape[0]

    data_cols = (x1, x2, y1, y2)
    def gen_configs(vindex, val):
        return (data_cols[i] if i != vindex else np.repeat(val, npoints) for i in range(4))

    ll_terms = []
    grad_terms = []
    masks = potential_mask()
    dpotentials = potentials(x1, x2, y1, y2)
    observed_pots_expectation = np.mean(dpotentials, axis=1)
    for vindex in range(4):

        mask = masks[vindex, :]
        masked_dpots = np.multiply(mask[:, np.newaxis], dpotentials)
        data_term = np.dot(weights, masked_dpots)

        masked_wpots = []
        ppotentials = []
        un_probs = []
        val_indices = []
        for val in [0, 1]:
            val_points = gen_configs(vindex, val)
            val_indices.extend(np.repeat(val, npoints))
            cur_potentials = potentials(*val_points)
            ppotentials.append(cur_potentials)
            cur_masked_pots = np.multiply(mask[:, np.newaxis], cur_potentials)

            cur_wpots = np.dot(weights, cur_masked_pots)
            masked_wpots.append(cur_wpots)
            un_probs.append(np.exp(np.dot(weights, cur_potentials)))

        all_un_probs = np.hstack(un_probs)
        all_probs = np.divide(un_probs, np.sum(un_probs, axis=0))
        all_masked_wpots = np.vstack(masked_wpots)

        expected_pots_given_params = np.sum(np.multiply(all_probs[:, np.newaxis, :], ppotentials), axis=0)
        assert(expected_pots_given_params.shape == (len(weights), npoints))
        potential_term = spmisc.logsumexp(all_masked_wpots, axis=0)

        ll_terms.append(data_term - potential_term)
        # take mean across data points
        gradient = np.mean(expected_pots_given_params - observed_pots_expectation[:, np.newaxis], axis=1)
        grad_terms.append(gradient)

    assert(np.array(ll_terms).shape == (4, npoints))
    assert(np.array(grad_terms).shape == (4, len(weights)))

    return (- (1.0 / npoints) * np.sum(ll_terms), np.sum(grad_terms, axis=0))

def log_prob_numerator(x1, x2, y1, y2, weights):
    return np.dot(weights, potentials(x1, x2, y1, y2))

def potentials(x1, x2, y1, y2):
    # re-code as -1/1
    x1p = (x1 - 1) + x1
    y1p = (y1 - 1) + y1
    x2p = (x2 - 1) + x2
    y2p = (y2 - 1) + y2
    return np.array([x1p * y1p, x2p * y2p, y1p * y2p, y1p, y2p])

def potential_mask():
    # returns an N x M matrix where N is the number of variables and M is the number of potential functions,
    # indicating whether a given potential is used by a given variable (in order x1, x2, y1, y2)
    return np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 1, 1],
    ])

def true_prob(x1, x2, y1, y2):
    return ((0.5) * (0.5) * 
        ((y1 == x1) * 0.8 + (1 - (y1 == x1)) * 0.2) * 
        ((y2 == x2) * 0.8 + (1 - (y2 == x2)) * 0.2)
        )

def flatmesh(*args):
    grid = np.meshgrid(*args)
    return [e.flatten() for e in grid]

def main():
    nsamp = 10000
    x1 = np.random.binomial(1, 0.5, nsamp)
    y1 = np.random.binomial(1, x1 * 0.8 + (1 - x1) * 0.2, nsamp)

    x2 = np.random.binomial(1, 0.5, nsamp)
    y2 = np.random.binomial(1, x2 * 0.8 + (1 - x2) * 0.2, nsamp)

    initweights = np.random.uniform(0, 1.0, 5)

    mle_result = spoptim.minimize(lambda weights: mle_objective_and_grad(x1, x2, y1, y2, weights), initweights, jac=True)
    print("-----------------------")
    print("MLE result:")
    print("-----------------------")
    print(mle_result)

    x1p, x2p, y1p, y2p = flatmesh([0, 1], [0, 1], [0, 1], [0, 1])
    mle_pterms = log_prob_numerator(x1p, x2p, y1p, y2p, mle_result.x)
    mle_probs = np.exp(mle_pterms) / sum(np.exp(mle_pterms))

    mple_result = spoptim.minimize(lambda weights: mple_objective_and_grad(x1, x2, y1, y2, weights), initweights, jac=True, tol=1e-3)
    print("-----------------------")
    print("MPLE result:")
    print("-----------------------")
    print(mple_result)
    mple_pterms = log_prob_numerator(x1p, x2p, y1p, y2p, mple_result.x)
    mple_probs = np.exp(mple_pterms) / sum(np.exp(mple_pterms))

    true_lik = true_prob(x1p, x2p, y1p, y2p)

    np.set_printoptions(suppress=True)
    print(np.column_stack((mle_probs, mple_probs, x1p, x2p, y1p, y2p, true_lik)))
    
if __name__ == "__main__":
    main()
