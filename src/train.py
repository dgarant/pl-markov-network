import numpy as np
import scipy as sp
from scipy import misc as spmisc
from scipy import optimize as spoptim
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from scipy import stats

def main()
    # read data and configuration, build potentials, optimize
    pass

def mple_objective_and_grad(data):
    # Objective function (negative log-pseudo-likelihood) and gradient for 
    # MRF maximum pseudo-likelihood estimation
    # 
    # negative PLL : [sum_{i in data} weights * potentials(i)] - [1/|data| log sum_{v in vars} sum_{j in domain(v)} exp(weights * potentials(j))]
    # negative PLL grad: [sum_{j in joint configs} P(j|weights) * potentials(j)] - [1/|data| sum_{i in data} potentials(i)]
    #   intuitively, the gradient represents the discrepancy between the potentials we would expect given the parameters, 
    #       and the potentials we saw in the data

    numerator = log_prob_numerator(x1, x2, y1, y2, weights)
    npoints = x1.shape[0]

    # pseudo-partition calculation
    zeros = np.repeat(0, npoints)
    ones = np.repeat(1, npoints)
    data_cols = (x1, x2, y1, y2)
    def gen_configs(vindex):
        data0 = (data_cols[i] if i != vindex else zeros for i in range(4))
        data1 = (data_cols[i] if i != vindex else ones for i in range(4))
        return np.hstack((np.vstack(data0), np.vstack(data1)))
    eval_data = np.hstack((gen_configs(v) for v in range(4)))
    pots = potentials(*eval_data)

    prob_terms = np.dot(weights, pots)
    probs = np.exp(prob_terms) / np.exp(prob_terms).sum()

    ppartition = spmisc.logsumexp(prob_terms)

    grad = (np.sum(probs * pots, axis=1) - 
            np.sum((1.0 / npoints) * potentials(x1, x2, y1, y2), axis=1))

    return (np.sum(npoints * ppartition) - np.sum(numerator), grad)

def log_mple_grad(data, params):
    # gradient of pseudo-likelihood wrt parameters
    pass

def local_log_partition(index, data, weights):
    # computes the "local" partition function for a given variable, 
    # given on connected variables and current weights

    

def build_potentials(data, adj):
    """ 
        Creates a dictionary of potential functions, 
        indexed by the column number for node potentials and by the pair of 
        indices for pairwise potentials (smaller index first).
        Each potential function is a kernel density estimator.

        Arguments:
            data -- n x p matrix, where n is the number of measurements and p is the number of variables
            adj --  p x p matrix providing the structure of the network
    """
    kde_dict = dict()
    nonzeros = np.nonzero(adj)
    for r in range(len(nonzeros[0])):
        i, j = (nonzeros[0][r], nonzeros[0][i])

        if i == j:
            kde_dict[i] = stats.gaussian_kde(data[:, i], bw_method="scott")
        elif i < j:
            kde_dict[(i, j)] = stats.gaussian_kde(data[:, [i, j]], bw_method="scott")
        else:
            kde_dict[(j, i)] = stats.gaussian_kde(data[:, [j, i]], bw_method="scott")

    return kde_dict


if __name__ == "__main__":
    main()

