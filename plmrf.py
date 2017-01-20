import sys
import numpy as np
import scipy as sp
from scipy import misc as spmisc
from scipy import optimize as spoptim
from scipy import stats
from scipy import interpolate as spinterp
from scipy import sparse as spsparse
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

class PotentialFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, dmap):
        """ 
            Evaluates the potential function at one or more points.

            Arguments:
                dmap: A map from variable name to an array-like type 
                      representing the values for that variable.
        """
        pass

    @abstractmethod
    def variables(self):
        """ Retrieves the set of variables used by the potential function """
        pass

class GaussianPotential(PotentialFunction):
    """ exp(-(v1 - v2)^2 / bandwidth) """

    def __init__(self, variables, bandwidth=None, samples=None, location=None):
        self.var_list = list(variables)
        self.location = location

        if not len(self.var_list) in [1, 2]:
            raise ValueError("The Gaussian product potential function does not support {0}-cliques.".format(len(self.var_list)))
        if len(self.var_list) == 1 and self.location is None:
            raise ValueError("The location parameter must be specified for univariate Gaussian potentials.")

        if bandwidth is None and samples is None:
            raise ValueError("One of 'bandwidth' or 'samples' " + 
                "(used to estimate bandwidth by rule-of-thumb) must be supplied.")
        elif not bandwidth is None:
            self.bandwidth = bandwidth
        else:
            # estimate bandwidth via Silverman's rule
            if len(self.var_list) == 2:
                sd = np.std(samples[self.var_list[0]] - samples[self.var_list[1]])
            else:
                sd = np.std(samples[self.var_list[0]] - self.location)
            npoints = len(samples[self.var_list[0]])
            self.bandwidth = 2.34 * sd * np.power(npoints, -1.0 / 5.6)

    def variables(self):
        return self.var_list
    
    def __call__(self, dmap):
        if len(self.var_list) == 1:
            return np.exp(-np.abs(dmap[self.var_list[0]] - self.location) / self.bandwidth)
        else:
            return np.exp(-np.abs(dmap[self.var_list[0]] - dmap[self.var_list[1]]) / self.bandwidth)

class IdentityPotential(PotentialFunction):
    def __init__(self, variables):
        self.var_list = list(variables)
        if len(self.var_list) != 1:
            raise ValueError("The identity potential function can only be used as a node potential.")

    def variables(self):
        return self.var_list

    def __call__(self, dmap):
        return dmap[self.var_list[0]]

class BinaryProductPotential(PotentialFunction):
    def __init__(self, variables):
        self.var_list = list(variables)
        if not len(self.var_list) in [1, 2]:
            raise ValueError("The binary product potential function does not support {0}-cliques.".format(len(self.var_list)))

    def variables(self):
        return self.var_list
    
    def __call__(self, dmap):
        # product of shifted values
        if len(self.var_list) == 1:
            vals = dmap[self.var_list[0]]
            return (vals - 1) + vals
        else:
            vals1 = dmap[self.var_list[0]]
            vals2 = dmap[self.var_list[1]]
            return ((vals1 - 1) + vals1) * ((vals2 - 1) + vals2)

class KernelDensityPotential(PotentialFunction):
    def __init__(self, variables, samples):
        """
            Creates the kernel density estimate potential function
            
            Arguments:
                dmap: A map from variable name to an array-like type 
                      representing the values for that variable
                variables:
                     Iterable of variables participating in the potential
        """
        self.var_list = list(variables)
        if not len(self.var_list) in [1, 2]:
            raise ValueError("The binary product potential function does not support {0}-cliques.".format(len(self.var_list)))
        data = np.array([samples[v] for v in self.var_list])
        self.kde = stats.gaussian_kde(data.flatten() if len(self.var_list) == 1 else data)

    def variables(self):
        return self.var_list

    def __call__(self, dmap):
        """ Evaluates the potential function at a set of points """
        data = np.array([dmap[v] for v in self.var_list])
        return self.kde(data.flatten() if len(self.var_list) == 1 else data)

class VariableDef(object):
    def __init__(self, name, ddomain=None, samples=None):
        self.name = name
        self.ddomain = ddomain
        if ddomain is None and samples is None:
            raise ValueError("For discrete data, you should supply 'ddomain'. " + 
                "For continuous data, you should supply 'samples'.")
        if not samples is None:
            self.int_points = stats.mstats.mquantiles(samples, np.linspace(0, 1, 50))

class LogLinearMarkovNetwork(object):
    
    def __init__(self, potential_funs, variable_spec):
        """
            Arguments:
                potential_funs: A sequence of PotentialFunction instances
                variable_spec: A sequence of VariableDef instances
        """
        self.potential_funs = potential_funs
        self.variable_spec = variable_spec
        
        # cosntructing a V x F sparse adjacency matrix where V 
        # is the number of variables and F is the number of factors
        varname_to_index = dict((v.name, i) for i, v in enumerate(self.variable_spec))
        adj_rows = []
        adj_cols = []
        for i, f in enumerate(self.potential_funs):
            for v in f.variables():
                adj_rows.append(varname_to_index[v])
                adj_cols.append(i)

        self.factor_adjacency = spsparse.csr_matrix((np.repeat(1, len(adj_rows)), (adj_rows, adj_cols)))

    def mle_objective_and_grad(self, dmap, weights, data_potentials=None, all_potentials=None):
        # Objective function (negative log-likelihood) and 
        # gradient for MRF maximum likelihood estimation
        
        # negative LL : [sum_{i in data} weights * potentials(i)] - [1/|data| log sum_{j in joint configs} exp(weights * potentials(j))]
        # negative LL grad: [sum_{j in joint configs} P(j|weights) * potentials(j)] - [1/|data| sum_{i in data} potentials(i)]
        #   intuitively, the gradient represents the discrepancy between the potentials we would expect given the parameters, 
        #       and the potentials we saw in the data

        npoints = len(dmap.itervalues().next())
        if data_potentials is None:
            data_potentials = self.potentials(dmap)
        if all_potentials is None:
            all_combos =  self.expand_joint(self.variable_spec)
            all_potentials = self.potentials(all_combos)

        varnames = [v.name for v in self.variable_spec]
        numerator = np.dot(weights, data_potentials)

        pterms = np.dot(weights, all_potentials)
        partition = spmisc.logsumexp(pterms)

        grad = (np.sum(np.exp(pterms) / np.exp(pterms).sum() * all_potentials, axis=1) - 
                np.sum((1.0 / npoints) * data_potentials, axis=1))

        return (np.sum(npoints * partition) - np.sum(numerator), grad)

    def fit_mle(self, dmap, initweights=None):
        if not initweights:
            initweights = np.random.uniform(0, 1.0, len(self.potential_funs))
        data_potentials = self.potentials(dmap)
        # make sure the data's all discrete
        for v in self.variable_spec:
            if v.ddomain is None:
                raise ValueError("Maximum likelihood estimation only supported for discrete data.")

        all_combos =  self.expand_joint(self.variable_spec)
        all_potentials = self.potentials(all_combos)

        result = spoptim.minimize(lambda weights: 
            self.mle_objective_and_grad(dmap, weights, 
                data_potentials=data_potentials, all_potentials=all_potentials), initweights, jac=True)
        self.weights = result.x
        return result

    def fit(self, dmap, initweights=None):
        if not initweights:
            initweights = np.random.uniform(0, 1.0, len(self.potential_funs))
        data_potentials = self.potentials(dmap)
        local_pots = self.get_local_partition_potentials(dmap)

        result = spoptim.minimize(lambda weights: 
            self.mple_objective_and_grad(dmap, weights, 
                data_potentials=data_potentials, 
                local_partition_potentials=local_pots), initweights, jac=True)
        self.weights = result.x
        return result

    def mple_local_partition_and_expected_potentials(self, dmap, vindex, weights, local_partition_potentials):
        """ Creates the local pseudo-likelihood partition function for a given variable """
        var = self.variable_spec[vindex]
        npoints = len(dmap.itervalues().next())
        mask = np.array(self.factor_adjacency[vindex, :].todense()).transpose()
        all_potentials = local_partition_potentials[var.name]
        weighted_potentials = np.dot(weights, np.multiply(mask, all_potentials))

        if not var.ddomain is None: # discrete
            all_probs = np.divide(np.exp(weighted_potentials), np.sum(np.exp(weighted_potentials), axis=0))
            local_partition = spmisc.logsumexp(weighted_potentials, axis=0)
            expected_pots_given_params = np.sum(np.multiply(all_probs[:, np.newaxis, :], all_potentials), axis=0)
            return (local_partition, expected_pots_given_params, all_probs)
        else:
            normalizer = sp.integrate.trapz(np.exp(weighted_potentials), var.int_points, axis=0)
            all_densities = np.divide(np.exp(weighted_potentials), normalizer)
            expected_pots_given_params = sp.integrate.trapz(np.multiply(all_densities[:, np.newaxis, :], all_potentials), var.int_points, axis=0)
            local_partition = np.log(normalizer)
            return (local_partition, expected_pots_given_params, all_densities)

    def get_local_partition_potentials(self, dmap):
        npoints = len(dmap.itervalues().next())

        def calc_potentials(vindex, val):
            dmap_prime = dict([(v.name, dmap[v.name]) 
                if i != vindex else (v.name, np.repeat(val, npoints)) for i, v in enumerate(self.variable_spec)])
            return self.potentials(dmap_prime)

        local_partition_potentials = dict()
        for vindex, v in enumerate(self.variable_spec):
            if not v.ddomain is None:
                local_partition_potentials[v.name] = np.array([calc_potentials(vindex, val) for val in v.ddomain])
            else:
                local_partition_potentials[v.name] = np.array([calc_potentials(vindex, val) for val in v.int_points])
        return local_partition_potentials

    def mple_objective_and_grad(self, dmap, weights, data_potentials=None, local_partition_potentials=None):
        # Objective function (negative log-pseudo-likelihood) and gradient for 
        # MRF maximum pseudo-likelihood estimation

        npoints = len(dmap.itervalues().next())

        if data_potentials is None:
            data_potentials = self.potentials(dmap)
        if local_partition_potentials is None:
            local_partition_potentials = self.get_local_partition_potentials(dmap)

        ll_terms = []
        grad_terms = []
        observed_pots_expectation = np.mean(data_potentials, axis=1)
        for vindex, var in enumerate(self.variable_spec):
            mask = np.array(self.factor_adjacency[vindex, :].todense()).transpose()
            masked_dpots = np.multiply(mask, data_potentials)
            data_term = np.dot(weights, masked_dpots)

            local_partition, expected_pots_given_params, _ = self.mple_local_partition_and_expected_potentials(
                dmap, vindex, weights, local_partition_potentials)
            assert(expected_pots_given_params.shape == (len(weights), npoints))

            ll_terms.append(data_term - local_partition)
            # take mean across data points
            gradient = np.mean(expected_pots_given_params - observed_pots_expectation[:, np.newaxis], axis=1)
            grad_terms.append(gradient)

        assert(np.array(ll_terms).shape == (len(self.variable_spec), npoints))
        assert(np.array(grad_terms).shape == (len(self.variable_spec), len(weights)))

        return (- (1.0 / npoints) * np.sum(ll_terms), np.sum(grad_terms, axis=0))

    def normalized_prob(self, dmap, weights=None):
        if weights is None and self.weights is None:
            raise ValueError("The network parameters must be fit before computing probabilities")
        elif weights is None:
            weights = self.weights

        data_potentials = self.potentials(dmap)
        all_combos =  self.expand_joint(self.variable_spec)
        all_potentials = self.potentials(all_combos)

        return np.exp(np.dot(weights, data_potentials)) / np.exp(np.dot(weights, all_potentials)).sum()

    def unnormalized_prob(self, dmap, weights):
        if weights is None and self.weights is None:
            raise ValueError("The network parameters must be fit before computing probabilities")

        data_potentials = self.potentials(dmap)
        return np.exp(np.dot(weights, data_potentials))

    def gibbs_sample(self, last_state, weights=None):
        if weights is None and self.weights is None:
            raise ValueError("The network parameters must be fit before running Gibbs sampling")
        elif weights is None:
            weights = self.weights

        all_potentials = self.get_local_partition_potentials(last_state)

        next_state = dict()
        for vindex, var in enumerate(self.variable_spec):
            _, _, probs = self.mple_local_partition_and_expected_potentials(
                    last_state, vindex, weights, all_potentials)
            if not var.ddomain is None:
                next_state[var.name] = np.apply_along_axis(lambda row: np.random.choice(var.ddomain, p=row), 0, probs)
            else:
                # getting inverse cdf
                cum_points = sp.integrate.cumtrapz(probs, var.int_points, axis=0, initial=0)
                # since numeric integration sometimes yields slightly more or less than unit area:
                cum_points[-1, :] = np.repeat(1, probs.shape[1])
                def sample_cdf(cdf_vals):
                    interpolator = spinterp.interp1d(cdf_vals, var.int_points)
                    return interpolator(np.random.uniform())
                next_state[var.name] = np.apply_along_axis(sample_cdf, 0, cum_points)

        return next_state

    def pseudonormalized_prob(self, dmap, weights=None):
        if weights is None and self.weights is None:
            raise ValueError("The network parameters must be fit before computing pseudo-probabilities")
        elif weights is None:
            weights = self.weights

        local_partition_potentials = self.get_local_partition_potentials(dmap)

        data_potentials = self.potentials(dmap)
        probs = np.zeros(len(dmap.itervalues().next()))
        for vindex, var in enumerate(self.variable_spec):
            mask = np.array(self.factor_adjacency[vindex, :].todense()).transpose()
            masked_dpots = np.multiply(mask, data_potentials)
            weighted_dpots = np.dot(weights, masked_dpots)
            local_partition, expected_pots_given_params, _ = self.mple_local_partition_and_expected_potentials(
                    dmap, vindex, weights, local_partition_potentials)
            probs = probs + weighted_dpots - local_partition

        return np.exp(probs)

    def potentials(self, dmap):
        return np.array([f(dmap) for f in self.potential_funs])

    def expand_joint(self, variable_spec):
        """ Creates a data map which specifies the joint assignments of all variables """
        grid = np.meshgrid(*(a.ddomain for a in variable_spec))
        return dict([(variable_spec[i].name, e.flatten()) for i, e in enumerate(grid)])

