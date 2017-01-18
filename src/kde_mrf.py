import numpy as np
import scipy as sp
from scipy import misc as spmisc
from scipy import optimize as spoptim
from scipy import stats
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

class ProductPotential(PotentialFunction):
    def __init__(self, variables):
        self.var_list = list(variables)

    def variables(self):
        return self.var_list
    
    def __call__(self, dmap):
        running = dmap[self.var_list[0]]
        for v in self.var_list[1:]:
            running = np.multiply(running, dmap[v])
        return running

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
    def __init__(self, dmap, variables):
        """
            Creates the kernel density estimate potential function
            
            Arguments:
                dmap: A map from variable name to an array-like type 
                      representing the values for that variable
                variables:
                     Iterable of variables participating in the potential
        """
        self.var_list = list(variables)
        self.kde = stats.gaussian_kde(np.array((dmap[v] for v in self.var_list)))

    def variables(self):
        return self.var_list

    def __call__(self, dmap):
        """ Evaluates the potential function at a set of points """
        return self.kde(np.array((dmap[v] for v in self.var_list)))

class VariableDef(object):
    def __init__(self, name, ddomain=None, samples=None):
        self.name = name
        self.ddomain = ddomain
        self.samples = samples
        if ddomain is None and samples is None:
            raise ValueError("One of ddomain (the discrete domain) or samples " + 
                "(for continuous variables) must be supplied.")

    def icdf(self, q):
        if self.ddomain:
            raise ValueError("Empirical CDF is undefined for variables with a discrete domain.")
        return stats.mstats.mquantiles(self.samples, q)

class MarkovNetwork(object):
    
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

    def mle_optimize(self, dmap, initweights=None):
        if not initweights:
            initweights = np.random.uniform(0, 1.0, len(self.potential_funs))
        data_potentials = self.potentials(dmap)
        all_combos =  self.expand_joint(self.variable_spec)
        all_potentials = self.potentials(all_combos)

        return spoptim.minimize(lambda weights: 
            self.mle_objective_and_grad(dmap, weights, 
                data_potentials=data_potentials, all_potentials=all_potentials), initweights, jac=True)

    def mple_optimize(self, dmap, initweights=None, jac=True):
        if not initweights:
            initweights = np.random.uniform(0, 1.0, len(self.potential_funs))
        data_potentials = self.potentials(dmap)

        if jac:
            return spoptim.minimize(lambda weights: 
                self.mple_objective_and_grad(dmap, weights, 
                    data_potentials=data_potentials), initweights, jac=True)
        else:
            return spoptim.minimize(lambda weights: 
                self.mple_objective_and_grad(dmap, weights, 
                    data_potentials=data_potentials)[0], initweights, jac=False)

    def mple_local_partition_and_expected_potentials(self, dmap, vindex, weights):
        """ Creates the local pseudo-likelihood partition function for a given variable """
        var = self.variable_spec[vindex]
        npoints = len(dmap.itervalues().next())
        mask = np.array(self.factor_adjacency[vindex, :].todense()).transpose()

        def calc_potentials(val):
            dmap_prime = dict([(v.name, dmap[v.name]) 
                if i != vindex else (v.name, np.repeat(val, npoints)) for i, v in enumerate(self.variable_spec)])
            return self.potentials(dmap_prime)

        if not var.ddomain is None: # discrete
            all_potentials = np.array([calc_potentials(v) for v in var.ddomain])
            weighted_potentials = np.dot(weights, np.multiply(mask, all_potentials))
            all_probs = np.divide(np.exp(weighted_potentials), np.sum(np.exp(weighted_potentials), axis=0))
            local_partition = spmisc.logsumexp(weighted_potentials, axis=0)
            expected_pots_given_params = np.sum(np.multiply(all_probs[:, np.newaxis, :], all_potentials), axis=0)
        else:
            eval_points = var.icdf(np.linspace(0, 1, 20))
            all_potentials = np.array([calc_potentials(v) for v in eval_points])
            weighted_potentials = np.dot(weights, np.multiply(mask, all_potentials))

            normalizer = sp.integrate.trapz(np.exp(weighted_potentials), eval_points, axis=0)

            all_densities = np.divide(np.exp(weighted_potentials), normalizer) # densities, not masses
            expected_pots_given_params = sp.integrate.trapz(np.multiply(all_densities[:, np.newaxis, :], all_potentials), eval_points, axis=0)
            local_partition = np.log(normalizer)

        return (local_partition, expected_pots_given_params)

    def mple_objective_and_grad(self, dmap, weights, data_potentials=None):
        # Objective function (negative log-pseudo-likelihood) and gradient for 
        # MRF maximum pseudo-likelihood estimation

        npoints = len(dmap.itervalues().next())
        def gen_configs(vname, val):
            # gets terms for the summation over configurations of a single variable
            return dict([(v.name, dmap[v.name]) 
                if v.name != vname else (v.name, np.repeat(val, npoints)) for v in self.variable_spec])

        if data_potentials is None:
            data_potentials = self.potentials(dmap)

        ll_terms = []
        grad_terms = []
        observed_pots_expectation = np.mean(data_potentials, axis=1)
        for vindex, var in enumerate(self.variable_spec):
            mask = np.array(self.factor_adjacency[vindex, :].todense()).transpose()
            masked_dpots = np.multiply(mask, data_potentials)
            data_term = np.dot(weights, masked_dpots)

            local_partition, expected_pots_given_params = self.mple_local_partition_and_expected_potentials(dmap, vindex, weights)
            assert(expected_pots_given_params.shape == (len(weights), npoints))

            ll_terms.append(data_term - local_partition)
            # take mean across data points
            gradient = np.mean(expected_pots_given_params - observed_pots_expectation[:, np.newaxis], axis=1)
            grad_terms.append(gradient)

        assert(np.array(ll_terms).shape == (len(self.variable_spec), npoints))
        assert(np.array(grad_terms).shape == (len(self.variable_spec), len(weights)))

        return (- (1.0 / npoints) * np.sum(ll_terms), np.sum(grad_terms, axis=0))

    def normalized_prob(self, dmap, weights):
        data_potentials = self.potentials(dmap)
        all_combos =  self.expand_joint(self.variable_spec)
        all_potentials = self.potentials(all_combos)

        return np.exp(np.dot(weights, data_potentials)) / np.exp(np.dot(weights, all_potentials)).sum()

    def normalized_pll_prob(self, dmap, weights):
        data_potentials = self.potentials(dmap)
        all_combos =  self.expand_joint(self.variable_spec)
        all_potentials = self.potentials(all_combos)
        return np.exp(np.dot(weights, data_potentials)) / np.exp(np.dot(weights, all_potentials)).sum()

    def normalized_pll_density(self, dmap, weights):
        data_potentials = self.potentials(dmap)
        #print(data_potentials)
        densities = np.zeros(len(dmap.itervalues().next()))
        for vindex, var in enumerate(self.variable_spec):
            mask = np.array(self.factor_adjacency[vindex, :].todense()).transpose()
            masked_dpots = np.multiply(mask, data_potentials)
            weighted_dpots = np.dot(weights, masked_dpots)
            local_partition, expected_pots_given_params = self.mple_local_partition_and_expected_potentials(dmap, vindex, weights)
            densities = densities + (np.exp(np.sum(masked_dpots, axis=0)) / np.exp(local_partition))
        return densities

    def potentials(self, dmap):
        return np.array([f(dmap) for f in self.potential_funs])

    def expand_joint(self, variable_spec):
        """ Creates a data map which specifies the joint assignments of all variables """
        grid = np.meshgrid(*(a.ddomain for a in variable_spec))
        return dict([(variable_spec[i].name, e.flatten()) for i, e in enumerate(grid)])

def binary_demo():
    nsamp = 1000
    x1 = np.random.binomial(1, 0.5, nsamp)
    y1 = np.random.binomial(1, x1 * 0.8 + (1 - x1) * 0.2, nsamp)

    x2 = np.random.binomial(1, 0.5, nsamp)
    y2 = np.random.binomial(1, x2 * 0.8 + (1 - x2) * 0.2, nsamp)

    var_defs = [
        VariableDef("x1", ddomain=[0, 1]),
        VariableDef("x2", ddomain=[0, 1]),
        VariableDef("y1", ddomain=[0, 1]),
        VariableDef("y2", ddomain=[0, 1])
    ]

    potentials = [
        BinaryProductPotential(["x1", "y1"]),
        BinaryProductPotential(["x2", "y2"]),
        BinaryProductPotential(["y1", "y2"]),
        BinaryProductPotential(["y1"]),
        BinaryProductPotential(["y2"])
    ]

    data = {
        "x1" : x1,
        "x2" : x2,
        "y1" : y1,
        "y2" : y2,
    }

    network = MarkovNetwork(potentials, var_defs)

    all_combos = network.expand_joint(var_defs)
    x1p, x2p, y1p, y2p = (all_combos["x1"], all_combos["x2"], all_combos["y1"], all_combos["y2"])
    true_probs= ((0.5) * (0.5) * 
        ((y1p == x1p) * 0.8 + (1 - (y1p == x1p)) * 0.2) * 
        ((y2p == x2p) * 0.8 + (1 - (y2p == x2p)) * 0.2)
        )

    np.set_printoptions(suppress=True)
    mle_result = network.mle_optimize(data)
    mle_probs = network.normalized_prob(all_combos, mle_result.x)

    print("-----------------------")
    print("MLE result:")
    print("-----------------------")
    print(mle_result)

    mple_result = network.mple_optimize(data)
    mple_probs = network.normalized_pll_prob(all_combos, mple_result.x)

    print("-----------------------")
    print("MPLE result:")
    print("-----------------------")
    print(mple_result)

    print(np.column_stack((mle_probs, mple_probs, x1p, x2p, y1p, y2p, true_probs)))

def gaussian_demo():
    nsamp = 10
    x1 = np.random.normal(size=nsamp)
    y1 = np.random.normal(loc=x1)

    x2 = np.random.normal(size=nsamp)
    y2 = np.random.normal(loc=x1)

    var_defs = [
        VariableDef("x1", samples=x1),
        VariableDef("x2", samples=x2),
        VariableDef("y1", samples=y1),
        VariableDef("y2", samples=y2)
    ]

    potentials = [
        ProductPotential(["x1", "y1"]),
        ProductPotential(["x2", "y2"]),
        ProductPotential(["y1", "y2"]),
        ProductPotential(["y1"]),
        ProductPotential(["y2"])
    ]

    data = {
        "x1" : x1,
        "x2" : x2,
        "y1" : y1,
        "y2" : y2,
    }

    network = MarkovNetwork(potentials, var_defs)
    mple_result = network.mple_optimize(data)
    print(mple_result)
    density = network.normalized_pll_density(data, mple_result.x)
    np.set_printoptions(suppress=True)
    print(np.column_stack((x1, x2, y1, y2, density)))

def main():
    #binary_demo()
    gaussian_demo()
    
if __name__ == "__main__":
    main()
