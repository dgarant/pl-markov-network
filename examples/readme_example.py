from plmrf import *
import numpy as np

np.random.seed(100)
x = np.random.binomial(1, 0.5, size=100)
x = (x-1) + x # transform to -1/1
y = np.random.normal(loc=x)
data = {"x" : x, "y" : y}

variable_defs = [VariableDef("x", ddomain=[-1, 1]),
                 VariableDef("y", samples=y)]

potentials = [GaussianPotential(["x", "y"], samples=data),
              GaussianPotential(["y"], samples=data, location=0)]

network = LogLinearMarkovNetwork(potentials, variable_defs)
result = network.fit(data)
print(result.x)

new_data = {"x" : np.array([1, -1]), "y" : np.array([-10, 0])}
print(network.gibbs_sample(new_data))
print(network.unnormalized_prob(new_data))


