# Demonstrates parameter estimation and Gibbs sampling for a multivariate Gaussian

from plmrf import *
import numpy as np

nsamp = 1000
x1 = np.random.normal(size=nsamp)
y1 = np.random.normal(loc=x1)

x2 = np.random.normal(size=nsamp)
y2 = np.random.normal(loc=x2)

var_defs = [
    VariableDef("x1", samples=x1),
    VariableDef("x2", samples=x2),
    VariableDef("y1", samples=y1),
    VariableDef("y2", samples=y2)
]

data = {
    "x1" : x1,
    "x2" : x2,
    "y1" : y1,
    "y2" : y2,
}

potentials = [
    GaussianPotential(["x1", "y1"], samples=data),
    GaussianPotential(["x2", "y2"], samples=data),
    GaussianPotential(["y1", "y2"], samples=data),
    GaussianPotential(["y1"], samples=data, location=0),
    GaussianPotential(["y2"], samples=data, location=0)
]

network = LogLinearMarkovNetwork(potentials, var_defs)
mple_result = network.fit(data)
print("MPLE optimization result:")
print(mple_result)

x1p, x2p, y1p, y2p = np.array(np.meshgrid(
    np.linspace(np.min(x1), np.max(x1), 5), 
    np.linspace(np.min(x2), np.max(x2), 5), 
    np.linspace(np.min(y1), np.max(y1), 5), 
    np.linspace(np.min(y2), np.max(y2), 5))).reshape(4, -1)
new_data = {
    "x1" : x1p,
    "x2" : x2p,
    "y1" : y1p,
    "y2" : y2p
}

print("\n")
var_order = ["x1", "x2", "y1", "y2"]
cur_state = {"x1" : np.array([0]), "x2" : np.array([0]), 
             "y1" : np.array([0]), "y2" : np.array([0])}
print("\nGibbs sampling from {0}".format([cur_state[v][0] for v in var_order]))
for i in range(20):
    cur_state = network.gibbs_sample(cur_state)
    print("Sample {0} state: {1}".format(i+1, [cur_state[v][0] for v in var_order]))
