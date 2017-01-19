from mrf import *
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
mple_result = network.mple_optimize(data)
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

true_covariance = np.array([[1, 0, 1, 0], 
                           [0, 1, 0, 1],
                           [1, 0, 2, 0],
                           [0, 1, 0, 2]])
true_mean = np.array([0, 0, 0, 0])
trueish_density = sp.stats.multivariate_normal.pdf(
    np.array([x1p, x2p, y1p, y2p]).T, mean=true_mean, cov=true_covariance)

print("\n")
density = network.pseudonormalized_prob(new_data, mple_result.x)
results = np.column_stack((density, trueish_density, new_data["x1"], new_data["x2"], new_data["y1"], new_data["y2"]))
np.savetxt(sys.stdout, results, fmt="%.3f", delimiter="\t", 
    header=", ".join(["MPLE Pseudo-density", "True(ish) Density", "X1", "X2", "Y1", "Y2"]))
