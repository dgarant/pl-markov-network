# Demonstrates pseudo-likelihood estimation 
# for a large system of continuous variables


from plmrf import *
import numpy as np
import scipy
import time

# Generating a big ring by sampling variables independently, 
# then sampling based on each configuration's 'true' potential
nvars = 1000
nsamp = 1000

print("Generating data ...")
indep_data = dict()
for vindex in range(nvars):
    samples = np.random.normal(size=nsamp*10)

    varname = "x{0}".format(vindex)
    indep_data[varname] = samples

# potentials functions are Gaussian kernels
def potential(pindex):
    return (1.0/nvars) * np.exp(-np.abs(indep_data["x{0}".format(vindex)], indep_data["x{0}".format((vindex+1) % nvars)]))

unnormalized_density = np.exp(np.sum([potential(p) for p in range(nvars)], axis=0))
relative_density = unnormalized_density / unnormalized_density.sum()
samp_indices = np.random.choice(range(nsamp*10), size=nsamp, p=relative_density)

print("Setting up potentials and variable definitions ...")
data = dict()
var_defs = []
for vindex in range(nvars):
    varname = "x{0}".format(vindex)
    next_var = "x{0}".format((vindex+1) % nvars)

    samples = indep_data[varname][samp_indices]
    data[varname] = samples

    var_defs.append(VariableDef(varname, samples=samples, num_int_points=10))

potentials = []
tied_params = [[], []]
for vindex in range(nvars):
    varname = "x{0}".format(vindex)
    next_var = "x{0}".format((vindex+1) % nvars)
    potentials.append(GaussianPotential([varname], samples=data, location=0))
    potentials.append(GaussianPotential([varname, next_var], samples=data))
    tied_params[0].append(len(potentials)-2)
    tied_params[1].append(len(potentials)-1)

for p in potentials:
    if p.bandwidth < 1e-16:
        print(p)

network = LogLinearMarkovNetwork(potentials, var_defs, tied_weights=tied_params)

print("Fitting parameters ...")
start = time.time()
mple_result = network.fit(data, log=True)
end = time.time()
print("Parameter estimation completed in {0} seconds".format(end - start))
print("MPLE optimization result:")
print(mple_result)

