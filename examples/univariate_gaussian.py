from plmrf import *
import numpy as np

nsamp = 10000
x = np.random.normal(size=nsamp)
data = {"x" : x}

potential = GaussianPotential(["x"], samples=data, location=0)
network = LogLinearMarkovNetwork([potential],
                                 [VariableDef("x", samples=x)])
mple_result = network.fit(data)
print("MPLE optimization result:")
print(mple_result)

xp = np.linspace(np.min(x), np.max(x), 100)
density = network.pseudonormalized_prob({"x" : xp})
true_density = sp.stats.norm.pdf(xp)
np.set_printoptions(suppress=True)
potentials = network.potentials({"x" : xp}).flatten()

print("\n")
results = np.column_stack((xp, density, true_density))
np.savetxt(sys.stdout, results, fmt="%.4f", delimiter="\t", 
    header=", ".join(["X", "MPLE Density", "True Density"]))
print("Area under density function")
print(sp.integrate.trapz(density, xp))
