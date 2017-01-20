# Learns parameters and then estimates the MAP state using DPMP
# see https://github.com/samuela/pyDPMP

import pyDPMP
import plmrf 
import numpy as np

nsamp = 1000
x1 = np.random.normal(loc=1, size=nsamp)
y1 = np.random.normal(loc=x1+0.5)

x2 = np.random.normal(size=nsamp)
y2 = np.random.normal(loc=x2)

var_defs = [
    plmrf.VariableDef("x1", samples=x1),
    plmrf.VariableDef("x2", samples=x2),
    plmrf.VariableDef("y1", samples=y1),
    plmrf.VariableDef("y2", samples=y2)
]

data = {
    "x1" : x1,
    "x2" : x2,
    "y1" : y1,
    "y2" : y2,
}

potentials = [
    plmrf.GaussianPotential(["x1", "y1"], samples=data),
    plmrf.GaussianPotential(["x2", "y2"], samples=data),
    plmrf.GaussianPotential(["y1", "y2"], samples=data),
    plmrf.GaussianPotential(["y1"], samples=data, location=1.5),
    plmrf.GaussianPotential(["y2"], samples=data, location=0),
    plmrf.GaussianPotential(["x1"], samples=data, location=1),
    plmrf.GaussianPotential(["x2"], samples=data, location=0)
]

network = plmrf.LogLinearMarkovNetwork(potentials, var_defs)
mple_result = network.fit(data)
print("Learned weights:")
print(mple_result.x)

node_potentials = {"y1" : potentials[3], "y2" : potentials[4], "x1" : potentials[5], "x2" : potentials[6]}
edge_potentials = {("x1", "y1") : potentials[0], ("x2", "y2") : potentials[1],
                   ("y1", "y2") : potentials[2]}
def node_pot(s, vs):
    return node_potentials[s]({s : vs})
def edge_pot(s, t, vs, vt):
    return edge_potentials[(s, t)]({s: vs, t: vt})


dpmp_network = pyDPMP.mrf.MRF(["x1", "x2", "y1", "y2"],
                              [("x1", "y1"), ("x2", "y2"), ("y1", "y2")],
                              node_pot, edge_pot)
proposal = pyDPMP.proposals.random_walk_proposal_1d(1)
initial_particles = {"x1" : [1.0], "x2" : [1.0], "y1" : [1.0], "y2" : [1.0]}
xMAP, xParticles, stats = pyDPMP.DPMP_infer(dpmp_network, initial_particles,
                                     50, # num paricles
                                     proposal,
                                     pyDPMP.particleselection.SelectDiverse(),
                                     pyDPMP.messagepassing.MaxSumMP(dpmp_network),
                                     conv_tol=None,
                                     max_iters=100)
print("Estimated MAP state (true state is {x1: 1,  y1: 1.5, y2: 0, x2: 0})")
print(xMAP)
