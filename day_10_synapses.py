# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# day_10_synapse
# template
# 
# We want to model the current that passes
# through a synapse after it is activated.
# Look at instruction PDF for more details

# TODO: Uncomment the lines below. Set the parameters
#       to realistic values:

tau_r =   2 # (ms) τR: rise time constant
tau_d = 8  # (ms) τD: decay time constant
esyn  = 0  # (mV) ESyn: reversal potential of the synapse
rmp     = -60  # (mV) membrane potential, here treated as a constant, like resting membrane potential
gmax  =  0.005 # (nS) maximum synaptic conductance (aka weight or amplitude) 
C     =  1 # (uF) capacitance of the cell

# TODO: Create a list or array of times for the simulation.
#       The list should start at 0 and range up to 100 (ms)
#       in small time steps, perhaps 0.5 ms:
    
import numpy as np 

times = np.arange(0,100,0.5)


# TODO: Create placeholder vectors for the results variables
#       i (synaptic current, units of nA) and g (synaptic 
#       conductance, units of uS) that have same number of
#       elements as the number of time steps in your simulation.
#       Fill them with 0s:
g = np.zeros(times.shape)
i = np.zeros(times.shape)

# TODO: Then create a placeholder variable of the same size,
#       for membrane potential v. Fill it with resting membrane potentials
v = np.ones(times.shape) * rmp

# TODO: Create a for loop that iterates over each time step
#       in the simulation:
for x,t in enumerate(times):
    g[x] = gmax * (-np.exp(-t/tau_r) + np.exp(-t/tau_d))
    i[x] = g[x] * (v[x] -esyn)
    if x > 0:
        v[x] = v[x-1] + (i[x] + -0.02 * (v[x-1] - rmp))/C

# TODO: Inside the loop, write the three equations given in the
#       instructions to update the g, i, and v each time step.
#       Include the parameters defined above in the equations.
#       When updating v, first check that the simulation is past
#       the 0th time step (don't update the 0th element in v):
#   g[...] = ...
#   i[...] = ...
#   v[...] = ...

# TODO: After the for loop, include code to plot the results.
#       Create one plot of the conductance with respect to
#       time and another of the current with respect to time.
#       (You may create them as subplots on the same figure
#       if you prefer.) Label your axes and include units in
#       the labels:
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,1)
axs[0].plot(times, g)
axs[1].plot(times, i)

# After verifying that your code works, you can try simulating
# different situations as prompted in the instruction document
# and interpreting the results from a biological perspective. 