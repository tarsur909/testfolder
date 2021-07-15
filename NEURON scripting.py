# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from neuron import h, rxd
import neuron
from neuron.units import ms, mV

soma = h.Section(name = 'soma')
h.topology()
soma.psection()

soma.L = 20
soma.diam = 20
soma.insert('hh')

iclamp = h.IClamp(soma(0.5))
iclamp.delay = 2
iclamp.dur = 0.1
iclamp.amp = 0.9

v = h.Vector().record(soma(0.5)._ref_v)             
t = h.Vector().record(h._ref_t)         

h.load_file('stdrun.hoc')
h.finitialize(-65 * ms )
h.continuerun(40 * mV )        

import matplotlib.pyplot as plt
fig = plt.figure()
plt.xlabel('t (ms)')
plt.ylabel('v (mV)')
plt.plot(t, v)
plt.show()    

import csv
with open('data.csv', 'w') as f:
    csv.writer(f).writerows(zip(t, v))

import pandas as pd
data = pd.read_csv('data.csv', header=None, names=['t', 'v'])

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

X = np.asarray(data['t']).reshape(-1,1)
y = np.asarray(data['v']).reshape(-1,1)
mdl = RandomForestRegressor()
mdl.fit(X,y)
print(mdl.score(X,y))
fig = plt.figure()
plt.plot(t, mdl.predict(X))
plt.xlabel('t (ms)')
plt.ylabel('v predictions (mV)')
plt.show()



