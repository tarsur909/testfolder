# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 08:40:57 2020
Modified from an online tutorial
@author: mbezaire
"""

import numpy as np
import pylab as plt
from scipy.integrate import odeint

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m  =   1.0
    """membrane capacitance, in uF/cm^2"""

    g_Na = 120.0
    """Sodium (Na) maximum conductances, in mS/cm^2"""

    g_K  =  36.0
    """Postassium (K) maximum conductances, in mS/cm^2"""

    g_L  =   0.3
    """Leak maximum conductances, in mS/cm^2"""

    E_Na =  50.0
    """Sodium (Na) Nernst reversal potentials, in mV"""

    E_K  = -77.0
    """Postassium (K) Nernst reversal potentials, in mV"""

    E_L  = -54.387
    """Leak Nernst reversal potentials, in mV"""

    t = np.arange(0.0, 450.0, 0.01)
    """ The time to integrate over """

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*np.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*np.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*np.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V: voltage
        |  :param m: activation parameter
        |  :param h: inactivation parameter
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :stimulation should be:
        |           step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        # TODO: Define the stimulation.
        # Given a time t in milliseconds
        # Check the value of the time and set the
        # output current at that time according
        # to the constraints given in the doc string above
        current_inj = 0
        if 100 < t <= 200:
            current_inj = 10
        elif 200 < t <= 300:
            current_inj = 0
        elif 300 < t <= 400:
            current_inj = 35
        elif t > 400:
            current_inj = 0
        #current_inj = current_inj # update right side of this expression with your answer
        return current_inj
    
    def dALLdt(self, X, t, obj):
    #def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt

    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)

        plt.figure()

        plt.subplot(4,1,1)
        plt.title('Hodgkin-Huxley Neuron')
        # TODO: plot the membrane potential as a function of time
        # plt...
        plt.plot(self.t, V)
        # TODO: Add labels to the x and y axis, ex:
        plt.ylabel('V (mV)')
        plt.xlabel('t (ms')

        plt.subplot(4,1,2)
        plt.plot(self.t, ina, 'r', label = 'ina')
        plt.plot(self.t, ik, 'g', label = 'ik')
        plt.plot(self.t, il, 'b', label = 'il')
        # TODO: plot three lines:
        #  - the sodium current as a function of time,
        #  - the potassium current as a function of time,
        #  - the leak current as a function of time
        # Use different colors for each line and label each line
        # Hint: you can make labels with nice subscripts
        # using LaTeX syntax, ex: '$I_{Na}$' corresponds to 
        # a nice look INa label for the sodium current line

        plt.ylabel('Current')
        
        # TODO: Add a legend to this plot

        plt.subplot(4,1,3)
        plt.plot(self.t, m, 'r', label='m')
        plt.plot(self.t, h, 'g', label='h')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()
        
        plt.subplot(4,1,4)
        
        # TODO: Define a sequence for the
        # stimulation. It can be a python list, a
        # numpy array, etc. It should have one
        # entry per time step and the value of the
        # elements at each time step must equal the value
        # that would be returned from the self.I_inj
        # method for that time step

        i_inj_values = [] # set your sequence here
        #
        # Bonus: set i_inj_values using a list comprehension
        i_inj_values = [self.I_inj(i) for i in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)

        plt.show()


# TODO: create a new instance (object) of
# the HodgkinHuxley class
# Then call one of the methods - which method
# will run and plot the simulation?       
hh = HodgkinHuxley()
hh.Main()
# TODO: After you've gotten your code to run successfully,
# let's clean it up a bit more:
# 1. First, create an __init__() method for your class
# 2. Next, make the parameters into properties of your
#    class that get defined within the __init__ method
#    and referenced elsewhere in the class as properties
# 3. Then, try altering parameters of the model?
#    * what happens if you change the capacitance of the cell?
#    * what happens if you increase the sodium channel conductance?
#    * what happens if you alter the reversal potential for potassium?


# Bonus 1: Alter the class definition so that you
# can optionally pass in any of the properties (capacitance,
# ion channel conductances, reversal potentials) if you want
# to use a different value than the default 

# If you are going to be editing parameters and replotting the
# results, can you think of something else to do to make your
# plots more informative?

# Bonus 2: Alter this code so that it can either
# be run independently or imported as a module