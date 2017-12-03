#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:46:36 2017

@author: Vishisht
"""

import numpy as np
from hidden_markov import hmm
import math

# to print full numpy arrays
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

observations = np.loadtxt('Observations.csv', delimiter=',')
print "Observations Loaded"

ACCURACY = 2

num_observable = int(1.57 * math.pow(10,ACCURACY))

quantities = [0 for i in xrange(0, num_observable)]

for i,n in enumerate(observations):
    for j,m in enumerate(n):
        observations[i][j] = int(round(observations[i][j]*math.pow(10,ACCURACY),0))

observations = observations.astype(int)

for i,n in enumerate(observations):
    for j,m in enumerate(n):
        if (0 < observations[i][j]) and (observations[i][j] < num_observable):
            quantities[observations[i][j]] += 1
        else:
            print "Observed value out of valid set !"

states = np.arange(10)
states = states.tolist()

list_observables = [i for i in xrange(0,num_observable)]

start_prob = np.random.random(len(states))
start_prob = np.asmatrix(start_prob)

trans_prob = np.random.random((len(states),len(states)))
trans_prob = np.asmatrix(trans_prob)

emis_prob = np.random.random((len(states),num_observable))
emis_prob = np.asmatrix(emis_prob)

for i,n in enumerate(start_prob):
    for j,m in enumerate(n):
        start_prob[i][j] = start_prob[i][j]/np.sum(n)
        
for i,n in enumerate(trans_prob):
    for j,m in enumerate(n):
        trans_prob[i][j] = trans_prob[i][j]/np.sum(n)
        
for i,n in enumerate(emis_prob):
    for j,m in enumerate(n):
        emis_prob[i][j] = emis_prob[i][j]/np.sum(n)

print len(states)
print num_observable
print start_prob.shape
print trans_prob.shape
print emis_prob.shape

print "Starting with hmm"
test = hmm(states,list_observables,start_prob,trans_prob,emis_prob)
print "Done with hmm"

iterations = 1
print "Starting with Baum-Welch"
e,t,s = test.train_hmm(observations, iterations, quantities)
print "Done with Baum-Welch"
