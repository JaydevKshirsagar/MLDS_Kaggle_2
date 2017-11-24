#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:12:12 2017

@author: Vishisht
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import time

# Num Observations
# 1.57 radians (first quadrant) divided into 1000 parts
NUM_OBSERVATIONS = 1570

# Num States
# Discretize the Domain between (0, 3) into 1000 parts
NUM_STATES = 3000

def forward_algo(observations, initialprob, trans, emis, NUM_STATES):
    forwardMatrix = np.zeros((NUM_STATES, observations.size))
    
    for i in xrange(NUM_STATES):
        forwardMatrix[i,0] = initialprob[i] * emis[i,0]
        
    for j in xrange(1,observations.size):
        for i in range(NUM_STATES):
            forwardMatrix[i, j] = emis[i, observations[j]] * sum([forwardMatrix[i2, j-1] * trans[i2, i] for i2 in range(NUM_STATES)])

    #print "ForwardMat"
    #print forwardMatrix

    return forwardMatrix
            
def backward_algo(observations, trans, emis, NUM_STATES):
    backwardMatrix = np.zeros((NUM_STATES, observations.size))
    
    for i in xrange(NUM_STATES):
        backwardMatrix[i,observations.size-1] = 1
        
    for j in xrange(observations.size-2, -1, -1):
        for i in range(NUM_STATES):
            backwardMatrix[i, j] = sum([backwardMatrix[i2, j+1] * trans[i, i2] * emis[i2, observations[j+1]] for i2 in range(NUM_STATES)])
           
    #print "BackwardMat"
    #print backwardMatrix

    return backwardMatrix
            
def expectation(observations, initialprob, trans, emis, NUM_STATES):
    gamma = np.zeros((NUM_STATES, observations.size))
    sum_observations_gamma = np.zeros(observations.size)
    
    forward = forward_algo(observations, initialprob, trans, emis, NUM_STATES)
    backward = backward_algo(observations, trans, emis, NUM_STATES)
    
    for i in xrange(NUM_STATES):
        for j in xrange(observations.size):
            gamma[i, j] = forward[i,j]*backward[i,j]
            sum_observations_gamma[j] += gamma[i,j]
  
    for i in xrange(NUM_STATES):
        for j in xrange(observations.size):
            gamma[i,j] /= sum_observations_gamma[j]

    xi = np.zeros((NUM_STATES,NUM_STATES, observations.size))
    sum_observations_xi = np.zeros((NUM_STATES,NUM_STATES))
    
    for i in xrange(NUM_STATES):
        for j in xrange(NUM_STATES):
            for k in xrange(observations.size):
                xi[i,j,k] = forward[i,k]*trans[i,j]*backward[j,k]*emis[j,observations[k]]
                sum_observations_xi[i,j] += xi[i,j,k]
                
    for i in xrange(NUM_STATES):
        for j in xrange(NUM_STATES):
            for k in xrange(observations.size):
                xi[i,j,k] /= sum_observations_xi[i,j]
                
    return (gamma,xi)
                
def maximization(observations, initialprob, trans, emis, NUM_STATES):
    gamma,xi = expectation(observations, initialprob, trans, emis, NUM_STATES)

    initialprob[:] = gamma[:, 0]

    for i in xrange(NUM_STATES):
        for j in xrange(NUM_STATES):
            numerator = 0
            denominator = 0

            for k in xrange(observations.size):
                numerator += xi[i, j, k]
                denominator += gamma[i, k]

            trans[i, j] = numerator / denominator

    for i in xrange(NUM_STATES):
        denominator = 0

        for k in xrange(observations.size):
            denominator += gamma[i, k]

            emis[i, observations[k]] += gamma[i, k]
      
        for j in xrange(NUM_OBSERVATIONS):
            emis[i, j] /= denominator

    return gamma, xi
            

def test_forwardbackward(observations, numiter):
    
    #####
    # HMM initialization
    
    # initialize initial probs
    unnormalized = np.random.rand(NUM_STATES)
    initialprob = unnormalized / sum(unnormalized)
    
    # initialize emission probs
    emis = np.zeros((NUM_STATES, NUM_OBSERVATIONS))
    for s in range(NUM_STATES):
        unnormalized = np.random.rand(NUM_OBSERVATIONS)
        emis[s] = unnormalized / sum(unnormalized)
    
    # initialize transition probs
    trans = np.zeros((NUM_STATES, NUM_STATES))
    for s in range(NUM_STATES):
        unnormalized = np.random.rand(NUM_STATES)
        trans[s] = unnormalized / sum(unnormalized)

    for count, sample in enumerate(observations):
        for iteration in range(numiter):
            print "In sample: %d iteration number: %d" %(count, iteration)
            gamma, xi = maximization(sample, initialprob, trans, emis, NUM_STATES)

if __name__ == "__main__":

    # Read the training data
    observations = np.loadtxt('Observations.csv', delimiter=',')

    test_forwardbackward(observations, 100)













