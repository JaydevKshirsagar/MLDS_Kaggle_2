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
NUM_OBSERVATIONS = 3

def forward_algo(observations, initialprob, trans, emis, numstates):
    forwardMatrix = np.zeros((numstates, observations.size))
    
    for i in xrange(numstates):
        forwardMatrix[i,0] = initialprob[i] * emis[i,0]
        
    for j in xrange(1,observations.size):
        for i in range(numstates):
            forwardMatrix[i, j] = emis[i, observations[j]] * sum([forwardMatrix[i2, j-1] * trans[i2, i] for i2 in range(numstates)])

    #print "ForwardMat"
    #print forwardMatrix

    return forwardMatrix
            
def backward_algo(observations, trans, emis, numstates):
    backwardMatrix = np.zeros((numstates, observations.size))
    
    for i in xrange(numstates):
        backwardMatrix[i,observations.size-1] = 1
        
    for j in xrange(observations.size-2, -1, -1):
        for i in range(numstates):
            backwardMatrix[i, j] = sum([backwardMatrix[i2, j+1] * trans[i, i2] * emis[i2, observations[j+1]] for i2 in range(numstates)])
           
    #print "BackwardMat"
    #print backwardMatrix

    return backwardMatrix
            
def expectation(observations, initialprob, trans, emis, numstates):
    gamma = np.zeros((numstates, observations.size))
    sum_observations_gamma = np.zeros(observations.size)
    
    forward = forward_algo(observations, initialprob, trans, emis, numstates)
    backward = backward_algo(observations, trans, emis, numstates)
    
    for i in xrange(numstates):
        for j in xrange(observations.size):
            gamma[i, j] = forward[i,j]*backward[i,j]
            sum_observations_gamma[j] += gamma[i,j]
  
    for i in xrange(numstates):
        for j in xrange(observations.size):
            gamma[i,j] /= sum_observations_gamma[j]
            
    xi = np.zeros((numstates,numstates, observations.size))
    sum_observations_xi = np.zeros((numstates,numstates))
    
    for i in xrange(numstates):
        for j in xrange(numstates):
            for k in xrange(observations.size):
                xi[i,j,k] = forward[i,k]*trans[i,j]*backward[j,k]*emis[j,observations[k]]
                sum_observations_xi[i,j] += xi[i,j,k]
                
    for i in xrange(numstates):
        for j in xrange(numstates):
            for k in xrange(observations.size):
                xi[i,j,k] /= sum_observations_xi[i,j]
                
    return (gamma,xi)
                
def maximization(observations, initialprob, trans, emis, numstates):
    gamma,xi = expectation(observations, initialprob, trans, emis, numstates)

    #print "GAMMA"
    #print gamma

    #print "XI"
    #print xi

    initialprob[:] = gamma[:, 0]

    for i in xrange(numstates):
        for j in xrange(numstates):
            numerator = 0
            denominator = 0

            for k in xrange(observations.size):
                numerator += xi[i, j, k]
                denominator += gamma[i, k]

            trans[i, j] = numerator / denominator

    for i in xrange(numstates):
        denominator = 0

        for k in xrange(observations.size):
            denominator += gamma[i, k]

            emis[i, observations[k]] += gamma[i, k]
      
        for j in xrange(NUM_OBSERVATIONS):
            emis[i, j] /= denominator

    return gamma, xi
            
            
def generate_observations(n):
    # probabilities of ice cream amounts given hot / cold:
    # shown here as amounts of 1's, 2's, 3's out of 10 days
    hot = [ 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    cold = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 2]

    observations  = [ ]
    # choose 2n observations from "hot"
    numhot = int(2.0/3 * n)
    for i in range(numhot): observations.append(random.choice(hot))
    # choose n observations from "cold"
    numcold = n - numhot
    for i in range(numcold): observations.append(random.choice(cold))

    observations = [2, 0, 2, 2, 1, 2, 1, 1, 1, 0, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 0, 2, 1, 0, 1, 2, 1, 2, 1, 2, 2, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]

    numhot = 33
    numcold = 17

    return (observations, numhot, numcold)


def test_forwardbackward(numobservations, numiter):
    ########
    # generate observation
    observations, truenumhot, truenumcold = generate_observations(numobservations)  # generates a data for the icecream observations5
    obs_indices = { 1 : 0, 2 : 1, 3: 2}
    numstates = 2
    vocabsize = 3
    
    observations = np.array(observations)

    #####
    # HMM initialization
    
    # initialize initial probs
    unnormalized = np.random.rand(numstates)
    initialprob = unnormalized / sum(unnormalized)
    
    # initialize emission probs
    emis = np.zeros((numstates, vocabsize))
    for s in range(numstates):
        unnormalized = np.random.rand(vocabsize)
        emis[s] = unnormalized / sum(unnormalized)
    
    # initialize transition probs
    trans = np.zeros((numstates, numstates))
    for s in range(numstates):
        unnormalized = np.random.rand(numstates)
        trans[s] = unnormalized / sum(unnormalized)

    print("OBSERVATIONS:")
    print(observations)
    print("\n")
    
    print("Random initialization:")
    print("INITIALPROB")
    print(initialprob)
    print("\n")

    print("EMIS")
    print(emis)
    print("\n")

    print("TRANS")
    print(trans)
    print("\n")

    #input()


    for iteration in range(numiter):
        
        gamma, xi = maximization(observations, initialprob, trans, emis, numstates)
        """
        print("Re-computed:")
        print("INITIALPROB")
        print(initialprob)
        print("\n")

        print("EMIS")
        print(emis)
        print("\n")

        print("TRANS")
        print(trans)
        print("\n")
    
        print("GAMMA(1)")
        print(gamma[0])
        print("\n")
        
        print("GAMMA(2)")
        print(gamma[1])
        print("\n")
        """     

        # the first truenumhot observations were generated from the "hot" state.
        # what is the probability of being in state 1 for the first
        # truenumhot observations as opposed to the rest
        avgprob_state1_for_truehot = sum(gamma[0][:truenumhot]) / truenumhot
        avgprob_state1_for_truecold = sum(gamma[0][truenumhot:]) / truenumcold
        print("Average prob. of being in state 1 when true state was Hot:", avgprob_state1_for_truehot)
        print("Average prob. of being in state 1 when true state was Cold:", avgprob_state1_for_truecold)


        #input()

        """
        # plot observations and probabilities of being in certain states
        from matplotlib import interactive
        interactive(True)
        xpoints = np.arange(len(observations))
        fig, ax1 = plt.subplots()
        ax1.plot(xpoints, observations, "b-")
        plt.ylim([0, 4])
        ax1.set_xlabel("timepoints")
        ax1.set_ylabel("observations", color = "b")
    
        ax2 = ax1.twinx()
        ax2.plot(xpoints, gamma[0], "r-")
        plt.ylim([0.0, 1.0])
        ax2.set_ylabel("prob", color = "r")
        plt.show()
#        input()
        plt.close()
        """     
test_forwardbackward(50,100)
