'''All calculus stuff goes here
'''

# Imports

import numpy as np
from numpy import *
import pandas as pd
from numpy.linalg import norm
import os
from numpy.linalg import inv
from numpy import linalg

################################################################################
# Binary Classification Functions
################################################################################

def f (x, y, theta):
	x = np.transpose(x)
	x = vstack((ones((1, x.shape[1])), x))
	return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = np.transpose(x)
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10   
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 10000
    ite  = 0

    while norm(t - prev_t) >  EPS and ite < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if ite % 5000 == 0:
            print "Iter", ite
            print "Gradient: ", df(x, y, t), "\n"
        ite += 1
    return t

################################################################################
# Multi Classification Functions
################################################################################