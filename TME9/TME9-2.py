#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
	Processus gaussien
	Partie Fonction de covariance
"""
from scipy.stats import multivariate_normal as m1
from numpy.random import multivariate_normal as m2
from sklearn.metrics import pairwise_distances as pd 
import numpy as np
import matplotlib.pyplot as plt

def lineaire(x1,x2,sigma = 1):
    return (sigma**2) * (x1@x2.T)

def exponentiel(X1,X2,sigma = 1,l = 1):
    return (sigma**2)*np.exp(-(0.5/l**2)*((X1**2).sum(1).reshape(-1, 1)+(X2**2).sum(1) - 2*X1.dot(X2.T)))


def periodique(x1,x2,sigma = 1,l = 1, p = 1):
    
    def f1(x1,x2):    
        return (sigma**2) * np.exp((-2/(l**2))*(np.sin(np.pi*np.linalg.norm(x2-x1)/p)**2)) 
    # return f1(x1[0],x2[0])
    return pd(x1,x2,metric = f1)

    
x = np.arange(-5,5,1)[...,None]

plt.imshow(lineaire(x,x))
plt.show()
plt.imshow(exponentiel(x,x))
plt.show()
plt.imshow(periodique(x,x,l = 10e-10, p=1))
