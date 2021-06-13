#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
	Processus gaussien
	Partie Processus gaussien et régression
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

    
def predict(trainx, trainy,testx,K,sigma):
    sigma_k = 1
    l_K = 1
    k = K(trainx,trainx,sigma=sigma_k,l = l_K)
    k_et = K(testx,trainx,sigma=sigma_k,l = l_K)
    k_et_et = K(testx,testx,sigma=sigma_k,l = l_K)
    
    inv = np.linalg.inv(k+(sigma**2)*np.identity(trainx.shape[0]))
    
    kt = k_et_et - np.dot(np.dot(k_et,inv),k_et.T)
    mu = np.dot(np.dot(k_et,inv),trainy)
    return mu, kt
    

nb_train = 10 

p = np.random.rand(nb_train,1)*10 - 5
test = np.arange(-5,5,0.1).reshape(-1,1)

def f_bruit(x,sigma):
    return np.sin(x)*x + np.random.normal(scale=sigma,size=x.shape)

def f(x):
    return np.sin(x)*x 

sigma = 0.1 # bruit sur le train
res = f_bruit(p,sigma)
mu,sig = predict(p,res,test,exponentiel,sigma)

# plt.scatter(p[nb_train:nb_test+1+nb_train][:,0],p[nb_train:nb_test+1+nb_train][:,1])
# plt.scatter(p[:nb_train][:,0],p[:nb_train][:,1])
# plt.show()

deb = mu.flatten() - 2*np.sqrt(np.diag(sig))
fin = mu.flatten() + 2*np.sqrt(np.diag(sig))
plt.fill_between(test.flatten(),deb,fin, alpha = 0.2)
plt.scatter(p[:nb_train],res,c="green") # y de train
excepted = f(test)
plt.plot(test,mu,c = "orange") # y trouvé en test
plt.plot(test,excepted,c = "red") # y attendu en test
plt.show()
