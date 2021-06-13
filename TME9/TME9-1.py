#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
	Processus gaussien
	Partie Gaussienne en folie
"""

from scipy.stats import multivariate_normal as m1
from numpy.random import multivariate_normal as m2
import numpy as np
import matplotlib.pyplot as plt

# rouge
mean_a = [1,2]
var_a = [[5,0],[0,1]]
size_a = 300

# vert
mean_b = [3,-3]
var_b = [[0.9,0.1],[0.1,0.9]] + np.identity(2) * np.random.random((2,2)) * 10e-5
size_b = 300

# bleu
mean_c = [-2,-2]
var_c = [[2,0.9],[0.9,2]] + np.identity(2) * np.random.random((2,2)) * 10e-5
size_c = 300

p1 = m2(mean_a,var_a,size = size_a)
p2 = m2(mean_b,var_b,size = size_b)
p3 = m2(mean_c,var_c,size = size_c)

x = np.arange(-5,7,0.1)
y = np.arange(-5,7,0.1)
xx, yy = np.meshgrid(x,y)

za = m1.pdf(list(zip(xx.flatten(),yy.flatten())),mean_a,var_a) 
za = za.reshape(xx.shape)


zb = m1.pdf(list(zip(xx.flatten(),yy.flatten())),mean_b,var_b)
zb = zb.reshape(xx.shape)

zc = m1.pdf(list(zip(xx.flatten(),yy.flatten())),mean_c,var_c)
zc = zc.reshape(xx.shape)

h1 = plt.contour(x,y,za)
h2 = plt.contour(x,y,zb)
h3 = plt.contour(x,y,zc)
taille=30
plt.scatter(p1[:,0],p1[:,1],c="red",marker="+",s=taille)
plt.scatter(p2[:,0],p2[:,1],c="green",marker="o",s=taille)
plt.scatter(p3[:,0],p3[:,1],c="blue",marker="^",s=taille)
plt.show()

nb_plot = 10
for x in p1[:nb_plot]:
    plt.plot(range(1,len(x)+1),x - mean_a,c="red")
for x in p2[:nb_plot]:
    plt.plot(range(1,len(x)+1),x - mean_b,c="green")
for x in p3[:nb_plot]:
    plt.plot(range(1,len(x)+1),x - mean_c,c ="blue")
plt.show()

dim = 20
constant = 10
size = 10
mean = np.random.random(dim)
var = np.identity(dim) + (1 - np.identity(dim)) * constant
p = m2(mean,var,size = size)
for x in p:
    plt.plot(range(1,len(x)+1),x)

