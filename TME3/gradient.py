#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from tme3 import reglog, reglog_grad


def descente_gradient(datax, datay, f_loss, f_grad, eps, iter
                      , verbose=False, wst=None):
    '''
        verbose permet d'afficher le cout à chaque étape
        wst permet de donnée une valeur initial pour w
    '''

    if wst is None: wst = np.random.rand(datax.shape[1])

    w = [wst]
    costs = []
    for i in range(iter):
        if verbose: print(f_loss(wst, datax, datay).mean(axis=0))
        gradient = f_grad(wst, datax, datay)
        wst = wst - eps * gradient.mean(axis=0)
        w += [wst]
        costs += [f_loss(wst, datax, datay).mean(axis=0)]
    return wst, w, costs


def plot_cost(costs):
    plt.figure()
    plt.title("Courbe d'évolution de cout en fonction des iterations.")
    plt.ylabel("Couts")
    plt.xlabel("Iterations")
    plt.plot(range(len(costs)), costs)


def plot_w(w, f):
    w = np.array(w)
    plt.figure()
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    plt.contourf(x, y, np.array([f(w, datax, datay).mean() for w in grid]).reshape(x.shape), levels=50)
    plt.scatter(w[-1][0], w[-1][1], c='r')
    plt.plot(w[:, 0], w[:, 1], lw=3)


if __name__ == "__main__":

    # fonction de coup
    f = reglog  # mse reglog
    # fonction de gradient
    f_grad = reglog_grad  # mse_grad reglog_grad
    # quand espilon est trop petit on est tres long a converger -> besoin de plus d'iter

    ## Tirage d'un jeu de données aléatoire
    # datax, datay = gen_arti(data_type=0,epsilon=0.1)
    # datax, datay = gen_arti(data_type=0,epsilon=5) # pour avoir une séparation non linéaire
    datax, datay = gen_arti(data_type=0, epsilon=0.1)

    # valeur de w initial pour test
    wst_b = np.random.rand(datax.shape[1])
    # wst_b  = np.array([1,-1]) # permet de voir l'évolution plus facilement
    # wst_b  = np.array([0,-1]) # permet de voir l'évolution plus facilement

    # Convergence Normale
    iter = 1000
    epsilon = 10e-3

    # Convergence Ralentie 
    iter = 1000
    epsilon = 10e-6

    # Convergence Accélérée 
    iter = 1000
    epsilon = 0.4

    # Divergence MSE
    iter = 1000
    epsilon = 0.5

    # Divergence MSE faible
    iter = 5
    epsilon = 0.5

    # Evolution des frontieres de décisions

    wst, w, couts = descente_gradient(datax, datay, f, f_grad, epsilon, iter, verbose=False, wst=wst_b)
    for ws in w[::199]:
        ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
        grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

        plt.figure()
        plot_frontiere(datax, lambda x: np.sign(x.dot(ws)), step=100)
        plot_data(datax, datay)

    plot_cost(couts)

    '''
    # frontiere optimale
    wst, w ,costs=descente_gradient(datax,datay,f,f_grad,epsilon,iter,wst=wst_b)
    

    # Evolution du cout en fonction des iterations
    #plot_cost(costs)    
    
    # Evolution de w en fonction des iterations
    plot_w(w,f)
    '''
