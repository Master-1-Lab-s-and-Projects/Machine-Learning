#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import graphtools as tools
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
class AdaBoost():
    def __init__(self,datax,datay,classifieur):
        self.cl = classifieur
        self.param = np.ones(len(datax))/len(datax)
        self.datax = datax
        
        self.datay = datay
        self.alphas = []
        self.list_classifieurs = []
    
    def learn(self):
        new_cl = self.cl()
        # print(self.param)
        new_cl.fit(self.datax,self.datay,sample_weight = self.param)
        self.list_classifieurs += [new_cl]
        self.alpha
    
    def error(self):
        datayhat = self.list_classifieurs[-1].predict(self.datax)
        self.error_t = np.dot(self.param.T,np.where(datayhat == self.datay,0,1))
        # print("error",self.error_t)
    
    def alpha(self):
        self.alphas += [1/2*np.log((1-self.error_t)/self.error_t)]
        # print("alpha",self.alphas[-1])
        
    def update_params(self):
        self.param = self.param\
            *np.exp(-self.alphas[-1]*self.datay*self.list_classifieurs[-1].predict(self.datax))
        # print("param",self.param.shape)
        self.param /= np.sum(self.param)
        
    def fit(self,nb_classifieurs):
        errors = []
        alphas = []
        w = []
        for x in range(nb_classifieurs):
            self.learn()
            self.error()
            self.alpha()
            self.update_params()
            errors += [self.error_t]
            w += [self.param]
            
        return errors, alphas, w
    
    def predict(self,datax):
        res = np.zeros(len(datax))
        for t,classif in enumerate(self.list_classifieurs):
            res += self.alphas[t]*classif.predict(datax)
        res /= len(self.list_classifieurs)
        
        return np.sign(res)
    
def test_2D():
    nb_iter = 10
    n = 1000
    
    
    for x in range(nb_iter):
        data, label = tools.gen_arti(centerx=1,centery=1,sigma=0.1,nbex=n,data_type=1,epsilon=0.2)
        cl = AdaBoost(data[:int(0.9*n)],label[:int(0.9*n)],Perceptron)
        error,_,ws = cl.fit(10)
        train = data[:int(0.9*n)]
        for w in ws:
            print(train[:,1])
            plt.contour(train[:,0],train[:,1],w)
            plt.show()
        tools.plot_frontiere(data[:int(0.9*n)],cl.predict,step=100)
            
        tools.plot_data(data[:int(0.9*n)],labels=label[:int(0.9*n)])
        plt.show()
        
        print("train score",np.sum(cl.predict(data[:int(0.9*n)]) == label[:int(0.9*n)])/(0.9*n))
        print("test score",np.sum(cl.predict(data[int(0.9*n):]) == label[int(0.9*n):])/(0.1*n), "\n")
    
test_2D()
