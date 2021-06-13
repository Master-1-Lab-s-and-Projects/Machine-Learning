# -*- coding: utf-8 -*-
"""
	KMeans
"""

import numpy as np
import  matplotlib . pyplot  as  plt
    
def assign(C,X):
    # C : centres clusters
    # X : nos points
    def apply(x):
        return np.argmin(np.sum(C - x,axis = 1)**2)
   
    return list(map(apply,X))

def centroides(C,X):
    means_cluster = []
    for cluster in np.unique(C):
        liste = X[np.where(C==cluster)] 
        means_cluster+=[liste.mean(axis=0)]
    return means_cluster

#on  garde  que  l e s  3  premieres  composantes ,  la  transparence  est  i n u t i l e
im=plt.imread("index.png")[:,:,:3 ]

im_h,im_l,_ = im.shape
pixels = im.reshape((im_h*im_l,3))*255
pixels_new = pixels.copy() 
nb_bary = 10
C = pixels[np.random.choice(len(pixels),nb_bary)] # np.random.uniform(low = 0, high = 1,size = (nb_bary,3) ) # 

for x in range(100):
    res = assign(C,pixels)
    
    C = centroides(res,pixels)
C = np.intc(C)

res = np.array(res)
for i,x in enumerate(C):
    pixels_new[np.where(res==i)]=x


pixels_new = np.intc(pixels_new)
#transformation  en  matrice n∗3 , n nombre de  p i x e l 
imnew=pixels_new.reshape((im_h,im_l,3)) #transformation  inverse
plt.imshow(imnew)
