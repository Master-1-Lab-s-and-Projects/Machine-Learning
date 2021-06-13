import itertools as it
import re

import sklearn.svm as svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate, GridSearchCV

from mltools import plot_data, plot_frontiere, gen_arti
from utils import *


# np.random.rand(testx.shape[1], 1)
def perceptron_usps(datax, datay, reg=1, step=1e-3, coef_init=None):
    clf = Perceptron(alpha=reg, penalty='l2', eta0=step, fit_intercept=False,
                     max_iter=1000, verbose=1, validation_fraction=0.1,
                     n_jobs=-1)
    if coef_init == None:
        clf.fit(datax, datay)
    else:
        clf.fit(datax, datay, coef_init=coef_init)
    return clf


def affiche_svm(datax, datay, kern, reg=1, deg=2, sup_v=True, co0=0.0, show=True):
    svm_clf = svm.SVC(C=reg, kernel=kern, degree=deg, coef0=co0,
                      shrinking=False, probability=True, verbose=1,
                      max_iter=-1, decision_function_shape='ovr')

    svm_clf.fit(datax, datay)
    # plt.figure()
    # plot_frontiere_proba(datax, lambda x: svm_clf.predict_proba(x)[:, 0], step=50)
    if show:
        plot_frontiere(datax, lambda x: svm_clf.predict(x), step=100)
        plot_data(datax, datay)
        if sup_v:
            plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1])
        plt.title("svm {}, pena : {}, degree : {}, coef0 : {}".format(kern, str(reg), str(deg), str(co0)))
        plt.show()
    return svm_clf


def affiche_points(x, y, title_x, title_y, model):
    plt.plot(y, x)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.title("Evolution de {} en fonction de {} pour un model {}".format(title_x, title_y, model))
    plt.show()


def cross_valid(k, pena=0.001, d=2, sup_v=True, co0=0.0, show=True):
    svm_clf = svm.SVC(C=pena, kernel=k, degree=d, coef0=co0,
                      shrinking=False, probability=True, verbose=1,
                      max_iter=-1, decision_function_shape='ovr')

    scores = cross_validate(svm_clf, datax, datay, \
                            cv=10, return_train_score=True)
    print(scores)
    return scores['train_score'].mean(), scores['test_score'].mean()


def print_cross_valid():
    l = [0.0001, 0.01, 0.1, 0.3]
    z = [10, 20, 50, 100]

    for m in ["poly", 'linear', 'sigmoid']:
        trains = []
        tests = []
        for i, j in zip(l, z):
            train, test = cross_valid(m, pena=i, d=2, co0=1, show=False)
            trains += [train]
            tests += [test]
        affiche_points(trains, l, "train score", "pena", m)
        affiche_points(tests, l, "test score", "pena", m)
    # print_cross_valid()


def GridSearchCV_valid(datax, datay, ker, params, cv=10):
    svm_clf = svm.SVC(kernel=ker, shrinking=False, probability=True, \
                      max_iter=-1, decision_function_shape='ovr')

    clf = GridSearchCV(svm_clf, params, \
                       cv=cv, return_train_score=True)

    clf.fit(datax, datay)
    return clf


def GridSearchCV_model(datax, datay, cv=5):
    params = {
        'C': np.arange(0.05, 2, 0.1)
    }
    params_poly = {
        'C': np.arange(0.05, 2, 0.1),
        'degree': [2, 3, 4],  # poly
        'coef0': np.arange(0.05, 2, 0.1)  # poly
    }
    params_sig = {
        'C': np.arange(0.05, 2, 0.1),
        'coef0': np.arange(0.05, 2, 0.1)
    }
    best = []

    for m in ["poly", 'linear', 'rbf']:
        if m == "poly":
            s = GridSearchCV_valid(datax, datay, m, params_poly, cv)
        elif m == "sigmoid":
            s = GridSearchCV_valid(datax, datay, m, params_sig, cv)
        else:
            s = GridSearchCV_valid(datax, datay, m, params, cv)

        best += [s]
    return best


# coef0 => biais que pour poly


# Perceptron
neg = 6
pos = 9
uspsdatatrain = "data/USPS_train.txt"
uspsdatatest = "data/USPS_test.txt"

alltrainx, alltrainy = load_usps(uspsdatatrain)
alltestx, alltesty = load_usps(uspsdatatest)

datax, datay = get_usps([neg, pos], alltrainx, alltrainy)
datay = np.where(datay == neg, -1, 1)

testx, testy = get_usps([neg, pos], alltestx, alltesty)
testy = np.where(testy == neg, -1, 1)
'''
perceptron_clf = perceptron_usps(datax, datay, 3 , 1e-3)
print("Classification score", perceptron_clf.score(testx, testy))
#perceptron_clf.coef_.toarray()
show_usps(perceptron_clf.coef_[0])
'''

# Test parameters
# datax, datay = gen_arti(epsilon=0.1, data_type=2, nbex=1000)
# poly / linear / sigmoid / rbf
'''
for reg in [0.01,1,10]:
    affiche_svm(datax, datay, "rbf", reg = reg, deg = 2, sup_v = True, co0 = 1, show = True)
'''
'''
-------------------------------------------------
    GridSearch
-------------------------------------------------
'''

datax, datay = gen_arti(epsilon=0.1, data_type=1, nbex=10000)
b = GridSearchCV_model(datax, datay)
print(b[0].best_score_)
print(b[1].best_score_)
print(b[2].best_score_)

'''
-------------------------------------------------
    Apprentissage multi-classe
-------------------------------------------------
'''
'''
#ovo par def si plus de 2 label
svm_clf_1 = svm.SVC(kernel="rbf", shrinking=False, probability=True,
                    max_iter=-1, decision_function_shape='ovo')
svm_clf_1.fit(alltrainx, alltrainy)
print(svm_clf_1.score(alltestx, alltesty))

#ovr
svm_clf_2 = []
for x in np.unique(alltrainy):
    datay = np.where(alltrainy == x, 1, -1)
    svm_clf_2 += [svm.SVC(kernel="rbf", shrinking=False, probability=True,
                          max_iter=-1, decision_function_shape='ovr')]
    svm_clf_2[x].fit(alltrainx, datay)
    testy = np.where(alltesty == x, 1, -1)
    if x > 0:
        a = svm_clf_2[x].predict_proba(alltestx)[:, 1]
        b = np.column_stack((b, a))
    else:
        a = svm_clf_2[x].predict_proba(alltestx)[:, 1]
        b = a
predict_vec = np.argmax([x for x in b], axis=1)
score = sum(np.where(predict_vec == alltesty, 1, 0)) / len(alltestx)
print(score)
'''

'''
-------------------------------------------------
    String Kernel
-------------------------------------------------
'''


def split(word):
    return [char for char in word]


def string_kernel(X, Y, lmbda=1, k=3):
    print(len(X), len(Y))
    ress = np.zeros((len(X), len(Y)))
    print(ress.shape)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            print(x, y)
            res = 0
            alphabet = "".join(np.unique(split(X + Y)))
            for seq in it.permutations(alphabet, k):
                # print(seq)
                a = re.search(".*".join(seq), x, flags=0)
                b = re.search(".*".join(seq), y, flags=0)
                if a and b:
                    res += lmbda ** (a.span()[1] - a.span()[0] + 1 + b.span()[1] - b.span()[0] + 1)
            ress[i, j] = res
    return ress


def firstNonRepeatingChar(str1):
    liste = []
    for c in str1:
        if c not in liste:
            liste.append(c)
    return liste


def suite(liste):
    for i in range(len(liste) - 1):
        if liste[i + 1] != liste[i] + 1:
            return False
    return True


'''def string_kernel2(X,Y,lbd=1, n=2):
    somme = 0
    X = ["ab","de"]
    Y = ["bc","ef"]
    matrice = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            x=X[i]
            y=Y[j]
            #print(x,y)
            alpha = firstNonRepeatingChar(x.join(y))
            print(alpha)
            #print(alpha)
            for u in it.permutations(alpha, n):
                u = ''.join(u)
                for combo in itertools.combinations(enumerate(x), len(u)):
                    a=0
                    if "".join(pair[1] for pair in combo) == u:
                        #print([pair[0] for pair in combo])
                        l=[pair[0] for pair in combo]
                        if suite(l):
                            a = len([pair[0] for pair in combo])
                    for combo in itertools.combinations(enumerate(y), len(u)):
                        b=0
                        if "".join(pair[1] for pair in combo) == u:
                            l=[pair[0] for pair in combo]
                            if suite(l):
                                b = len([pair[0] for pair in combo])
                        somme += lbd**(a+b)
            matrice[i][j] = somme
            somme = 0
    print(matrice)
    return matrice
'''
'''
def string_kernel2(X,Y,lbd=1, n=2):
    somme = 0
    #X = ["abdef","de"]
    #Y = ["ab","defab"]
    matrice = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            x=X[i]
            y=Y[j]
            #print(x,y)
            alpha = firstNonRepeatingChar(x.join(y))
            #print(alpha)
            #print(alpha)
            for u in it.permutations(alpha, n):
                u = ''.join(u)
                if (u in x or x in u) and (u in y or y in u):
                    a = min(len(x),len(u))
                    b = min(len(y),len(u))
                    #print(x,y,u,lbd**(a+b))
                    somme += lbd**(a+b)
            matrice[i][j] = somme
            somme = 0
    return matrice

import pandas as pd 
df = pd.read_csv(r"data/test_final.csv", encoding ="latin-1") 

statement_1 = []
statement_2 = []

for x,y in zip(df.statement,df.speaker):
    if y == "donald-trump":
        statement_1+=[re.sub(r'[^a-zA-Z ]+', '',x.lower())]
    if y == "barack-obama":
        statement_2+=[re.sub(r'[^a-zA-Z ]+', '',x.lower())]

X=statement_1[0].split(" ")
Y=statement_2[0].split(" ")
print(X)
print(Y)
res=string_kernel2(X,Y)
print(res)
'''
'''
import pandas as pd 
df = pd.read_csv(r"data/test_final.csv", encoding ="latin-1") 

statement_1 = []
statement_2 = []

for x,y in zip(df.statement,df.speaker):
    if y == "donald-trump":
        statement_1+=[re.sub(r'[^a-zA-Z ]+', '',x.lower())]
    if y == "barack-obama":
        statement_2+=[re.sub(r'[^a-zA-Z ]+', '',x.lower())]

# res = string_kernel(statement_1[0].split(" "),statement_2[0].split(" "))
res_2 = string_kernel(statement_1[0].split(" "),statement_2[1].split(" "))
'''
