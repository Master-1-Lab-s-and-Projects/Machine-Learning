import matplotlib.pyplot as plt
import numpy as np

from mltools import plot_data, plot_frontiere, make_grid, gen_arti


def mse(w, x, y):
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    x = x.reshape(y.shape[0], w.shape[0])
    return (x.dot(w) - y) ** 2


def reglog(w, x, y):
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    x = x.reshape(y.shape[0], w.shape[0])
    return np.log(1 + np.exp(-y * np.dot(x, w)))


def mse_grad(w, x, y):
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    x = x.reshape(y.shape[0], w.shape[0])
    return x * (x.dot(w) - y) * 2


def reglog_grad(w, x, y):
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    x = x.reshape(y.shape[0], w.shape[0])
    return -y * x / (1 + np.exp(y * x.dot(w)))


def grad_check(f, f_grad, n=100, t=1e-5):
    ''' 
    renvoie le nombre de test positif et négatif 
    '''
    # génération aléatoire
    ws = np.random.rand(n, 1)

    datax = np.random.rand(n, 1)
    datay = np.random.randint(0, 2, n)
    res = []
    for w in ws:
        values = f(w, datax, datay)
        valuest = f(w + t, datax, datay)
        pts_grad = f_grad(w, datax, datay)
        res1, res2 = (valuest - values) / t, pts_grad
        res += [np.max(np.abs(res1 - res2)) < t]
    return sum(res), len(res) - sum(res)


if __name__ == "__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax, datay = gen_arti(epsilon=0.1)
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

    plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    w = np.random.randn(datax.shape[1], 1)
    plot_frontiere(datax, lambda x: np.sign(x.dot(w)), step=100)
    plot_data(datax, datay)

    ## Visualisation de la fonction de coût en 2D
    plt.figure()
    plt.contourf(x, y, np.array([mse(w, datax, datay).mean() for w in grid]).reshape(x.shape), levels=50)

    print("grad check mse", grad_check(mse, mse_grad))
    print("grad check reglog", grad_check(reglog, reglog_grad))
