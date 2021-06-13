import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def perceptron_loss(w, x, y):
    res = -y * np.dot(x, w)
    return np.where(res > 0, res, 0)


def perceptron_grad(w, x, y):
    res = np.zeros(x.shape[1])
    poids = -y * np.dot(x, w) > 0
    res -= np.sum(x * y * poids, axis=0)
    return (res / len(x)).reshape(-1, 1)


def hinge_loss(w, x, y, alpha, l):
    res = alpha - y * np.dot(x, w)
    return np.where(res > 0, res, 0) + l * (np.linalg.norm(w) ** 2)


def hinge_loss_grad(w, x, y, alpha, l):
    res = np.zeros(x.shape[1])
    poids = alpha - y * np.dot(x, w) > 0
    res -= np.sum((x * y - (l * 2) * w.reshape(-1)) * poids, axis=0)
    return (res / len(x)).reshape(-1, 1)


class Lineaire(object):
    def __init__(self, loss=perceptron_loss, loss_g=perceptron_grad, max_iter=100, eps=0.01, proj=None):
        self.max_iter, self.eps = max_iter, eps
        self.w = None
        self.loss, self.loss_g = loss, loss_g
        self.proj = proj

    def fit(self, datax, datay, testx=None, testy=None, batch_size=None):
        if batch_size == None:  # par defaut : batch
            batch_size = len(datax)
        if self.proj != None:
            datax = self.proj(datax)
        datay = datay.reshape(-1, 1)
        self.w = np.random.rand(datax.shape[1], 1) * -1
        # self.w = np.zeros((datax.shape[1],1))
        costs = []
        erreurs = []
        for i in range(self.max_iter):
            datax_rand, datay_rand = unison_shuffled_copies(datax, datay)
            datax_rand_batch, datay_rand_batch = list(chunks(datax_rand, batch_size)), list(
                chunks(datay_rand, batch_size))
            n = len(datax_rand_batch)
            erreur = []
            for j in range(n):

                gradient = self.loss_g(self.w, datax_rand_batch[j], datay_rand_batch[j])
                self.w = self.w - self.eps * gradient  # descente de gradient
                costs += [self.loss(self.w, datax_rand_batch[j], datay_rand_batch[j]).mean(axis=0)]
                if not (testx is None):
                    erreur += [[self.score(datax, datay), self.score(testx, testy)]]
            if not (testx is None):
                erreurs += [np.array(erreur).mean(axis=0)]
        return self.w, costs, erreurs

    def predict(self, datax):
        if self.proj != None:
            datax = self.proj(datax)

        return np.sign(np.dot(datax, self.w))

    def score(self, datax, datay):
        datay = datay.reshape(-1, 1)
        y_hat = self.predict(datax)
        return np.mean(np.where(y_hat == datay, 1, 0))


def proj_poly(datax):
    polynomial_features = PolynomialFeatures(degree=2)
    return polynomial_features.fit_transform(datax)


def proj_biais(datax):
    return np.concatenate(([[1]] * datax.shape[0], datax), axis=1)


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


def get_usps(l, datax, datay):
    if type(l) != list:
        resx = datax[datay == l, :]
        resy = datay[datay == l]
        return resx, resy
    tmp = list(zip(*[get_usps(i, datax, datay) for i in l]))
    tmpx, tmpy = np.vstack(tmp[0]), np.hstack(tmp[1])
    return tmpx, tmpy


def show_usps(data):
    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="gray")  # cmap="gray"


def plot_cost(costs):
    plt.figure()
    plt.title("Courbe d'évolution de cout en fonction des iterations.")
    plt.ylabel("Couts")
    plt.xlabel("Iterations")
    plt.plot(range(len(costs)), costs)
    plt.show()


def proj_gauss_x(datax, base, sigma):
    return np.exp(-np.sum((datax - base) ** 2, axis=1) / (2 * sigma)).reshape(-1, 1)


def proj_gauss(datax, base, sigma):
    return np.concatenate(tuple(map(lambda b: proj_gauss_x(datax, b, sigma), base)), axis=1)


def courbe(X_train, X_test, Y_train, Y_test, start, end, pas=1, eps=0.01, batch_size=None):
    plt.figure()
    color = ["green", "red"]
    patchs = [mpatches.Patch(color="green", label='Test'), mpatches.Patch(color="red", label='Apprentissage')]

    perceptron = Lineaire(max_iter=end, eps=eps)
    _, _, scores = perceptron.fit(X_train, Y_train, X_test, Y_test, batch_size=batch_size)
    scores = np.array(scores)
    plt.legend(handles=patchs)
    plt.title("Courbe d'évolution du taux de bonne classification en fonction des itérations.")
    plt.ylabel("Taux de bonne classification")
    plt.xlabel("époque")
    plt.plot(range(start, end, pas), scores[:, 1][start::pas], "--", color=color[0])
    plt.plot(range(start, end, pas), scores[:, 0][start::pas], "--", color=color[1])
    # plt.plot([x for x,y in train],[y for x,y in train],color=color[1])
    plt.show()


def courbe_2(X_train, X_test, Y_train, Y_test, start, end, pas=1, eps=0.01, batch_size=None):
    plt.figure()
    color = ["green", "red"]
    patchs = [mpatches.Patch(color="green", label='Test'), mpatches.Patch(color="red", label='Apprentissage')]
    alpha = 0
    lambdab = 1
    perceptron = Lineaire(max_iter=end, eps=eps, loss=lambda w, x, y: hinge_loss(w, x, y, alpha, lambdab),
                          loss_g=lambda w, x, y: hinge_loss_grad(w, x, y, alpha, lambdab))
    _, _, scores = perceptron.fit(X_train, Y_train, X_test, Y_test, batch_size=batch_size)
    scores = np.array(scores)
    plt.legend(handles=patchs)
    plt.title("Courbe d'évolution du taux de bonne classification en fonction des itérations.")
    plt.ylabel("Taux de bonne classification")
    plt.xlabel("époque")
    plt.plot(range(start, end, pas), scores[:, 1][start::pas], "--", color=color[0])
    plt.plot(range(start, end, pas), scores[:, 0][start::pas], "--", color=color[1])
    # plt.plot([x for x,y in train],[y for x,y in train],color=color[1])
    plt.show()


if __name__ == "__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    # Données USPS
    '''
    neg = 6
    pos = 9
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
    datay = np.where(datay==neg,-1,1)
    testy = np.where(testy==neg,-1,1)
    l = Lineaire(max_iter = 1000)
    _, costs,_ = l.fit(datax,datay)
    
    plot_cost(costs)
    show_usps(l.w)
    # courbe(datax,testx,datay,testy, 1, 10000, pas=1)
    '''
    # Divergence
    '''
    # courbe(datax,testx,datay,testy, 1, 1000, pas=1)
    '''
    # Différents batchs
    '''
    courbe(datax,testx,datay,testy, 1, 300, pas=1)
    courbe(datax,testx,datay,testy, 1, 300, pas=1, batch_size = 1)
    courbe(datax,testx,datay,testy, 1, 300, pas=1, batch_size = 200)
    '''
    # ajout de bruit
    '''
    var = 2
    datax_bruit = datax + np.random.randn(datax.shape[0],datax.shape[1])*var
    data_type = 0
    datax_2, datay_2 = gen_arti(epsilon=0.7,data_type=data_type)
    testx_2, testy_2 = gen_arti(epsilon=0.1,data_type=data_type)
    courbe(datax_2,testx_2,datay_2,testy_2, 1, 300, pas=1)
    courbe(datax_2,testx_2,datay_2,testy_2, 1, 300, pas=1, batch_size = 1)
    courbe(datax_2,testx_2,datay_2,testy_2, 1, 300, pas=1, batch_size = 200)
    '''
    # Données TME3
    '''
    data_type = 0
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax_2, datay_2 = gen_arti(epsilon=0.1,data_type=data_type)
    testx_2, testy_2 = gen_arti(epsilon=0.1,data_type=data_type)
    
    
    l = Lineaire(max_iter=100000)
    _, costs, _ = l.fit(datax_2,datay_2)
    
    plot_cost(costs)
    
    
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

    plt.figure()
    plot_frontiere(datax_2, lambda x: np.sign(x.dot(l.w)), step=100)
    plot_data(datax_2, datay_2)
    plt.show()
    
    # Erreur d'apprentissage
    courbe(datax_2,testx_2,datay_2,testy_2, 1, 1000, pas=1)
    '''
    # Un contre tous
    '''
    pos = 6
    datax,datay = alltrainx,alltrainy
    testx,testy = alltestx,alltesty
    datay = np.where(datay==pos,1,-1)
    testy = np.where(testy==pos,1,-1)
    courbe(datax,testx,datay,testy, 1, 5000, pas=1)
    # courbe_2(datax,testx,datay,testy, 1, 5000, pas=1)
    '''
    '''
    l = Lineaire(max_iter = 1000)
    
    _, costs, _ = l.fit(datax,datay)
    
    plot_cost(costs)
    show_usps(l.w)
    '''
    # projection polynomial
    '''
    data_type = 2
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax_2, datay_2 = gen_arti(epsilon=0.1,data_type=data_type)
    testx_2, testy_2 = gen_arti(epsilon=0.1,data_type=data_type)
    l = Lineaire(proj = proj_poly,max_iter = 2000)
    _, costs, _ = l.fit(datax_2,datay_2)
    
    plot_cost(costs)
    
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

    plt.figure()
    plot_frontiere(datax_2, lambda x: l.predict(x), step=100)
    plot_data(datax_2, datay_2)
    plt.show()
    '''
    # projection gaussienne
    '''
    data_type = 2
    sigma = 0.4
    
    # bases dans le jeux de données
    nb_base = 1000
    
    bases, _ = gen_arti(epsilon=0.1,data_type=data_type)
    bases = bases [:nb_base]
    
    # bases en grille
    # bases, x, y = make_grid(xmin=-3.8, xmax=4.1, ymin=-3.8, ymax=4.1, step=10)

    
    datax_2, datay_2 = gen_arti(epsilon=0.1,data_type=data_type)
    testx_2, testy_2 = gen_arti(epsilon=0.1,data_type=data_type)
    
    
    l = Lineaire(proj = lambda datax: proj_gauss(datax,bases,sigma),max_iter=5000)
    _, costs, _ = l.fit(datax_2,datay_2)
    
    plot_cost(costs)
    
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

    
    plt.figure()
    
    plot_frontiere(datax_2, lambda x: l.predict(x), step=100)
    bests = bases[l.w.argsort(axis=0)[-10:]]
    
    # plt.scatter(bases[:,0],bases[:,1],marker='+',c='yellow')
    plot_data(datax_2, datay_2)
    # plt.scatter(bests[:,0,0],bests[:,0,1],marker='*',c='white',s=150)
    
    plt.show()
    
    '''
    # projection gaussienne et Hinge loss
    data_type = 2
    alpha = 0
    lambdab = 1
    sigma = 0.1
    '''
    # bases dans le jeux de données
    nb_base = 10000
    
    bases, _ = gen_arti(epsilon=0.1,data_type=data_type,nbex=10000)
    bases = bases [:nb_base]
    '''
    '''
    bases, x, y = make_grid(xmin=-3.5, xmax=4.5, ymin=-3.5, ymax=4.5, step=8)
    
    datax_2, datay_2 = gen_arti(epsilon=0.1,data_type=data_type)
    testx_2, testy_2 = gen_arti(epsilon=0.1,data_type=data_type)

    l = Lineaire(loss = lambda w,x,y : hinge_loss(w,x,y,alpha,lambdab), 
                 loss_g = lambda w,x,y : hinge_loss_grad(w, x, y, alpha ,lambdab),
                 max_iter = 5000,proj = lambda datax: proj_gauss(datax,bases,sigma))
    _, costs, _ = l.fit(datax_2,datay_2)
    
    plot_cost(costs)
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

    plt.figure()
    plot_frontiere(datax_2, lambda x: l.predict(x), step=100)
    plt.scatter(bases[:,0],bases[:,1],marker='+',c='yellow')
    plot_data(datax_2, datay_2)
    plt.show()
    '''
    # Test hinge_loss
    '''
    neg = 6
    pos = 9
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
    datay = np.where(datay==neg,-1,1)
    testy = np.where(testy==neg,-1,1)
    alpha = 1
    lambdab = 5
    l = Lineaire(loss = lambda w,x,y : hinge_loss(w,x,y,alpha,lambdab), 
                 loss_g = lambda w,x,y : hinge_loss_grad(w, x, y, alpha ,lambdab))
    _, costs,_ = l.fit(datax,datay)
    
    plot_cost(costs)
    show_usps(l.w)
    '''
    # Un contre tous hinge loss
    '''
    pos = 6
    datax,datay = alltrainx,alltrainy
    testx,testy = alltestx,alltesty
    datay = np.where(datay==pos,1,-1)
    testy = np.where(testy==pos,1,-1)
    l = Lineaire(loss = lambda w,x,y : hinge_loss(w,x,y,alpha,lambdab), 
                 loss_g = lambda w,x,y : hinge_loss_grad(w, x, y, alpha ,lambdab))
    _, costs, _ = l.fit(datax,datay,testx,testy)
    
    plot_cost(costs)
    show_usps(l.w)
    '''
