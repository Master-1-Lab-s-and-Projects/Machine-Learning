import pickle

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

POI_FILENAME = "../data/poi-paris.pkl"
parismap = mpimg.imread('../data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = np.array([xmin, xmax, ymin, ymax])


class Density(object):
    def fit(self, data):
        pass

    def predict(self, data):
        pass

    def score(self, data):
        # ajout de 10e-10 pour eviter un log de 0
        y = self.predict(data) + 10e-10
        return np.sum(np.log(y))


class Histogramme(Density):
    def __init__(self, steps=10):
        Density.__init__(self)
        self.steps = steps

    def fit(self, x):
        self.histogram = np.histogramdd(x, bins=self.steps)
        self.histogram = [np.array(self.histogram[0]), np.array(self.histogram[1])]
        self.xmax = np.max(x[:, 0])  # np.max(self.histogram[1][0])
        self.xmin = np.min(x[:, 0])  # np.min(self.histogram[1][0])
        self.ymax = np.max(x[:, 1])  # np.max(self.histogram[1][1])
        self.ymin = np.min(x[:, 1])  # np.min(self.histogram[1][1])
        self.min = np.array([self.xmin, self.ymin])
        self.max = np.array([self.xmax, self.ymax])
        self.volume = ((self.xmax - self.xmin) * (self.ymax - self.ymin)) / (self.steps ** 2)
        self.histogram[0] /= (len(x) * self.volume)

    def to_bin(self, x):
        # epsilon pour données avec valeur max
        l = ((x - self.min) / (self.max - self.min))
        l *= self.steps
        l = np.intc(np.floor(l))
        return np.where(l >= self.steps, self.steps - 1, l)

    def predict(self, x):
        coord = self.to_bin(x)
        # print(coord)
        # print(x)
        return self.histogram[0][coord[:, 0], coord[:, 1]]


def trace_courbe_histo(geo_mat, start, end, pas=1):
    plt.close()
    X_train, X_test = train_test_split(geo_mat, test_size=0.2, random_state=None)
    test, train = [], []
    patchs = [mpatches.Patch(color="green", label='20 % test'), mpatches.Patch(color="red", label='80 % train')]

    for step in range(start, end, pas):
        h = Histogramme(step)
        h.fit(X_train)
        score = h.score(X_test)
        test += [score]
        score = h.score(X_train)
        train += [score]

    plt.legend(handles=patchs)
    plt.title("Courbe d'évolution de la vraissemblance du nombre de bins.")
    plt.ylabel("Vraissemblance")
    plt.xlabel("Nombre de bins")
    plt.plot(range(start, end, pas), test, color="green")
    # plt.plot(np.arange(start,end,pas),train,color="red")
    plt.show()


class KernelDensity(Density):
    def __init__(self, kernel=None, sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self, x):
        self.x = x

    def predict(self, data):
        def f(d):
            return np.sum(self.kernel((d - self.x) / self.sigma))

        r = 1 / (len(self.x) * (self.sigma ** len(data[0])))
        return r * np.array(list(map(f, data)))


def kernel_uniform(data):
    return np.intc(np.all(np.abs(data) <= 0.5, axis=1))


def kernel_gaussian(data):
    tmp = (2 * np.pi) ** (-len(data[0]) / 2)
    return tmp * np.exp(-0.5 * np.sum(data ** 2, axis=1))


def trace_courbe_kernel(geo_mat, kernel, start, end, pas=1):
    plt.close()
    X_train, X_test = train_test_split(geo_mat, test_size=0.2, random_state=42)
    test, train = [], []
    patchs = [mpatches.Patch(color="green", label='20 % test'), mpatches.Patch(color="red", label='80 % train')]
    for sigma in np.arange(start, end, pas):
        d = KernelDensity(kernel, sigma)
        d.fit(X_train)
        score = d.score(X_test)
        test += [score]
        score = d.score(X_train)
        train += [score]
    plt.legend(handles=patchs)
    plt.title("Courbe d'évolution de la vraissemblance en fonction de sigma.")
    plt.ylabel("Vraissemblance")
    plt.xlabel("Sigma")
    plt.plot(np.arange(start, end, pas), test, color="green")
    plt.plot(np.arange(start, end, pas), train, color="red")
    plt.show()


class Nadaraya(Density):

    def __init__(self, kernel=None, sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self, x, y):
        self.x = x
        self.y = y

    def score(self, x, y):
        return ((self.predict(x) - y) ** 2).mean()

    def predict(self, data):
        def f(d):
            tmp = np.sum(self.kernel((d - self.x) / self.sigma))
            if tmp == 0:
                return -1
            return np.sum(self.y * self.kernel((d - self.x) / self.sigma) / tmp)

        return np.array(list(map(f, data)))


def trace_courbe_nadaraya(geo_mat, notes, kernel, start, end, pas=1):
    plt.close()
    X_train, X_test, notes_train, notes_test = train_test_split(geo_mat, notes, test_size=0.2, random_state=42)

    test, train = [], []
    patchs = [mpatches.Patch(color="green", label='20 % test'), mpatches.Patch(color="red", label='80 % train')]
    for sigma in np.arange(start, end, pas):
        d = Nadaraya(kernel, sigma)
        d.fit(X_train, notes_train)
        score = d.score(X_test, notes_test)
        test += [score]
        score = d.score(X_train, notes_train)
        train += [score]

    plt.legend(handles=patchs)
    plt.title("Courbe d'évolution de la vraissemblance en fonction de sigma.")
    plt.ylabel("Erreur Moindres Carrés")
    plt.xlabel("Sigma")
    plt.plot(np.arange(start, end, pas), test, color="green")
    plt.plot(np.arange(start, end, pas), train, color="red")
    plt.show()


def get_density2D(f, data, steps=100):
    """ 
    Calcule la densité en chaque case d'une grille steps x steps dont 
    les bornes sont calculées à partir du min/max de data. 
    Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    xlin, ylin = np.linspace(xmin, xmax, steps), np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    print(grid.shape)
    res = f.predict(grid).reshape(steps, steps)
    # !! Il faut faire la transposée 
    return res, xlin, ylin


def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res + 1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan


def load_poi(typepoi, fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])
    
    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1], v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data, note
