from sklearn.linear_model import Perceptron
from TME5.src.utils import *
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import numpy as np
import sklearn.svm as svm


def perceptron_usps(alltrainx, alltrainy, alltestx, alltesty):
    datax, datay = get_usps([neg, pos], alltrainx, alltrainy)
    datay = np.where(datay == neg, -1, 1)
    clf = Perceptron(alpha=5, penalty='l2', eta0=1e-3, fit_intercept=False,
                     max_iter=1000, verbose=1, validation_fraction=0.1,
                     n_jobs=-1)
    clf.fit(datax, datay)
    # clf.fit(datax, datay, coef_init=np.random.rand(testx.shape[1], 1))
    return clf


if __name__ == "__main__":
    neg = 6
    pos = 9
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    testx, testy = get_usps([neg, pos], alltestx, alltesty)
    testy = np.where(testy == neg, -1, 1)
    # perceptron_clf = perceptron_usps(alltrainx, alltrainy, alltestx, alltesty)
    # print("Classification score", perceptron_clf.score(testx, testy))
    # clf.coef_.toarray()
    # show_usps(perceptron_clf.coef_[0])
    svm_clf = svm.SVC(C=0.001, kernel='rbf', degree=2,
                      shrinking=False, probability=True, verbose=1,
                      max_iter=-1, decision_function_shape='ovr')
    datax, datay = gen_arti(epsilon=0.1, data_type=1, nbex=1000)
    svm_clf.fit(datax, datay)
    plt.figure()
    # plot_frontiere_proba(datax, lambda x: svm_clf.predict_proba(x)[:, 0], step=50)
    plot_frontiere(datax, lambda x: svm_clf.predict(x), step=100)
    plot_data(datax, datay)
    plt.show()
