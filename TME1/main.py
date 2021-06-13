import collections as c
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTree

# Exercice 1
print("\nExercice 1 : " + "\n")


# Entropie
def entropy(vect):
    if vect.size == 0:
        return 0
    count = c.Counter(vect)
    py = np.array(list(count.values())) / sum(count.values())
    return - np.sum(py * np.log2(py))


# Entropie Conditionnelle
def cond_entropy(list_vect):
    Hyp = np.zeros(len(list_vect))
    p = np.zeros(len(list_vect))
    sizee = sum([vect.size for vect in list_vect])
    for i, vect in enumerate(list_vect):
        Hyp[i] = entropy(vect)
        p[i] = vect.size / sizee
    return np.sum(p * Hyp)


list_vect = np.zeros((3, 3))
list_vect[0][0] = 1
list_vect[0][2] = 1
list_vect[1][1] = 1
print("Entropy : ", entropy(np.array([1, 2, 1, 2, 1, 1])))
print("Condtionnal entropy : ", cond_entropy(list_vect))

# Application : Entropie et Entropie Conditionnelle

# data : tableau ( films , features ) , id2titles : dictionnaire id -> titre ,
# fields : id feature -> nom
[data, id2titles, fields] = pickle.load(open(os.path.join(os.getcwd(), "imdb_extrait.pkl"), "rb"))

# la derniere colonne est le vote
datax = data[:, :32]

# binarisation de la note moyenne
datay = np.array([1 if x[33] > 6.5 else -1 for x in data])

# entropie de la note
maxi = 0
val_i = 0
for i in range(33 - 5):
    attribut = fields[i]
    e = entropy(datay)

    # on veut toutes les lignes / tous les films dont la valeur du vote = 1 Puis une seconde partition = 0
    part1 = datay[data[:, i] == 1]
    part2 = datay[data[:, i] == 0]
    ec = cond_entropy([part1, part2])

    gainInfo = e - ec
    if gainInfo > maxi:
        maxi = gainInfo
        val_i = i

    print("Information gain with " + fields[i] + " : ", gainInfo)

# si ec vaut 0 = pas de desordre, classe parfaitement separer sur l'attribut
# e - ec doit etre max et ec min si e-ec= 0 attribut est pas bon car ec trop grand si = 1  attribut parfait => ec = 0 et e -ec = e entropie max = 1
# le meilleur attribut pour la premiere partition : gain info max donc ec min
# plus l'entropie est faible, plus les labels sont proches, donc plus les classes sont bien séparées -> on a donc choisit le meilleur attribut et le meilleur test pour séparer le noeud si les entropies cond a l attribut sont min => gain info max
attribut = fields[val_i]
print(maxi, val_i, attribut)

# Exercice 2 : Sklearn
print("\nExercice 2 : " + "\n")

profondeurs = range(1,18)
"""data = []
for i, p in enumerate(profondeurs):
    dt = DTree()
    dt.max_depth = p  # on fixe la taille max de l ’ arbre a 5
    dt.min_samples_split = 2  # nombre minimum d ’ exemples pour spliter un noeud
    dt.fit(datax, datay)
    dt.predict(datax[:5, :])
    score = dt.score(datax, datay)
    print("Percentage of well-classified data : ", score)
    data += [[i, score]]
    print(data)
    if not os.path.exists('dots'):
        os.makedirs('dots')
    export_graphviz(dt, out_file="dots/tree_depth" + str(i) + ".dot", feature_names=list(fields.values())[:32])
    dirname = os.path.dirname(__file__)
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    os.system("dot -Tpng dots/tree_depth" + str(i) + ".dot > imgs/tree_depth" + str(i) + ".png")
plt.title("Courbe d'évolution du taux de bonne classification en fonction de la profondeur de l'arbre).")
plt.ylabel("Taux de bonne classification")
plt.xlabel("Profondeur")
plt.plot([x for x, y in data])
plt.savefig('trace_score.png', bbox_inches="tight")"""
# plus la profondeur est grande plus les exemples sont séparés.
# plus la profondeur est grande plus le score de bonne classif est grand, attention on test sur données d'apprentissage
# tester sur jeu de donnée de test


# Exercice 3 : Sur apprentissage et Sous apprentissage
print("\nExercice 3 : " + "\n")


def trace_complet(datax, datay, start, end, pas=1):
    X_train0_2, X_test0_8, y_train0_2, y_test0_8 = train_test_split(datax, datay, test_size=0.8, random_state=42)
    X_train0_5, X_test0_5, y_train0_5, y_test0_5 = train_test_split(datax, datay, test_size=0.5, random_state=42)
    X_train0_8, X_test0_2, y_train0_8, y_test0_2 = train_test_split(datax, datay, test_size=0.2, random_state=42)

    color = ["green", "red", "blue"]
    patchs = [mpatches.Patch(color="green", label='20 % apprentissage'),
              mpatches.Patch(color="red", label='50 % apprentissage'),
              mpatches.Patch(color="blue", label='80 % apprentissage')]
    Xtrains = [X_train0_2, X_train0_5, X_train0_8]
    Xtests = [X_test0_8, X_test0_5, X_test0_2]
    ytrains = [y_train0_2, y_train0_5, y_train0_8]
    ytests = [y_test0_8, y_test0_5, y_test0_2]
    for i in range(1):
        test = []
        train = []
        for n in range(start, end, pas):
            X_test = Xtests[2]
            y_test = ytests[2]
            X_train = Xtrains[2]
            y_train = ytrains[2]

            dt = DTree()
            dt.max_depth = n  # on fixe la taille max de l ’ arbre a 5
            dt.min_samples_split = 2  # nombre minimum d ’ exemples pour spliter un noeud
            dt.fit(X_train, y_train)
            # dt.predict(datax[:5, :])
            score = dt.score(X_test, y_test)
            test += [[n, score]]

            score = dt.score(X_train, y_train)
            train += [[n, score]]

        plt.title("Courbe d'évolution du taux de bonne classification en fonction de la profondeur de l'arbre).")
        plt.ylabel("Taux de bonne classification")
        plt.xlabel("Profondeur")
        plt.plot([x for x, y in test], [y for x, y in test], color=color[2])
        plt.savefig('trace_complet.png' , bbox_inches="tight")
    plt.show()


trace_complet(datax, datay, 2, 30)

# on voit peu de différences...
# validation croisé si peu de donnée

# Exercice 4 : Validation croisée
print("\nExercice 4 : " + "\n")


def trace_complet_croisee(datax, datay, start, end, pas=1):
    color = ["green", "red", "blue"]

    for i in range(1):
        test = []
        for n in range(start, end, pas):
            dt = DTree()
            dt.max_depth = n  # on fixe la taille max de l ’ arbre a 5
            dt.min_samples_split = 2  # nombre minimum d ’ exemples pour spliter un noeud

            # dt.predict(datax[:5, :])
            scores = cross_val_score(dt, datax, datay)
            test += [[n, np.mean(scores)]]

        plt.title("Courbe d'évolution du taux de bonne classification en fonction de la profondeur de l'arbre).")
        plt.ylabel("Taux de bonne classification")
        plt.xlabel("Profondeur")
        plt.plot([x for x, y in test], [y for x, y in test], color=color[2])
        plt.savefig('trace_croisee.png', bbox_inches="tight")

    plt.show()


trace_complet_croisee(datax, datay, 2, 30)

## Bonus hyperparamètres

def trace_complet_croisee_min(datax, datay, start, end, pas=1):
    color = ["green", "red", "blue"]
    patchs = [mpatches.Patch(color="green", label='20 % apprentissage'),
              mpatches.Patch(color="red", label='50 % apprentissage'),
              mpatches.Patch(color="blue", label='80 % apprentissage')]
    for i in range(1):
        test = []
        train = []
        for n in range(start, end, pas):
            dt = DTree()
            dt.max_depth = 30
            dt.min_samples_split = n  # nombre minimum d ’ exemples pour spliter un noeud

            # dt.predict(datax[:5, :])
            scores = cross_val_score(dt, datax, datay)
            test += [[n, np.mean(scores)]]

        plt.legend(handles=patchs)
        plt.title("Courbe d'évolution du taux de bonne classification en fonction du nombre min d'exemple par noeud).")
        plt.ylabel("Taux de bonne classification")
        plt.xlabel("Nombre minimal d'exemples par noeud")
        plt.plot([x for x, y in train], [y for x, y in train], "--", color=color[i])
        plt.plot([x for x, y in test], [y for x, y in test], color=color[i])
        plt.savefig('complet_croisee_min.png', bbox_inches="tight")
    plt.show()


trace_complet_croisee_min(datax, datay, 2, 30)

## Bonus hyperparamètres
def trace_complet_croisee_entropie(datax, datay, start, end, pas=1):
    color = ["green", "red", "blue"]
    patchs = [mpatches.Patch(color="green", label='20 % apprentissage'),
              mpatches.Patch(color="red", label='50 % apprentissage'),
              mpatches.Patch(color="blue", label='80 % apprentissage')]
    for i in range(3):
        test = []
        train = []
        for n in range(start, end, pas):
            dt = DTree(criterion='entropy')
            dt.max_depth = 30
            dt.min_samples_split = n  # nombre minimum d ’ exemples pour spliter un noeud
            # dt.min_impurity_decrease = n
            # dt.predict(datax[:5, :])
            scores = cross_val_score(dt, datax, datay)
            test += [[n, np.mean(scores)]]

        plt.legend(handles=patchs)
        plt.title("Courbe d'évolution du taux de bonne classification en fonction du gain d'entropie minimal).")
        plt.ylabel("Taux de bonne classification")
        plt.xlabel("Nombre minimal d'exemples par noeud")
        plt.plot([x for x, y in test], [y for x, y in test], color=color[i])
        plt.savefig('complet_croisee_entropie.png', bbox_inches="tight")
    plt.show()


trace_complet_croisee_entropie(datax, datay, 2, 30)
