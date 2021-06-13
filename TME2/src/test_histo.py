import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from tme2 import load_poi, show_img, trace_courbe_histo

POI_FILENAME = "../data/poi-paris.pkl"
parismap = mpimg.imread('../data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]

plt.ion()
# Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
# La fonction charge la localisation des POIs dans geo_mat et leur note.


geo_mat1, _ = load_poi("bar")
geo_mat2, _ = load_poi("restaurant")
geo_mat = np.concatenate((geo_mat1, geo_mat2))

geo_night, _ = load_poi("night_club")

# Affiche la carte de Paris
show_img()
# Affiche les POIs
plt.scatter(geo_mat[:, 0], geo_mat[:, 1], alpha=0.8, s=3)

'''
steps = range(5,31,5)
for step in steps:
    print(step)
    h = Histogramme(step)
    h.fit(geo_night)
    res, xlin, ylin = get_density2D(h, geo_night, step)
    show_density(h, geo_night, step, log=False)

'''

'''
# Verification densité : doit sommer à 1
step= 11
h = Histogramme(step)
h.fit(geo_mat)
res, xlin, ylin = get_density2D(h, geo_mat, step)
somme = np.sum(res)*h.volume
print(somme)
'''
# # affichage de la courbe (ne fonctionne pas si show density est appelé avant)
trace_courbe_histo(geo_mat, 5, 35)
