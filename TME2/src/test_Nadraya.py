import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from tme2 import Nadaraya, kernel_gaussian, kernel_uniform
from tme2 import load_poi

POI_FILENAME = "../data/poi-paris.pkl"
parismap = mpimg.imread('../data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]

plt.ion()
# Liste des POIs : furniture_store, laundry, bakery, cafe, 
# home_goods_store, clothing_store, atm, lodging, night_club,
# convenience_store, restaurant, bar

# La fonction charge la localisation des POIs dans geo_mat et leur note.


geo_mat1, notes1 = load_poi("bar")
geo_mat2, notes2 = load_poi("restaurant")
geo_mat = np.concatenate((geo_mat1, geo_mat2))
notes = np.concatenate((notes1, notes2))

# suppression des notes -1
geo_mat = geo_mat[notes >= 0]
notes = notes[notes >= 0]

sigma = 0.025

h = Nadaraya(kernel=kernel_uniform, sigma=sigma)
h.fit(geo_mat[10:], notes[10:])

print(h.predict(geo_mat[0:10]))
print(notes[0:10])
print(h.predict([geo_mat.mean(axis=0)]))

sigma = 0.002

h = Nadaraya(kernel=kernel_gaussian, sigma=sigma)
h.fit(geo_mat[10:], notes[10:])

print(h.predict(geo_mat[0:10]))
print(notes[0:10])
print(h.predict([geo_mat.mean(axis=0)]))

# trace_courbe_nadaraya(geo_mat,notes, kernel_gaussian, 0.001, 0.010, pas=0.001)
