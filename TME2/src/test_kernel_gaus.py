import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from tme2 import KernelDensity, kernel_gaussian
from tme2 import load_poi, show_img, get_density2D, show_density

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

geo_night, notes_night = load_poi("night_club")

sigma = 0.0006

steps = 30

# Affiche la carte de Paris
show_img()
# Affiche les POIs
plt.scatter(geo_mat[:, 0], geo_mat[:, 1], alpha=0.8, s=3)

h = KernelDensity(kernel=kernel_gaussian, sigma=sigma)
h.fit(geo_mat)

res, xlin, ylin = get_density2D(h, geo_mat, steps)
show_density(h, geo_mat, steps, log=False)

#  verification somme à 1
somme = np.sum(res) * ((xmax - xmin) * (ymax - ymin)) / (steps ** 2)
print(somme)

# trace_courbe_kernel(geo_mat,kernel_gaussian,0.0001,0.004,0.0001)
