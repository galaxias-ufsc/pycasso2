'''
Created on 21 de set de 2016

@author: andre
'''

from pycasso2 import FitsCube
import matplotlib.pyplot as plt

c = FitsCube('../data/manga-7443-12703_synth.fits')
plt.ioff()
plt.imshow(c.LickIndex('D4000'), vmin=0.9, vmax=2.0)
plt.colorbar()
plt.show()
