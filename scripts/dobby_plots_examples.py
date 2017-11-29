'''
Checking emission lines fitted with dobby in MaNGA cubes.

Usage:

python3 dobby_plots_examples.py 7960-6101

Natalia@UFSC - 29/Nov/2017
'''

import sys
from os import path

import numpy as np
import matplotlib.pyplot as plt

from pycasso2 import FitsCube

########################################################################
# Get galaxy name
galname = sys.argv[1]


########################################################################
# Defining dirs
in_cassiopea = False

if in_cassiopea:
    in_dir = '/home/ASTRO/manga/dr14/starlight/dr14.bin2.cA.Ca0c_6x16/'
    el_dir = '/home/ASTRO/manga/dr14/starlight_el/dr14.bin2.cA.Ca0c_6x16/'    
else:
    in_dir = '/Users/natalia/data/MaNGA/dr14/starlight/dr14.bin2.cA.Ca0c_6x16/'
    el_dir = '/Users/natalia/data/MaNGA/dr14/starlight_el/dr14.bin2.cA.Ca0c_6x16/'    

    
########################################################################
# Reading file
c = FitsCube(path.join(el_dir, 'manga-%s.dr14.bin2.cA.Ca0c_6x16.El.fits') % galname)

# Check cube
print(c._HDUList.info())

# Get fluxes and EQ
El_info = c._getSynthExtension('El_info')
El_F    = c._getSynthExtension('El_F')
El_EW   = c._getSynthExtension('El_EW')

# Get position of Ha in the cubes
flag_Ha = (El_info['lambda'] == 6563)

# Plots EW(Ha)
plt.figure(1)
plt.clf()
plt.imshow(El_EW[flag_Ha, ...][0])
plt.title(r'$\log \mathrm{EW}_{\mathrm{H}\alpha}$')
plt.colorbar()

