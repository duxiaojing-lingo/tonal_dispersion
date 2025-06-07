import numpy as np
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from matplotlib.ticker import MultipleLocator
import emcee
import os
import pandas as pd
import corner
import warnings
from tqdm import tqdm
from scipy.integrate import simpson
import multiprocessing
import pickle
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import MatplotlibDeprecationWarning



def create_array(start, stop, steps):
    step_width = (stop - start) / steps
    array = np.array(np.arange(start + step_width, stop + step_width, step_width))
    return array


astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/T1_T5.csv')
astro_table["F_array"] = [np.array([row[f"F0_{i}"] for i in range(1, 11)]) for row in astro_table]

ru_table = astro_table[astro_table['Label'] == 'ru']
yin_table = astro_table[astro_table['Label'] == 'yin']

plt.figure(figsize = (10,8))
for ru in ru_table:
    time_array = create_array(ru['Start_Time'], ru['End_Time'], 10)
    time_array = time_array - ru['Start_Time']
    time_array = time_array * 1000
    # plt.errorbar(time_array, ru['F_array'], fmt = 'o', markersize = 10, label = )
    plt.plot(time_array, ru['F_array'], lw = 8)
    
plt.tick_params(labelsize = 18, direction  = 'in', length = 8, width = 1.9)
plt.xlabel('Time (ms)', fontsize = 23)
plt.ylabel('F0', fontsize = 23)
plt.title('Ru', fontsize = 23)
plt.show()
    





plt.figure(figsize = (10,8))
for yin in yin_table:
    time_array = create_array(yin['Start_Time'], yin['End_Time'], 10)
    time_array = time_array - yin['Start_Time']
    time_array = time_array * 1000
    # plt.errorbar(time_array, ru['F_array'], fmt = 'o', markersize = 10, label = )
    plt.plot(time_array, yin['F_array'], lw = 8)
    
plt.tick_params(labelsize = 18, direction  = 'in', length = 8, width = 1.9)
plt.xlabel('Time (ms)', fontsize = 23)
plt.ylabel('F0', fontsize = 23)
plt.title('Yin', fontsize = 23)
plt.show()
    

































