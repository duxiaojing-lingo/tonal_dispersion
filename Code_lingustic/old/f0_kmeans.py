from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline







def plot_file(use_mean, smoothing_factor):
    if use_mean:
        saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_50_mean.npz'
    else:
        saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_50_median.npz'
    
    data_loaded = np.load(saving_location)
    
    counter = 0
    colors = np.array([[5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
       [1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00],
       [5.03921569e-01, 9.99981027e-01, 7.04925547e-01, 1.00000000e+00],
       [1.00000000e+00, 7.00543038e-01, 3.78411050e-01, 1.00000000e+00],
       [1.96078431e-03, 7.09281308e-01, 9.23289106e-01, 1.00000000e+00]
       ])
    
    plt.figure(figsize=(10, 8))    
    while counter <= 4:
        x = data_loaded['tone_x'][counter]
        y = data_loaded['tone_y'][counter]
        label_dd = data_loaded['label_list'][counter]
        color = colors[counter]
        
        
        spline = UnivariateSpline(x, y, s=smoothing_factor)
        x_smooth = np.linspace(np.min(x), np.max(x), 200)
        y_smooth = spline(x_smooth)
        
        plt.plot(x_smooth, y_smooth, color= color, label = label_dd, lw = 8)
        plt.plot(x, y, color='black', lw=2, alpha = 0.2)
        
        
        counter  += 1
        
    plt.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=8)
    plt.xlabel(r"$\mathrm{Time [ms]}$", fontsize=25)
    plt.ylabel(r"$f_0 \,\, \mathrm{[Hz]}$", fontsize=25)
    plt.ylim(100,300)
    plt.legend(fontsize=25)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)    
    plt.tight_layout()
    plt.show()







use_mean = True
smoothing_factor = 10000


plot_file(use_mean, smoothing_factor)






































