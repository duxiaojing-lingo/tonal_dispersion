from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
import pickle


def plot_mds(mds_coords, label, color,  flip_x, flip_y):
    if len(mds_coords) == 0:
        print(f"No data to plot for {label}.")
        return
    if flip_x:
        mds_coords[0] = -mds_coords[0]
    if flip_y:
        mds_coords[1] =  -mds_coords[1]
        
    plt.errorbar(mds_coords[0], mds_coords[1], label=label, alpha=1.0, markersize = 25, fmt = 'o', mec = 'black',
                 mew = 4, color =color)
    
    

def plot_mds_modify(version, use_mean, niter, flip_x, flip_y):
    if use_mean:
        pkl_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/fitted_mds_new/mds_niter_{niter}_mean_{version}.pkl'
    else:
        pkl_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/fitted_mds_new/mds_niter_{niter}_median_{version}.pkl'
    
    with open(pkl_dir, 'rb') as f:
        print(f'open {pkl_dir}')
        mds_results = pickle.load(f)
    
    
    plt.figure(figsize = (10,8))

    labels =  ['qu', 'yin', 'shang', 'yang', 'ru']
    num_labels = len(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))

    for label, color in zip(labels, colors):
        coords = mds_results[label]
        plot_mds(coords, label, color=color, flip_x = flip_x, flip_y = flip_y)

    # plt.text(0.5, 0.9, f'niter = {niter}', x
    #          ha='center', va='center', transform=plt.gca().transAxes, fontsize = 25)

    plt.title(f'niter = {niter}_{version}', fontsize = 28)
    plt.axhline(y = 0, lw = 2, c = 'grey', ls = '--')
    plt.axvline(x = 0, lw = 2, c = 'grey', ls = '--')
    plt.tick_params(axis = 'both', direction = 'in', labelsize = 24, width = 3.5, length = 8)
    plt.xlabel("MDS X", fontsize = 25)
    plt.ylabel("MDS Y", fontsize=  25)

    plt.gca().spines['top'].set_linewidth(3)
    plt.gca().spines['right'].set_linewidth(3)
    plt.gca().spines['bottom'].set_linewidth(3)
    plt.gca().spines['left'].set_linewidth(3)
    plt.legend(fontsize = 23, framealpha = 0.4)
    plt.tight_layout()
    # if not os.path.exists(png_dir):
    #     plt.savefig(png_dir, format = 'png')
    #     print('Saved Figure')
    # plt.xlim(-170, 70)
    # plt.ylim(-170, 70)
    
    # plt.xlim(-85, 70)
    # plt.ylim(-85, 70)
    
    plt.show()
    
    
        
        
use_mean = False
niter = 100000
version = 'v3'


# if version == 'v1':
#     flip_x = True
#     flip_y = False
    
# if version == 'v2':
#     flip_x = False
#     flip_y = False

# if version == 'v3':
#     flip_x = True
#     flip_y = False
    


if version == 'v1':
    flip_x = False
    flip_y = False
    
if version == 'v2':
    flip_x = True
    flip_y = True

if version == 'v3':
    flip_x = False
    flip_y = False
    

plot_mds_modify(version, use_mean, niter, flip_x, flip_y)
        
        
        
        