from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.ticker import MultipleLocator


def plot_mds(mds_coords, label, color,  flip_x, flip_y):
    if len(mds_coords) == 0:
        print(f"No data to plot for {label}.")
        return
    if flip_x:
        mds_coords[0] = -mds_coords[0]
    if flip_y:
        mds_coords[1] =  -mds_coords[1]
    
    
    if version == 'f0mu_1':
        plt.errorbar(mds_coords[0], mds_coords[1], label=tone_label_dict[label][0], alpha=1.0, markersize = 25, fmt = 'o', mec = 'black',
                     mew = 3, color =tone_label_dict[label][1])
    
    elif version == 'h1h2_1':
        plt.errorbar(mds_coords[0], mds_coords[1], label=tone_label_dict[label][0], alpha=1.0, markersize = 35, fmt = '*', mec = 'black',
                     mew = 2.5, color =tone_label_dict[label][1])
    else:
        plt.errorbar(mds_coords[0], mds_coords[1], label=tone_label_dict[label][0], alpha=1.0, markersize = 25, fmt = 'p', mec = 'black',
                     mew = 3, color =tone_label_dict[label][1])
    

def plot_mds_modify(version, use_mean, niter, flip_x, flip_y):
    if use_mean:
        pkl_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/fitted_mds_new/mds_niter_{niter}_mean_{version}.pkl'
    else:
        pkl_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/fitted_mds_new/mds_niter_{niter}_median_{version}.pkl'
    
    with open(pkl_dir, 'rb') as f:
        print(f'open {pkl_dir}')
        mds_results = pickle.load(f)
    
    # === Extract X-coordinates, apply flipping ===
    labels = ['yin', 'yang', 'shang', 'qu', 'ru']
    x_coords = {}
    for label in labels:
        x = mds_results[label][0]
        if flip_x:
            x = -x
        x_coords[label] = x
    # === Compute pairwise absolute X-distance matrix ===
    distance_matrix = np.zeros((len(labels), len(labels)))
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            distance_matrix[i, j] = np.abs(x_coords[label_i] - x_coords[label_j])

    # === Compute average distance to other labels for each label ===
    avg_x_distances = {}
    for i, label in enumerate(labels):
        # exclude self (i != j)
        total_dist = sum(distance_matrix[i, j] for j in range(len(labels)) if j != i)
        avg_dist = total_dist / (len(labels) - 1)
        avg_x_distances[label] = avg_dist
    
    # Print results
    print("\nAverage X-axis distance to other labels:")
    avg_distance_list = []
    for label, avg_dist in avg_x_distances.items():
        print(f"{label}: {avg_dist:.4f}")
        avg_distance_list.append(avg_dist)
    avg_distance_list = np.array(avg_distance_list)
    
        

    plt.figure(figsize = (10,8))

    labels = ['yin', 'yang', 'shang', 'qu', 'ru']
    num_labels = len(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))

    for label, color in zip(labels, colors):
        coords = mds_results[label]
        plot_mds(coords, label, color=color, flip_x = flip_x, flip_y = flip_y)

    # plt.text(0.5, 0.9, f'niter = {niter}', x
    #          ha='center', va='center', transform=plt.gca().transAxes, fontsize = 25)

    # plt.title(f'niter = {niter}_{version}', fontsize = 28)
    plt.axhline(y = 0, lw = 2, c = 'grey', ls = '--')
    plt.axvline(x = 0, lw = 2, c = 'grey', ls = '--')
    
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))   # Major ticks every 0.4
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))   # Minor ticks every 0.1
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))   # Major ticks every 0.4
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))   # Minor ticks every 0.1

    plt.tick_params(axis = 'both', direction = 'in', labelsize = 24, width = 3.5, length = 14, which = 'major')
    plt.tick_params(axis = 'both', direction = 'in', labelsize = 24, width = 3.5, length = 10, which = 'minor')

    plt.xlabel("MDS X", fontsize = 30)
    plt.ylabel("MDS Y", fontsize=  30)

    plt.gca().spines['top'].set_linewidth(3)
    plt.gca().spines['right'].set_linewidth(3)
    plt.gca().spines['bottom'].set_linewidth(3)
    plt.gca().spines['left'].set_linewidth(3)
    plt.legend(fontsize = 23, framealpha = 0.4)
    plt.text(s = f'$\mu_x = {np.mean(avg_distance_list):.1f}$', x = 0.2, y = 0.92, horizontalalignment = 'center',
             verticalalignment = 'center', transform =plt.gca().transAxes, fontsize = 40, fontweight = 'bold')
    plt.tight_layout()
    # if not os.path.exists(png_dir):
    #     plt.savefig(png_dir, format = 'png')
    #     print('Saved Figure')
    # plt.xlim(-170, 70)
    # plt.ylim(-170, 70)
    
    plt.xlim(-0.84,0.84)
    plt.ylim(-0.84,0.84)
    
    plt.show()
    

        
        
    
label_list = ['T1', 'T2', 'T3', 'T4', 'T5']
tone_label_dict = {
    'yin': ['T1', '#00b5eb'], # Blue
    'yang': ['T2', '#ffb360'], # Orange
    'shang': ['T3', '#81ffb4'], # Green
    'qu': ['T4', '#8000ff'], # Purple
    'ru': ['T5', '#ff0000'] # Red
    }

colors = [tone_label_dict[keys][1] for keys in tone_label_dict]




use_mean = False
niter = 100000

version = 'f0mu_1'
version = 'HNR05_1'
version = 'h1h2_1'
#

flip_x = False
flip_y = False

if version == 'f0mu_1':
    flip_x = True
    flip_y = False 

if version == 'h1h2_1':
    flip_x = True
    flip_y = False

    

plot_mds_modify(version, use_mean, niter, flip_x, flip_y)
        
        
        
        