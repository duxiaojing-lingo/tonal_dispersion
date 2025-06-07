from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
import pickle



def compute_f0_params(data_loaded, offset_percentage):

    label_list = data_loaded['label_list']
    tone_y_list = data_loaded['tone_y']
    tone_x_list = data_loaded['tone_x']
    five_tone_y_list = data_loaded['five_tone_y']

    result_dict = {}
    
    for label, tone_y, tone_x, five_tone_y in zip(label_list, tone_y_list, tone_x_list, five_tone_y_list):
        n = len(tone_y)
        n_percent = int(n * offset_percentage)
        onset_F0 = int(np.mean(tone_y[:n_percent]))
        offset_F0 = np.mean(tone_y[-n_percent:])
        mean_F0 = np.mean(tone_y)
        delta_F0 = np.max(tone_y) - np.min(tone_y)
        duration = np.max(tone_x) - np.min(tone_x) 
        
        result_dict[label] = {
            "onset_F0": onset_F0,
            "offset_F0": offset_F0,
            "mean_F0": mean_F0,
            "delta_F0": delta_F0,
            "duration": duration
        }
    
    return result_dict




def apply_mds_single_point(f0_tables_dict, niter, max_per_iter, hnr_header):
    mean_vectors = []
    labels = []

    for label, f0_table in f0_tables_dict.items():
        
        mean_vector = []
        with open(f'/Users/emilydu/Code/Code_lingustic/Data/data/f0/f0_mu_sigma.pkl', 'rb') as f:
            fitted_gaussians = pickle.load(f)
            print('f0 dict')
            # print(fitted_gaussians)        
        all_mu = [v[1] for v in fitted_gaussians.values()]
        max_fo_mu = max(all_mu)
        fo_mu = fitted_gaussians[label][1] / max_fo_mu
        mean_vector.append(fo_mu)  
        
        
        with open(f'/Users/emilydu/Code/Code_lingustic/Data/data/duration/duration_mu_sigma.pkl', 'rb') as f:
            fitted_gaussians = pickle.load(f)
            # print(fitted_gaussians)
        all_mu = [v[1] for v in fitted_gaussians.values()]
        max_fo_mu = max(all_mu)
        duration_mu = fitted_gaussians[label][1] / max_fo_mu
        mean_vector.append(duration_mu)  



        print(f'MDS vecotr dim = {len(mean_vector)}')
        mean_vectors.append(mean_vector)
        labels.append(label)

    mean_vectors = np.array(mean_vectors)
    mds = MDS(n_components=2, random_state= None, dissimilarity='euclidean', n_init=niter, max_iter=max_per_iter)
    reduced_coords = mds.fit_transform(mean_vectors)
    mds_results_single = {labels[i]: reduced_coords[i] for i in range(len(labels))}
    return mds_results_single


def plot_mds(mds_coords, label, color):
    if len(mds_coords) == 0:
        print(f"No data to plot for {label}.")
        return
    
    plt.errorbar(mds_coords[0], mds_coords[1], label=label, alpha=1.0, markersize = 25, fmt = 'o', mec = 'black',
                 mew = 4, color =color)
    


def mds_analysis(niter, max_per_iter, use_mean, smoothing_factor, offset_percentage, hnr_header):

    if use_mean:
        saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_{smoothing_factor}_mean.npz'
    else:
        saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_{smoothing_factor}_median.npz'
    
    print(f"Load {saving_location}")
    data_loaded = np.load(saving_location)
    f0_tables_dict = compute_f0_params(data_loaded, offset_percentage)
    
    mds_results = apply_mds_single_point(f0_tables_dict, niter, max_per_iter, hnr_header)

    
    return f0_tables_dict, mds_results
    

use_mean = False
smoothing_factor = 50
niter = 100000
max_per_iter = 100000
offset_percentage = 0.05
hnr_headers = ['HNR05','HNR15','HNR25','HNR35']
hnr_header = 'HNR35'

f0_table_dict, mds_results = mds_analysis(niter, max_per_iter, use_mean, smoothing_factor, offset_percentage, hnr_header)


if use_mean:
    png_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/fitted_mds_new/mds_niter_{niter}_mean_f0mu_1.png'
else:
    png_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/fitted_mds_new/mds_niter_{niter}_median_f0mu_1.png'
pkl_dir = png_dir[:-3] + 'pkl'


plt.figure(figsize = (10,8))

labels =  ['qu', 'yin', 'shang', 'yang', 'ru']
num_labels = len(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))

for label, color in zip(labels, colors):
    coords = mds_results[label]
    plot_mds(coords, label, color=color)

# plt.text(0.5, 0.9, f'niter = {niter}', x
#          ha='center', va='center', transform=plt.gca().transAxes, fontsize = 25)

plt.title(f'niter = {niter}_v3', fontsize = 28)
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
if not os.path.exists(png_dir):
    plt.savefig(png_dir, format = 'png')
    print('Saved Figure')
plt.show()


with open(pkl_dir, 'wb') as f:
    pickle.dump(mds_results, f)








