from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os

def compute_f0_params(astro_table):
    unique_segments = np.unique(np.vstack([astro_table['seg_Start'], astro_table['seg_End']]).T, axis=0)


    mean_table = Table(names=['seg_Start', 'seg_End', 'mean_strF0'], 
                       dtype=[astro_table['seg_Start'].dtype, astro_table['seg_End'].dtype, astro_table['strF0'].dtype])

    for seg_start, seg_end in unique_segments:
        mask = (astro_table['seg_Start'] == seg_start) & (astro_table['seg_End'] == seg_end)
        mean_strF0 = np.mean(astro_table['strF0'][mask])
        mean_table.add_row([seg_start, seg_end, mean_strF0])

    return mean_table

def compute_f0_params(astro_table):

    unique_segments = np.unique(np.vstack([astro_table['seg_Start'], astro_table['seg_End']]).T, axis=0)
    result_table = Table(names=['seg_Start', 'seg_End', 'duration', 'onset_F0', 'offset_F0', 'mean_F0', 'delta_F0'],
                         dtype=[astro_table['seg_Start'].dtype, astro_table['seg_End'].dtype,
                                astro_table['strF0'].dtype, astro_table['strF0'].dtype, 
                                astro_table['strF0'].dtype, astro_table['strF0'].dtype, astro_table['strF0'].dtype])

    for seg_start, seg_end in unique_segments:
        mask = (astro_table['seg_Start'] == seg_start) & (astro_table['seg_End'] == seg_end)
        segment_f0 = astro_table['strF0'][mask]

        if len(segment_f0) > 0:
            segment_f0 = np.array(segment_f0)

            n = len(segment_f0)
            onset_n = max(1, int(n * 0.10))
            offset_n = max(1, int(n * 0.10)) 

            onset_F0 = np.mean(segment_f0[:onset_n])
            offset_F0 = np.mean(segment_f0[-offset_n:])
            mean_F0 = np.mean(segment_f0)
            delta_F0 = np.max(segment_f0) - np.min(segment_f0)
            duration = seg_end - seg_start 
            result_table.add_row([seg_start, seg_end, duration, onset_F0, offset_F0, mean_F0, delta_F0])

    return result_table



def apply_mds(f0_table, niter, max_per_iter, duration_add):
    if len(f0_table) == 0:
        return np.array([])

    if duration_add:
        print('Duration Add')
        feature_matrix = np.vstack([f0_table['onset_F0'], f0_table['offset_F0'], 
                                    f0_table['mean_F0'], f0_table['delta_F0']], f0_table['duration']).T
    else:
        feature_matrix = np.vstack([f0_table['onset_F0'], f0_table['offset_F0'], 
                                    f0_table['mean_F0'], f0_table['delta_F0']]).T

    mds = MDS(n_components=2, random_state = None, dissimilarity='euclidean', n_init = niter, max_iter = max_per_iter)
    reduced_coords = mds.fit_transform(feature_matrix)

    return reduced_coords


def apply_mds_single_point(f0_tables_dict, niter, max_per_iter, duration_add):
    mean_vectors = []
    labels = []

    for label, f0_table in f0_tables_dict.items():
        if duration_add:
            print('Duration add')
            # mean_vector = [
            #     np.median(f0_table['onset_F0']),
            #     np.median(f0_table['offset_F0']),
            #     np.median(f0_table['mean_F0']),
            #     np.median(f0_table['delta_F0']),
            #     np.median(f0_table['duration']),
            # ]
            
            mean_vector = [
                np.mean(f0_table['onset_F0']),
                np.mean(f0_table['offset_F0']),
                np.mean(f0_table['mean_F0']),
                np.mean(f0_table['delta_F0']),
                np.mean(f0_table['duration']),
            ]
            
        else:
            mean_vector = [
                np.median(f0_table['onset_F0']),
                np.median(f0_table['offset_F0']),
                np.median(f0_table['mean_F0']),
                np.median(f0_table['delta_F0']),
            ]
            
        mean_vectors.append(mean_vector)
        labels.append(label)

    mean_vectors = np.array(mean_vectors)
    mds = MDS(n_components=2, random_state= None, dissimilarity='euclidean', n_init=niter, max_iter=max_per_iter)
    reduced_coords = mds.fit_transform(mean_vectors)
    mds_results_single = {labels[i]: reduced_coords[i] for i in range(len(labels))}
    return mds_results_single


def plot_mds(mds_coords, label, mds_single, color):
    if len(mds_coords) == 0:
        print(f"No data to plot for {label}.")
        return
    
    if mds_single:
        plt.errorbar(mds_coords[0], mds_coords[1], label=label, alpha=1.0, markersize = 25, fmt = 'o', mec = 'black',
                     mew = 4, color =color)
    
    else:
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1], label=label, alpha=0.7)



def mds_analysis(astro_table, mds_single, niter, max_per_iter, duration_add):
    unique_labels = np.unique(astro_table['Label'])
    tables_dict = {label: astro_table[astro_table['Label'] == label] for label in unique_labels}
    
    f0_tables_dict = {label: compute_f0_params(tables_dict[label]) for label in tables_dict}


    print(f'start MDS')
    if mds_single:
        mds_results = apply_mds_single_point(f0_tables_dict, niter, max_per_iter, duration_add)
    else:
        mds_results = {label: apply_mds(f0_tables_dict[label], niter, max_per_iter, duration_add) for label in f0_tables_dict}
    
    
    return f0_tables_dict, mds_results
    
    
# astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/jiyuan_tier_3_middle_chinese.csv')
astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/jiyuan_middle_chinese.csv')
mds_single = True
duration_add = True
niter = 100000
max_per_iter = 10000
f0_table_dict, mds_results = mds_analysis(astro_table, mds_single, niter, max_per_iter, duration_add)


if duration_add:
    png_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/mds_niter_{niter}_duration_add.png'
else:
    png_dir = f'/Users/emilydu/Code/Code_lingustic/Data/data/mds/mds_niter_{niter}.png'

    
plt.figure(figsize = (10,8))

labels =  ['qu', 'yin', 'shang', 'yang', 'ru']
num_labels = len(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))

for label, color in zip(labels, colors):
    coords = mds_results[label]
    plot_mds(coords, label, mds_single, color=color)

# plt.text(0.5, 0.9, f'niter = {niter}', x
#          ha='center', va='center', transform=plt.gca().transAxes, fontsize = 25)
if duration_add:
    plt.title(f'niter = {niter}, duration add', fontsize = 28)
else:
    plt.title(f'niter = {niter}', fontsize = 28)
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


    
    

save_dir = "/Users/emilydu/Code/Code_lingustic/Data/f0_computed_table"
for table_name, astro_table in f0_table_dict.items():
    df = astro_table.to_pandas()
    file_path = os.path.join(save_dir, f"{table_name}.xlsx")    
    df.to_excel(file_path, index=False, engine='openpyxl')

    print(f"Saved {table_name} to {file_path}")











