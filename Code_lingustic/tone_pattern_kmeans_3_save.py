from astropy.table import Table, Column
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from scipy.stats import zscore
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colors as mpl_colors
from matplotlib.colors import ListedColormap
from astropy.table import vstack
from collections import defaultdict
import math

def f0_kmeans(astro_table, time_normalise):
    unique_labels = np.unique(astro_table['Label'])
    tables_dict = {label: astro_table[astro_table['Label'] == label] for label in unique_labels}    
    fitted_spline_dict = {}
    
    # ------------------
    # Plot Plot ALL Segments
    # ------------------
    
    
    for label, table in tables_dict.items():
        plt.figure(figsize = (10,8))
        unique_segments = np.unique(np.vstack([table['seg_Start'], table['seg_End']]).T, axis=0)

        fitted_spline = []
        
        for seg_start, seg_end in unique_segments:            
            mask = (table['seg_Start'] == seg_start) & (table['seg_End'] == seg_end)
            segment_f0 = np.array(table['strF0'][mask])
            t_value = table['t_ms'][mask] - np.min(table['t_ms'][mask])
            t_value = t_value / np.max(t_value)       
            spline = UnivariateSpline(t_value, segment_f0, s=0.1)
            # t_smooth = np.linspace(0, 1, 1000)
            t_smooth = np.linspace(0.05, 0.95, 10)

            spline_y = spline(t_smooth)
            if label == 'qu':
                if spline_y[0] > 125 and spline_y[0] < 300:
                    fitted_spline.append((t_smooth, spline_y, seg_start, seg_end))  
                    plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                    plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    

            if label == 'ru':
                if spline_y[0] > 125 and spline_y[0] < 250:
                    fitted_spline.append((t_smooth, spline_y, seg_start, seg_end))  
                    plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                    plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    

            if label == 'shang':
                if spline_y[0] > 125 and spline_y[0] < 320:
                    fitted_spline.append((t_smooth, spline_y, seg_start, seg_end))  
                    plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                    plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    

            if label == 'yang':
                if spline_y[0] > 170:
                    fitted_spline.append((t_smooth, spline_y, seg_start, seg_end))  
                    plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                    plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    

            if label == 'yin':
                if spline_y[0] > 150 and spline_y[0] < 320:
                    fitted_spline.append((t_smooth, spline_y, seg_start, seg_end))  
                    plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                    plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    
            if label == 'ru_*':
                if spline_y[0] > 125 and spline_y[0] < 300:
                    fitted_spline.append((t_smooth, spline_y, seg_start, seg_end))  
                    plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                    plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    


        fitted_spline_dict[label] = fitted_spline        
        plt.title(label)
        plt.show()

    
    features = []
    segment_info = []
    
    for label, splines in fitted_spline_dict.items():
        for (t_smooth, spline_y, seg_start, seg_end) in splines:
            features.append(spline_y)
            segment_info.append((label, seg_start, seg_end))
            
    features = np.array(features)
    features_z = (features - np.mean(features)) / np.std(features)
    features= features_z

                                                                     
                                                                     
    n_clusters = 6
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans_model.fit_predict(features)
    
    mds = MDS(n_components=2, random_state=42)
    features_2d = mds.fit_transform(features)
    

    
    custom_colors = np.array([[5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
        [1.00000000e-01, 5.87785252e-01, 9.51056516e-01, 1.00000000e+00],
        [3.00000000e-01, 9.51056516e-01, 8.09016994e-01, 1.00000000e+00],
        [7.00000000e-01, 9.51056516e-01, 5.87785252e-01, 1.00000000e+00],
        [1.00000000e+00, 5.87785252e-01, 3.09016994e-01, 1.00000000e+00],
        [1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00]])
    
   
    
    custom_cmap = ListedColormap(custom_colors)
    
    plt.figure(figsize=(14,10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels,
                          cmap=custom_cmap, s=80)
    plt.title('MDS Projection of Fitted Spline Clusters', fontsize=25)
    plt.xlabel('MDS Dimension 1', fontsize=25)
    plt.ylabel('MDS Dimension 2', fontsize=25)
    plt.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=15, right=True, top=True)
    cbar = plt.colorbar(scatter, cmap=custom_cmap)
    cbar.set_label('Cluster Label', fontsize=25)
    cbar.ax.tick_params(labelsize=20)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/data/tone_pattern/mds_3.pdf')
    plt.show()


    # ------------------
    # Plot 1: Plot all segments colored by their cluster assignment using custom colors
    # ------------------
    fig, ax = plt.subplots(figsize=(16, 10))
    counter = 0
    flat_segments = []
    
    for label, splines in fitted_spline_dict.items():
        for seg_idx, (t_smooth, spline_y, seg_start, seg_end) in enumerate(splines):
            flat_segments.append((t_smooth, spline_y))
            cluster = cluster_labels[counter]
            color = custom_colors[cluster]
            ax.plot(t_smooth, spline_y, color=color, alpha=0.3)
            counter += 1
    
    norm = mpl_colors.Normalize(vmin=0, vmax=n_clusters - 1)
    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, ticks=range(n_clusters))
    cbar.set_label('Cluster Label', fontsize=25)
    cbar.ax.tick_params(labelsize=24)
    
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=15, right=True, top=True)
    ax.set_title("All Segments Colored by Cluster", fontsize=25)
    ax.set_xlabel("Normalized time", fontsize=25)
    ax.set_ylabel("f0", fontsize=25)
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/data/tone_pattern/all_seg_3.pdf')
    plt.show()
    
    # ------------------
    # Plot 2: For each cluster, compute the mean y values and plot the mean curve with error bars
    # ------------------
    fig, ax = plt.subplots(figsize=(16, 10))

    for cl in range(n_clusters):
        indices = np.where(cluster_labels == cl)[0]
        if len(indices) == 0:
            continue
        y_arrays = np.array([flat_segments[i][1] for i in indices])
        mean_y = np.mean(y_arrays, axis=0)
        t_smooth = flat_segments[indices[0]][0]
        plt.plot(t_smooth, mean_y, label=f'Cluster {cl}', linewidth=10,
                 color=custom_colors[cl])
        plt.errorbar(t_smooth, mean_y, markersize=30,
                     color=custom_colors[cl], fmt='o')
    cbar = fig.colorbar(sm, ax=ax, ticks=range(n_clusters))
    cbar.set_label('Cluster Label', fontsize=25)
    cbar.ax.tick_params(labelsize=24)
    
    plt.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=15, right=True, top=True)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)
    plt.title("Mean Spline Curve for Each Cluster", fontsize=25)
    plt.xlabel("Normalized time", fontsize=25)
    plt.ylabel("Mean f0", fontsize=25)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/data/tone_pattern/all_seg_mean_3.pdf')
    plt.show()


    return fitted_spline_dict, segment_info, cluster_labels
    
    
    



time_normalise = True

    
astro_table_mono = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/pure_mono_tone.csv')
astro_table_ru = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/ru_variant_tone.csv')
astro_table_sandhi = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/tone_sandhi_tone.csv')

mask = (astro_table_ru['Label'] == 'ru_1') | (astro_table_ru['Label'] == 'ru_2')
astro_table_ru['Label'] = np.where(mask, 'ru_*', astro_table_ru['Label'])

mask = (astro_table_sandhi['Label'] == 'ru_1') | (astro_table_sandhi['Label'] == 'ru')
astro_table_sandhi['Label'] = np.where(mask, 'ru_*', astro_table_sandhi['Label'])

astro_table = vstack([astro_table_mono, astro_table_ru, astro_table_sandhi])


fitted_spline_dict, segment_info, cluster_labels = f0_kmeans(astro_table, time_normalise)
astro_table.add_column(Column(np.full(len(astro_table), -1), name='Cluster_kmeans'))
for (seg_label, seg_start, seg_end), cl in zip(segment_info, cluster_labels):
    mask = ((astro_table['Label'] == seg_label) & 
            (astro_table['seg_Start'] == seg_start) & 
            (astro_table['seg_End'] == seg_end))
    astro_table['Cluster_kmeans'][mask] = cl
    
    

label_cluster_counts = defaultdict(lambda: defaultdict(int))
for (seg_label, seg_start, seg_end), cl in zip(segment_info, cluster_labels):
    label_cluster_counts[seg_label][cl] += 1

labels = sorted(label_cluster_counts.keys())
n_labels = len(labels)
ncols = int(math.ceil(math.sqrt(n_labels)))
nrows = int(math.ceil(n_labels / ncols))

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
axs = axs.flatten()
for i, label in enumerate(labels):
    clusters_dict = label_cluster_counts[label]
    clusters = sorted(clusters_dict.keys())
    counts = [clusters_dict[cl] for cl in clusters]
    
    bars = axs[i].bar(clusters, counts, color='skyblue', alpha=0.8, edgecolor='black', linewidth=2)
    axs[i].set_xlabel('Cluster Label', fontsize=20)
    axs[i].set_ylabel('Count', fontsize=20)
    axs[i].set_title(f'{label}', fontsize=20)
    axs[i].set_xticks(clusters)
    axs[i].tick_params(axis='both', direction='in', labelsize=18, width=2.5, length=7)
    for bar in bars:
        height = bar.get_height()
        axs[i].text(bar.get_x() + bar.get_width()/2, height+0.2, f'{int(height)}', 
                    ha='center', va='bottom', fontsize=16)
    axs[i].set_ylim(0,np.max(counts) + 4)
    for spine in axs[i].spines.values():
        spine.set_linewidth(2)
        
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

    
    

output_path = '/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/all_3_kmeans.csv'
astro_table.write(output_path, format='csv', overwrite=True)
print(f"Updated table saved to {output_path}")



    
    
    
    



