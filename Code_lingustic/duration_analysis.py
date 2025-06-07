from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.optimize import curve_fit
import pickle


def duration_analysis(astro_table, density_value):
    unique_labels = np.unique(astro_table['Label'])
    tables_dict = {label: astro_table[astro_table['Label'] == label] for label in unique_labels}
    
    tone_x = []
    tone_y = []
    std_y_value = []
    label_list = []
    fitted_spline_dict = {}
    
    
    
    # ------------------
    # Plot Plot ALL Segments
    # ------------------
    for label, table in tables_dict.items():
        print(f'start {label} ---------------------')
        unique_segments = np.unique(np.vstack([table['seg_Start'], table['seg_End']]).T, axis=0)
        label_array = []
        for seg_start, seg_end in unique_segments:            
            mask = (table['seg_Start'] == seg_start) & (table['seg_End'] == seg_end)
            segment_f0 = np.array(table['strF0'][mask])
            duration_value = np.max(table['t_ms'][mask]) - np.min(table['t_ms'][mask])

            t_value = table['t_ms'][mask] - np.min(table['t_ms'][mask])
            t_value = t_value / np.max(t_value)       
            spline = UnivariateSpline(t_value, segment_f0, s=0.5)
            t_smooth = np.arange(0, 1, 0.005)
            spline_y = spline(t_smooth)

        
            if label == 'qu':
                if spline_y[0] > 125 and spline_y[0] < 300:
                    label_array.append(duration_value)
            if label == 'ru':
                if spline_y[0] > 125 and spline_y[0] < 300:
                    label_array.append(duration_value)

            if label == 'shang':
                if spline_y[0] > 125 and spline_y[0] < 320:
                    label_array.append(duration_value)

            if label == 'yang':
                if spline_y[0] > 170:
                    label_array.append(duration_value)
    
            if label == 'yin':
                if spline_y[0] > 150 and spline_y[0] < 320:
                    label_array.append(duration_value)
                    
        print(len(label_array))
        print(label_array)
        fitted_spline_dict[label] = label_array
    

        
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)    
    
    fitted_gaussians = {}
    for label, subarrays in fitted_spline_dict.items():
        try:
            flat_data = np.array(subarrays)
            print(flat_data)
            if flat_data.size == 0:
                print(f"Skipping {label} (flattened data is empty)")
                continue                
    
            flat_data = flat_data[~np.isnan(flat_data)]            
            hist_values, bin_edges = np.histogram(flat_data, bins=30, density= density_value)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            initial_guess = [np.max(hist_values), np.median(flat_data), np.std(flat_data)]     
            if label in ['yang', 'yin']:
                A_fit = 0
                mu_fit = np.median(flat_data)
                sigma_fit = np.median(flat_data)
            else:
                popt, pcov = curve_fit(gaussian, bin_centers, hist_values, p0=initial_guess)
                A_fit, mu_fit, sigma_fit = popt
            
            fitted_gaussians[label] = (A_fit, mu_fit, sigma_fit)
    
            plt.figure(figsize=(10, 8))
            plt.hist(flat_data, bins=30, density= density_value, alpha=1.0, color='#a2d2ff', edgecolor='black', label="Data Histogram")  
            
            x_fit = np.linspace(min(flat_data), max(flat_data), 1000)
            y_fit = gaussian(x_fit, *popt)
            plt.plot(x_fit, y_fit, color='red', linewidth=2, label="Gaussian Fit")
            
            plt.axvline(mu_fit, color='#FF70A5', linewidth=5, label=f'Fitted Mean: {mu_fit:.4f}')
            plt.axvline(np.percentile(flat_data, 16), color='#FF70A5', linestyle='dashed', linewidth=5, label='16th Percentile')
            plt.axvline(np.percentile(flat_data, 84), color='#FF70A5', linestyle='dashed', linewidth=5, label='84th Percentile')
            plt.tick_params(axis='both', direction='in', labelsize=22, width=3.5, length=14, right = True, top = True)
            for spine in plt.gca().spines.values():
                spine.set_linewidth(3) 
                
            plt.title(f"{label}", fontsize=25)
            plt.xlabel('f0 [Hz]', fontsize=25)
            plt.ylabel('Density', fontsize=25)
            plt.legend(fontsize=  15)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/duration/{label}_hist_self.png', dpi = 300)
            plt.show()
            
            
        except Exception as e:
            print(f"Error processing {label}: {e}")
    
    plt.figure(figsize=(10, 8))
    colors = np.array([[5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
       [1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00],
       [5.03921569e-01, 9.99981027e-01, 7.04925547e-01, 1.00000000e+00],
       [1.00000000e+00, 7.00543038e-01, 3.78411050e-01, 1.00000000e+00],
       [1.96078431e-03, 7.09281308e-01, 9.23289106e-01, 1.00000000e+00]
       ])
    
    x_fit_global = np.linspace(100, max(flat_data), 1000)
    counter = 0
    for label, (A_fit, mu_fit, sigma_fit) in fitted_gaussians.items():
        y_fit_global = gaussian(x_fit_global, A_fit, mu_fit, sigma_fit)
        plt.plot(x_fit_global, y_fit_global, linewidth=5, label=label, color = colors[counter])
        counter += 1        
    plt.xlabel(r"$f_0$", fontsize=25)
    plt.ylabel("Density", fontsize=25)
    plt.legend(fontsize = 25)
    plt.grid(True)
    plt.tick_params(axis='both', direction='in', labelsize=22, width=3.5, length=15, right = True, top = True)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)    
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/duration/{label}_hist_all.png', dpi = 300)
    plt.show()

    labels = list(fitted_gaussians.keys())
    sigma_values = [fitted_gaussians[label][2] for label in labels]
    mu_values = [fitted_gaussians[label][1] for label in labels]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sigma_values, color=colors[:len(labels)], edgecolor='black', linewidth=3)
    plt.ylabel(r"$\sigma$ (Standard Deviation)", fontsize=25)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', direction='in', labelsize=22, width=3.5, length=7, right = True, top = True)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)   
    plt.title("$\sigma$ of Fitted Gaussians", fontsize=22)
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/duration/{label}_hist_sigma.png', dpi = 300)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, mu_values, color=colors[:len(labels)], edgecolor='black', linewidth=3)
    plt.ylabel(r"$\mu$ (Mean)", fontsize=25)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title("$\mu$ of Fitted Gaussians", fontsize=22)
    plt.tick_params(axis='both', direction='in', labelsize=22, width=3.5, length=7, right = True, top = True)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)   
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/duration/{label}_hist_mu.png', dpi = 300)
    plt.show()
    
    with open(f'/Users/emilydu/Code/Code_lingustic/Data/data/duration/duration_mu_sigma.pkl', 'wb') as f:
        pickle.dump(fitted_gaussians, f)
        
        



num_person = 'both'
density_value = True

astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/pure_mono_tone.csv')

if num_person == 'jy':
    astro_table = astro_table[astro_table['\ufeffFilename'] == 'jy_pure_mono.mat']
    print('jy')
elif num_person == 'jz':
    astro_table = astro_table[astro_table['\ufeffFilename'] == 'jz_pure_mono.mat']
    print('jz')
else:
    print('Both')

    
duration_analysis(astro_table, density_value)


    
    



