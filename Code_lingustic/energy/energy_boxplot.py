from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.optimize import curve_fit


def energ_analysis(astro_table_list, label_list):

    energy_save_dict = {}

    # ------------------
    # Plot Plot ALL Segments
    # ------------------
    for table, label_dd in zip(astro_table_list, label_list):
        unique_segments = np.unique(np.vstack([table['seg_Start'], table['seg_End']]).T, axis=0)
        
        energy_list = []
        for seg_start, seg_end in unique_segments:            
            mask = (table['seg_Start'] == seg_start) & (table['seg_End'] == seg_end)            
            segment_f0 = np.array(table['strF0'][mask])
            energy_segment_value = np.array(table['Energy'][mask])
            
            t_value = table['t_ms'][mask] - np.min(table['t_ms'][mask])
            t_value = t_value / np.max(t_value)       
            spline = UnivariateSpline(t_value, segment_f0, s=0.5)
            t_smooth = np.arange(0, 1, 0.005)
            spline_y = spline(t_smooth)

            if spline_y[0] > 125 and spline_y[0] < 300:
                energy_list.extend(energy_segment_value)
            
        energy_list =  np.array(energy_list)
        energy_save_dict[label_dd] = energy_list
                
            
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)    
    
    for label, flat_data in energy_save_dict.items():             
        flat_data = flat_data[~np.isnan(flat_data)]            
        hist_values, bin_edges = np.histogram(flat_data, bins=30, density= True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # initial_guess = [np.max(hist_values), np.median(flat_data), np.std(flat_data)]            
        # popt, pcov = curve_fit(gaussian, bin_centers, hist_values, p0=initial_guess)            
        # A_fit, mu_fit, sigma_fit = popt
        

        plt.figure(figsize=(10, 8))
        plt.hist(flat_data, bins=25, density= True, alpha=1.0, color='#a2d2ff', edgecolor='black', label="Data Histogram")  
        
        x_fit = np.linspace(min(flat_data), max(flat_data), 1000)
        # y_fit = gaussian(x_fit, *popt)
        # plt.plot(x_fit, y_fit, color='#FF337E', linewidth=5, label="Gaussian Fit")
        
        # plt.axvline(mu_fit, color='#FF70A5', linewidth=5, label=f'Fitted Mean: {mu_fit:.4f}')
        plt.axvline(np.percentile(flat_data, 16), color='#FF70A5', linestyle='dashed', linewidth=5, label='16th Percentile')
        plt.axvline(np.percentile(flat_data, 84), color='#FF70A5', linestyle='dashed', linewidth=5, label='84th Percentile')
        plt.tick_params(axis='both', direction='in', labelsize=22, width=3.5, length=13, right = True, top = True)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(3) 
            
        plt.title(f"{label}", fontsize=25)
        plt.xlabel('Energy', fontsize=25)
        plt.ylabel('Density', fontsize=25)
        plt.legend(fontsize=  15)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    
    
    
    fig, ax = plt.subplots(figsize=(10,8))    
    data_to_plot = [energy_save_dict[label] for label in label_list]    
    vp = ax.violinplot(
        data_to_plot,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )
    
    for body in vp['bodies']:
        body.set_facecolor('#a2d2ff')     # Fill color (shaded area)
        body.set_edgecolor('#ffafcc')     # Edge color
        body.set_alpha(0.7)
        body.set_linewidth(7)
    
    vp['cmedians'].set_color('#990038')
    vp['cmins'].set_color('#990038')
    vp['cmaxes'].set_color('#990038')
    vp['cbars'].set_color('#990038')
    vp['cmedians'].set_linewidth(4)
    vp['cmins'].set_linewidth(4)
    vp['cmaxes'].set_linewidth(4)
    vp['cbars'].set_linewidth(4)

    ax.set_xticks(np.arange(1, len(label_list) + 1))
    ax.set_xticklabels(label_list, fontsize=30)
    ax.set_ylabel('Duration (ms)', fontsize = 25)
    plt.tick_params(axis='y', direction='in', labelsize=25, width=3.5, length=10, right = True, top = True)
    ax.tick_params(axis='x', pad=7)  # Adjusts space between tick and label
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3) 
    ax.grid(alpha = 0.5)
    plt.tight_layout()
    # plt.savefig('/Users/emilydu/Code/Code_lingustic/Data/plots/duration_violin/duration_violin.pdf', format = 'pdf')
    plt.show()



astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/pure_mono_tone.csv')
astro_table_ru = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/ru_variant_tone.csv')
astro_table_sandhi = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/tone_sandhi_tone.csv')

mask = (astro_table_ru['Label'] == 'ru_1') | (astro_table_ru['Label'] == 'ru_2')
astro_table_ru['Label'] = np.where(mask, 'ru', astro_table_ru['Label'])
mask = (astro_table_sandhi['Label'] == 'ru_1') | (astro_table_sandhi['Label'] == 'ru')
astro_table_sandhi['Label'] = np.where(mask, 'ru', astro_table_sandhi['Label'])

astro_table = astro_table[astro_table['Label'] == 'ru']
astro_table_ru = astro_table_ru[astro_table_ru['Label'] == 'ru']
astro_table_sandhi = astro_table_sandhi[astro_table_sandhi['Label'] == 'ru']


astro_table_list = [astro_table, astro_table_ru, astro_table_sandhi]
label_list = [r'$\mathrm{Ru}^*$', 'Ru-1', 'Ru-2']
energ_analysis(astro_table_list, label_list)


    
    



