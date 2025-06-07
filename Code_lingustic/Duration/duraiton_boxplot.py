from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.optimize import curve_fit


def duration_analysis(astro_table_list, label_list):

    duration_save_dict = {}

    # ------------------
    # Plot Plot ALL Segments
    # ------------------
    for table, label_dd in zip(astro_table_list, label_list):
        unique_segments = np.unique(np.vstack([table['seg_Start'], table['seg_End']]).T, axis=0)
        
        duration_list = []
        for seg_start, seg_end in unique_segments:            
            mask = (table['seg_Start'] == seg_start) & (table['seg_End'] == seg_end)            
            segment_f0 = np.array(table['strF0'][mask])
            duration_value = np.max(table['t_ms'][mask]) - np.min(table['t_ms'][mask])
            
            t_value = table['t_ms'][mask] - np.min(table['t_ms'][mask])
            t_value = t_value / np.max(t_value)       
            spline = UnivariateSpline(t_value, segment_f0, s=0.5)
            t_smooth = np.arange(0, 1, 0.005)
            spline_y = spline(t_smooth)
        

            if spline_y[0] > 125 and spline_y[0] < 300:
                duration_list.append(duration_value)
            
        duration_list =  np.array(duration_list)
        duration_save_dict[label_dd] = duration_list
        
            
    fig, ax = plt.subplots(figsize=(10,8))    
    data_to_plot = [duration_save_dict[label] for label in label_list]    
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
    
    label_list = [r'T5$^*$', 'T5-1', 'T5-2']
    ax.set_xticks(np.arange(1, len(label_list) + 1))
    ax.set_xticklabels(label_list, fontsize=30)
    ax.set_ylabel('Duration [ms]', fontsize = 30)
    plt.tick_params(axis='y', direction='in', labelsize=28, width=3.5, length=12, right = True, top = True)
    ax.tick_params(axis='x', pad=7, direction = 'in', width = 3.5, length = 10)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3) 
    ax.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig('/Users/emilydu/Code/Code_lingustic/Data/plots/duration_violin/duration_violin.pdf', format = 'pdf')
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
duration_analysis(astro_table_list, label_list)


    
    



