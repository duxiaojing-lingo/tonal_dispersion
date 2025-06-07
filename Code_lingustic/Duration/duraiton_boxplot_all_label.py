from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.optimize import curve_fit


def duration_analysis(astro_table_list, label_list, colors):

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
            if label_dd == 'T4':
                if spline_y[0] > 125 and spline_y[0] < 300:
                    duration_list.append(duration_value)
            if label_dd == 'T5':
                if spline_y[0] > 125 and spline_y[0] < 300:
                    duration_list.append(duration_value)
            if label_dd == 'T3':
                if spline_y[0] > 125 and spline_y[0] < 320:
                    duration_list.append(duration_value)
            if label_dd == 'T2':
                if spline_y[0] > 170:
                    duration_list.append(duration_value)
            if label_dd == 'T1':
                if spline_y[0] > 150 and spline_y[0] < 320:
                    duration_list.append(duration_value)





        duration_list =  np.array(duration_list)
        duration_save_dict[label_dd] = duration_list
        
            
    fig, ax = plt.subplots(figsize=(10,8))    
    print(label_list)
    data_to_plot = [duration_save_dict[label] for label in label_list]    
    vp = ax.violinplot(
        data_to_plot,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )
    
    

        
    
    for body, color in zip(vp['bodies'], colors):
        body.set_facecolor(color)
        # body.set_edgecolor('#ffafcc')
        body.set_edgecolor(color)

        body.set_alpha(0.5)
        body.set_linewidth(5)
        
    
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
    ax.set_ylabel('Duration [ms]', fontsize = 30)
    plt.tick_params(axis='y', direction='in', labelsize=28, width=3.5, length=12, right = True, top = True)
    ax.tick_params(axis='x', pad=7, direction = 'in', width = 3.5, length = 12, labelsize = 30)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3) 
    ax.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig('/Users/emilydu/Code/Code_lingustic/Data/plots/duration_violin/duration_violin_all_label.pdf', format = 'pdf')
    plt.show()



astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/pure_mono_tone.csv')
astro_table_qu = astro_table[astro_table['Label'] == 'qu']
astro_table_ru = astro_table[astro_table['Label'] == 'ru']
astro_table_shang = astro_table[astro_table['Label'] == 'shang']
astro_table_yang = astro_table[astro_table['Label'] == 'yang']
astro_table_yin = astro_table[astro_table['Label'] == 'yin']




astro_table_list = [astro_table_qu, astro_table_ru, astro_table_shang, astro_table_yang, astro_table_yin]
astro_table_list = [astro_table_yin, astro_table_yang, astro_table_shang, astro_table_qu, astro_table_ru]

# label_list = ['Qu', 'Ru', 'Shang', 'Yang', 'Yin']
# colors = np.array([[5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
#    [1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00],
#    [5.03921569e-01, 9.99981027e-01, 7.04925547e-01, 1.00000000e+00],
#    [1.00000000e+00, 7.00543038e-01, 3.78411050e-01, 1.00000000e+00],
#    [1.96078431e-03, 7.09281308e-01, 9.23289106e-01, 1.00000000e+00]
#    ])
    
label_list = ['T1', 'T2', 'T3', 'T4', 'T5']
tone_label_dict = {
    'yin': ['T1', '#00b5eb'], # Blue
    'yang': ['T2', '#ffb360'], # Orange
    'shang': ['T3', '#81ffb4'], # Green
    'qu': ['T4', '#8000ff'], # Purple
    'ru': ['T5', '#ff0000'] # Red
    }

colors = [tone_label_dict[keys][1] for keys in tone_label_dict]



duration_analysis(astro_table_list, label_list, colors)


    
    



