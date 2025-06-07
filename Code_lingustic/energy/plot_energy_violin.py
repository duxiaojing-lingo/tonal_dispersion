from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d


def plot_f0(astro_table, time_normalise, use_mean, label_all, color_all):
    unique_labels = np.unique(astro_table['Label'])
    tables_dict = {label: astro_table[astro_table['Label'] == label] for label in unique_labels}
    
    tone_x = []
    tone_y = []
    std_y_value = []
    label_list = []
    fitted_spline_dict = {}
    results_dict = {}    
    
    
    
    # ------------------
    # Plot Plot ALL Segments
    # ------------------
    for label, table in tables_dict.items():
        print(f'start {label} ---------------------')
        counter = 0
        unique_segments = np.unique(np.vstack([table['seg_Start'], table['seg_End']]).T, axis=0)

        fitted_spline = []
        segment_info = []
        

                
        if time_normalise:
            print('Time Normalised')
            for seg_start, seg_end in unique_segments:            
                mask = (table['seg_Start'] == seg_start) & (table['seg_End'] == seg_end)
                segment_f0 = np.array(table['strF0'][mask])
                t_value = table['t_ms'][mask] - np.min(table['t_ms'][mask])
                t_value = t_value / np.max(t_value)       
                spline = UnivariateSpline(t_value, segment_f0, s=0.5)
    
                t_smooth = np.arange(0, 1, 0.005)
                spline_y_f0 = spline(t_smooth)
                
                segment_h1h2 = np.array(table['Energy'][mask])
                spline_h1h2 = UnivariateSpline(t_value, segment_h1h2, s = 200)
                interpolator_h1h2 = interp1d(t_value, segment_h1h2, kind='linear', fill_value='extrapolate')
                spline_y_h1h2 = interpolator_h1h2(t_smooth)
                spline_y_h1h2 = spline_h1h2(t_smooth)

                if label == 'ru':
                    if spline_y_f0[0] > 125 and spline_y_f0[0] < 300:
                        fitted_spline.append((t_smooth, spline_y_h1h2))        
                        if np.isnan(segment_h1h2[0]):
                            segment_info.append([seg_start, "#NUM!"])
                        else:
                            segment_info.append([seg_start, round(segment_h1h2[0])])
                counter += 1
                
        fitted_spline = np.array(fitted_spline)  
        
        
        y_values = fitted_spline[:, 1, :]
        
        if use_mean:
            median_y = np.nanmean(y_values, axis=0)
            saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_mean_h1h2.npz'
        else:
            median_y = np.nanmedian(y_values, axis=0)
            saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_median_h1h2.npz'


        std_y = np.nanstd(y_values, axis = 0)
                
        tone_x.append(t_smooth)
        tone_y.append(median_y)
        std_y_value.append(std_y)
        label_list.append(label)
        
        
        if label == 'qu':
            print('qu----------')
            print(np.min(median_y))
            print(std_y[np.argmin(median_y)])
        if label == 'yin':
            print('yin----------')
            print(np.max(median_y))
            print(std_y[np.argmax(median_y)])
            
        fitted_spline_dict[label] = np.array(segment_info)
        
    std_y_value = np.array(std_y_value)
    tone_y = np.array(tone_y)
    tone_x = np.array(tone_x)
    
    
    
    
    
    
    
    # ------------------
    # Plot f0
    # ------------------
    
    import pickle
    import os
    x_y_dict = {}
    for x, y, label_dd, std_individual in zip(tone_x, tone_y, label_list, std_y_value):
        plt.plot(x, y, color=color_all, label=label_all, lw=8)
        if label_dd not in x_y_dict:
            x_y_dict[label_dd] = {}
        x_y_dict[label_dd]['x'] = x
        x_y_dict[label_dd]['y'] = y
    
    # pkl_location = f'/Users/emilydu/Code/Code_lingustic/Data/f0_h1h2_xy_pkl_save/{num_person}/h1h2_xy.pkl'
    # os.makedirs(os.path.dirname(pkl_location), exist_ok=True)
    # with open(pkl_location, 'wb') as f:
    #     pickle.dump(x_y_dict, f) 

  
    
    return fitted_spline_dict, std_y_value, label_list, tone_y


time_normalise = True
use_mean = True



astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/pure_mono_tone.csv')
astro_table_ru = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/ru_variant_tone.csv')
astro_table_sandhi = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/tone_sandhi_tone.csv')
mask = (astro_table_ru['Label'] == 'ru_1') | (astro_table_ru['Label'] == 'ru_2')
astro_table_ru['Label'] = np.where(mask, 'ru', astro_table_ru['Label'])
mask = (astro_table_sandhi['Label'] == 'ru_1') | (astro_table_sandhi['Label'] == 'ru')
astro_table_sandhi['Label'] = np.where(mask, 'ru', astro_table_sandhi['Label'])
astro_table_pure = astro_table[astro_table['Label'] == 'ru']
astro_table_ru = astro_table_ru[astro_table_ru['Label'] == 'ru']
astro_table_sandhi = astro_table_sandhi[astro_table_sandhi['Label'] == 'ru']
astro_table_list = [astro_table_pure,astro_table_ru,  astro_table_sandhi]    
label_list_all = [r'$\mathrm{Ru}^*$', 'Ru-1', 'Ru-2']
    
color_phi = '#FF9224'
color_m = '#3BD03B'
color_alpha = '#AF37FF'    
color_list_all = [color_phi, color_m, color_alpha]
color_list_all = ['#FF146A', '#4AA7FF', '#00E271']

energy_save_dict = []
plt.figure(figsize=(10, 8))    
for astro_table, label_all, color_all in zip(astro_table_list, label_list_all, color_list_all):
    fitted_spline_dict, std_y, label_list, tone_y = plot_f0(astro_table, time_normalise, use_mean, label_all, color_all)
    energy_save_dict.append(tone_y[0])

plt.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=8)
plt.xlabel(r"Normalised Time", fontsize=25)
plt.ylabel(r"Energy", fontsize=25)
plt.legend(fontsize=25, loc = 'upper right')
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)    
plt.tight_layout()
plt.show()


    
    
    
    
fig, ax = plt.subplots(figsize=(10,8))    
data_to_plot = energy_save_dict  
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

ax.set_xticks(np.arange(1, len(label_list_all) + 1))
ax.set_xticklabels(label_list_all, fontsize=30)
ax.set_ylabel('Energy', fontsize = 25)
plt.tick_params(axis='y', direction='in', labelsize=25, width=3.5, length=10, right = True, top = True)
ax.tick_params(axis='x', pad=7)
for spine in plt.gca().spines.values():
    spine.set_linewidth(3) 
ax.grid(alpha = 0.5)
plt.tight_layout()
plt.show()



