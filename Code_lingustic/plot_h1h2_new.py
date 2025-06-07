from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d


def plot_f0(astro_table, time_normalise, use_mean, num_person):
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

        plt.figure(figsize=(10, 8))
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
                
                segment_h1h2 = np.array(table['H1H2c'][mask])
                spline_h1h2 = UnivariateSpline(t_value, segment_h1h2, s=0.5)
                interpolator_h1h2 = interp1d(t_value, segment_h1h2, kind='linear', fill_value='extrapolate')
                spline_y_h1h2 = interpolator_h1h2(t_smooth)

                
                if label == 'qu':
                    if spline_y_f0[0] > 125 and spline_y_f0[0] < 300:
                        plt.plot(t_value, segment_h1h2, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y_h1h2, color='red', lw=1, label="Spline Fit", alpha=1) 
                        fitted_spline.append((t_smooth, spline_y_h1h2))        
                        if np.isnan(segment_h1h2[0]):
                            segment_info.append([seg_start, "#NUM!"])
                        else:
                            segment_info.append([seg_start, round(segment_h1h2[0])])
                    if spline_y_f0[-1] < 136 and spline_y_f0[-2] < 136:
                        plt.plot(t_smooth, spline_y_h1h2, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                  
                if label == 'ru':
                    if spline_y_f0[0] > 125 and spline_y_f0[0] < 300:
                        plt.plot(t_value, segment_h1h2, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y_h1h2, color='red', lw=1, label="Spline Fit", alpha=1) 
                        fitted_spline.append((t_smooth, spline_y_h1h2))        
                        if np.isnan(segment_h1h2[0]):
                            segment_info.append([seg_start, "#NUM!"])
                        else:
                            segment_info.append([seg_start, round(segment_h1h2[0])])
                    if spline_y_f0[-1] < 136 and spline_y_f0[-2] < 136:
                        plt.plot(t_smooth, spline_y_h1h2, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                        
                if label == 'shang':
                    if spline_y_f0[0] > 125 and spline_y_f0[0] < 320:
                        plt.plot(t_value, segment_h1h2, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y_h1h2, color='red', lw=1, label="Spline Fit", alpha=1) 
                        fitted_spline.append((t_smooth, spline_y_h1h2))        
                        if np.isnan(segment_h1h2[0]):
                            segment_info.append([seg_start, "#NUM!"])
                        else:
                            segment_info.append([seg_start, round(segment_h1h2[0])])
                    if spline_y_f0[-1] < 136 and spline_y_f0[-2] < 136:
                        plt.plot(t_smooth, spline_y_h1h2, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                        
                if label == 'yang':
                    if spline_y_f0[0] > 170:
                        fitted_spline.append((t_smooth, spline_y_h1h2))        
                        if np.isnan(segment_h1h2[0]):
                            segment_info.append([seg_start, "#NUM!"])
                        else:
                            segment_info.append([seg_start, round(segment_h1h2[0])])
                        plt.plot(t_value, segment_h1h2, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y_h1h2, color='red', lw=1, label="Spline Fit", alpha=1)    
                    if spline_y_f0[-1] < 136 and spline_y_f0[-2] < 136:
                        plt.plot(t_smooth, spline_y_h1h2, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                  
                if label == 'yin':
                    if spline_y_f0[0] > 150 and spline_y_f0[0] < 320:
                        fitted_spline.append((t_smooth, spline_y_h1h2))        
                        if np.isnan(segment_h1h2[0]):
                            segment_info.append([seg_start, "#NUM!"])
                        else:
                            segment_info.append([seg_start, round(segment_h1h2[0])])
                        plt.plot(t_value, segment_h1h2, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y_h1h2, color='red', lw=1, label="Spline Fit", alpha=1)    
                    if spline_y_f0[-1] < 136 and spline_y_f0[-2] < 136:
                        plt.plot(t_smooth, spline_y_h1h2, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                        
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
                
        plt.plot(t_smooth, median_y, color='blue', lw=7)
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
            

        plt.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=8)
        plt.xlabel(r"$\mathrm{Normalised \, Time}$", fontsize=25)
        plt.ylabel(r"$f_0 \,\, \mathrm{[Hz]}$", fontsize=25)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(3)
        plt.title(f"{label}, n_seg = {counter}", fontsize=25)
        plt.tight_layout()
        plt.show()
        fitted_spline_dict[label] = np.array(segment_info)
        
    std_y_value = np.array(std_y_value)
    tone_y = np.array(tone_y)
    tone_x = np.array(tone_x)
    
    
    
    
    
    
    
    # ------------------
    # Plot f0
    # ------------------
    plt.figure(figsize=(10, 8))    
    colors = np.array([[5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
       [1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00],
       [5.03921569e-01, 9.99981027e-01, 7.04925547e-01, 1.00000000e+00],
       [1.00000000e+00, 7.00543038e-01, 3.78411050e-01, 1.00000000e+00],
       [1.96078431e-03, 7.09281308e-01, 9.23289106e-01, 1.00000000e+00]
       ])
    
    
    import pickle
    import os
    x_y_dict = {}
    for x, y, label_dd, color, std_individual in zip(tone_x, tone_y, label_list, colors, std_y_value):
        plt.plot(x, y, color=color, label=label_dd, lw=4)
        # Initialize dictionary for this label if it doesn't exist
        if label_dd not in x_y_dict:
            x_y_dict[label_dd] = {}
        x_y_dict[label_dd]['x'] = x
        x_y_dict[label_dd]['y'] = y
    
    pkl_location = f'/Users/emilydu/Code/Code_lingustic/Data/f0_h1h2_xy_pkl_save/{num_person}/h1h2_xy.pkl'
    os.makedirs(os.path.dirname(pkl_location), exist_ok=True)
    with open(pkl_location, 'wb') as f:
        pickle.dump(x_y_dict, f) 

    plt.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=8)
    plt.xlabel(r"Normalised Time", fontsize=25)
    plt.ylabel(r"H1H2c", fontsize=25)
    plt.legend(fontsize=25, loc = 'upper right')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)    
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/h1h2/h1h2_t.png', dpi = 300)
    plt.show()
    
    
    np.savez(saving_location,
             tone_x = tone_x,
             tone_y = tone_y, 
             label_list = label_list,
             ) 



    return fitted_spline_dict, std_y_value, label_list, tone_y


time_normalise = True
use_mean = True
num_person = 'both'
    
astro_table = Table.read('/Users/emilydu/Code/Code_lingustic/Data/catalog/csv_format/pure_mono_tone.csv')

if num_person == 'jy':
    astro_table = astro_table[astro_table['\ufeffFilename'] == 'jy_pure_mono.mat']
    print('jy')
elif num_person == 'jz':
    astro_table = astro_table[astro_table['\ufeffFilename'] == 'jz_pure_mono.mat']
    print('jz')
else:
    print('Both')
    
    
astro_table['H1H2c'] = [float(value) if value != '#NUM!' else np.nan for value in astro_table['H1H2c']]

fitted_spline_dict, std_y, label_list, tone_y = plot_f0(astro_table, time_normalise, use_mean, num_person)




    
    
    
    



