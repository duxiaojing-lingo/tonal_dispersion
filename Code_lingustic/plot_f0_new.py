from astropy.table import Table
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline


def plot_f0(astro_table, smoothing_factor, time_normalise, use_mean, num_person):
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
                spline_y = spline(t_smooth)
                
                if label == 'qu':
                    if spline_y[0] > 125 and spline_y[0] < 300:
                        plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1) 
                        fitted_spline.append((t_smooth, spline_y))        
                        segment_info.append([seg_start, round(segment_f0[0])])
                    if spline_y[-1] < 136 and spline_y[-2] < 136:
                        plt.plot(t_smooth, spline_y, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                  
                if label == 'ru':
                    if spline_y[0] > 125 and spline_y[0] < 300:
                        plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1) 
                        fitted_spline.append((t_smooth, spline_y))        
                        segment_info.append([seg_start, round(segment_f0[0])])
                    if spline_y[-1] < 136 and spline_y[-2] < 136:
                        plt.plot(t_smooth, spline_y, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                        
                if label == 'shang':
                    if spline_y[0] > 125 and spline_y[0] < 320:
                        plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1) 
                        fitted_spline.append((t_smooth, spline_y))        
                        segment_info.append([seg_start, round(segment_f0[0])])
                    if spline_y[-1] < 136 and spline_y[-2] < 136:
                        plt.plot(t_smooth, spline_y, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                        
                if label == 'yang':
                    if spline_y[0] > 170:
                        fitted_spline.append((t_smooth, spline_y))        
                        segment_info.append([seg_start, round(segment_f0[0])])
                        plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    
                    if spline_y[-1] < 136 and spline_y[-2] < 136:
                        plt.plot(t_smooth, spline_y, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                  
                if label == 'yin':
                    if spline_y[0] > 150 and spline_y[0] < 320:
                        fitted_spline.append((t_smooth, spline_y))        
                        segment_info.append([seg_start, round(segment_f0[0])])
                        plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                        plt.plot(t_smooth, spline_y, color='red', lw=1, label="Spline Fit", alpha=1)    
                    if spline_y[-1] < 136 and spline_y[-2] < 136:
                        plt.plot(t_smooth, spline_y, color='blue', lw=5, label="Spline Fit", alpha=0.3)
                        
                counter += 1
        else:
            print('Time Not Normalised')
            for seg_start, seg_end in unique_segments:            
                mask = (table['seg_Start'] == seg_start) & (table['seg_End'] == seg_end)
                segment_f0 = np.array(table['strF0'][mask])
                t_value = table['t_ms'][mask] - np.min(table['t_ms'][mask])    
                plt.plot(t_value, segment_f0, color='black', alpha=1.0, lw=1, label="Original")        
                segment_info.append([seg_start, round(segment_f0[0])])
                                
                t_value = np.array(t_value, dtype=int)                
                max_length = 700                  
                full_t_value = np.arange(0, max_length)                
                full_segment_f0 = np.full(max_length, np.nan)                
                full_segment_f0[t_value] = segment_f0 
                t_smooth = full_t_value
                fitted_spline.append((full_t_value, full_segment_f0))
                counter += 1
        fitted_spline = np.array(fitted_spline)  
        
        
        y_values = fitted_spline[:, 1, :]
        
        if use_mean:
            median_y = np.nanmean(y_values, axis=0)
            saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_{smoothing_factor}_mean.npz'
        else:
            median_y = np.nanmedian(y_values, axis=0)
            saving_location = f'/Users/emilydu/Code/Code_lingustic/Data/data/fitted_version/tone_fitted_s_{smoothing_factor}_median.npz'


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
        plt.xlabel(r"$\mathrm{Normalised\, Time}$", fontsize=25)
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
    for x, y, label_dd, color,std_individual in zip(tone_x, tone_y, label_list, colors, std_y_value):
        plt.plot(x, y, color=color, label=label_dd, lw=4)
        if label_dd not in x_y_dict:
            x_y_dict[label_dd] = {}
        x_y_dict[label_dd]['x'] = x
        x_y_dict[label_dd]['y'] = y
 
    pkl_location = f'/Users/emilydu/Code/Code_lingustic/Data/f0_h1h2_xy_pkl_save/{num_person}/f0_xy.pkl'
    os.makedirs(os.path.dirname(pkl_location), exist_ok=True)
    with open(pkl_location, 'wb') as f:
        pickle.dump(x_y_dict, f)

        
        # plt.fill_between(x, y -std_individual, y + std_individual, color = color, alpha = 0.20)
    # tone_hz = np.arange(150,220,14)
    # for hz_values in tone_hz:
    #     plt.axhline(y=hz_values, color='grey', lw=3, ls='--', alpha=0.5)
    plt.tick_params(axis='both', direction='in', labelsize=24, width=3.5, length=8)
    plt.xlabel(r"Normalised Time", fontsize=25)
    plt.ylabel(r"$f_0 \,\, \mathrm{[Hz]}$", fontsize=25)
    plt.ylim(100,300)
    plt.legend(fontsize=25)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)    
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/f0/f0_hz.png', dpi = 300)
    plt.show()
    

    # ------------------
    # Plot Five Tone
    # ------------------
    
    plt.figure(figsize = (10,8))
    plt.hist(std_y_value.flatten(), color = 'skyblue', edgecolor = 'black', lw = 3)
    plt.axvline(np.median(std_y_value.flatten()), lw = 3, c = 'red')
    plt.axvline(np.percentile(std_y_value.flatten(), 16), lw = 3, c = 'red')
    plt.axvline(np.percentile(std_y_value.flatten(), 84), lw = 3, c = 'red')
    plt.xlabel('std', fontsize = 25)
    plt.ylabel('count', fontsize = 25)
    plt.show()
    
    

    
    
    min_index = np.unravel_index(np.nanargmin(tone_y), tone_y.shape)
    max_index = np.unravel_index(np.nanargmax(tone_y), tone_y.shape)
    
    min_tone = tone_y[min_index]
    max_tone = tone_y[max_index]
    min_std = std_y_value[min_index]
    max_std = std_y_value[max_index]
    
    if use_mean:
        if num_person == 'jy':
            print('jy mean')
            min_tone = 188.2
            max_tone = 272.32
            min_std = 10
            max_std = 10
        if num_person == 'jz':
            print('jz mean')
            min_tone = 114.5
            max_tone = 239
            min_std = 10
            max_std = 10
        if num_person == 'both':
            print('jz mean')
            min_tone = 162
            max_tone = 247
            min_std = 10
            max_std = 10



    else:
        if num_person == 'both':
            print('jz mean')
            min_tone = 162
            max_tone = 247
            min_std = 10
            max_std = 10


    print(f'min_tone = {min_tone}')
    print(f'max_tone = {max_tone}')
    print(f'min_std = {min_std}')
    print(f'max_std = {max_std}')

    
    five_tone_y = (np.log10(tone_y) - np.log10(min_tone - min_std)) * 5 / (np.log10(max_tone + max_std) - np.log10(min_tone - min_std))



    plt.figure(figsize=(14, 10))
    for x, y, label_dd, color in zip(tone_x, five_tone_y, label_list, colors):
        spline = UnivariateSpline(x, y, s = smoothing_factor)
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        spline = UnivariateSpline(x_valid, y_valid, s=smoothing_factor)
        
        x_smooth = np.linspace(np.min(x), np.max(x), 200)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, color= color, label = label_dd, lw = 8)
        plt.plot(x, y, color='black', lw=2, alpha = 0.2)

    tone_hz = np.array([-3, -2, -1,0,1,2,3,4, 5])
    for hz_values in tone_hz:
        plt.axhline(y=hz_values, color='grey', lw=3, ls='--', alpha=0.5)
    
    # plt.axvline(x = 0.1, color='grey', lw=3, ls='--', alpha=0.5)
    # plt.axvline(x = 0.9, color='grey', lw=3, ls='--', alpha=0.5)


    plt.tick_params(axis='both', direction='in', labelsize=28, width=3.5, length=15, right = True, top = True)
    plt.xlabel(r"Normalised Time", fontsize=30)
    plt.ylabel("Five Tone Scale", fontsize=30)
    plt.legend(fontsize=25, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(label_list))
    plt.title(f'Smoothing Factor = {smoothing_factor}', fontsize =25)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)    
    plt.tight_layout()
    # plt.ylim(-0.5, 5.5)
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/f0/f0_5tone.png', dpi = 300)
    plt.show()
    
    
    
    ### ----------------
    ### Zscore
    ### ----------------

    
    tone_y = np.log10(tone_y)
    five_tone_y = (tone_y - np.mean(tone_y)) / np.std(tone_y)

    plt.figure(figsize=(14, 10))
    for x, y, label_dd, color in zip(tone_x, five_tone_y, label_list, colors):
        spline = UnivariateSpline(x, y, s = smoothing_factor)
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        spline = UnivariateSpline(x_valid, y_valid, s=smoothing_factor)
        
        x_smooth = np.linspace(np.min(x), np.max(x), 200)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, color= color, label = label_dd, lw = 8)
        plt.plot(x, y, color='black', lw=2, alpha = 0.2)

    tone_hz = np.array([-3, -2, -1, 0, 1, 2])
    for hz_values in tone_hz:
        plt.axhline(y=hz_values, color='grey', lw=3, ls='--', alpha=0.5)
    
    # plt.axvline(x = 0.1, color='grey', lw=3, ls='--', alpha=0.5)
    # plt.axvline(x = 0.9, color='grey', lw=3, ls='--', alpha=0.5)


    plt.tick_params(axis='both', direction='in', labelsize=28, width=3.5, length=15, right = True, top = True)
    plt.xlabel(r"Normalised Time", fontsize=30)
    plt.ylabel("Zscore", fontsize=30)
    plt.legend(fontsize=25, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(label_list))
    plt.title(f'Smoothing Factor = {smoothing_factor}', fontsize =25)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)    
    plt.tight_layout()
    plt.ylim(-3.5, 2.5)
    plt.tight_layout()
    plt.savefig(f'/Users/emilydu/Code/Code_lingustic/Data/plots/f0/f0_zscore.png', dpi = 300)
    plt.show()
    
    np.savez(saving_location,
             tone_x = tone_x,
             tone_y = tone_y, 
             std_y_value = std_y_value,
             five_tone_y = five_tone_y,
             label_list = label_list,
             ) 

    
    return fitted_spline_dict, std_y_value, label_list, tone_y



smoothing_factor = 5
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
    
    

fitted_spline_dict, std_y, label_list, tone_y = plot_f0(astro_table, smoothing_factor, time_normalise, use_mean, num_person)




    
    
    
    



