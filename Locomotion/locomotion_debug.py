#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:54:28 2022

@author: shiva
"""
#%% Consts

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
import pandas as pd
from pandas import read_excel
from matplotlib.font_manager import FontProperties
# from scipy.ndimage.interpolation import shift
from scipy.ndimage import shift

import os
import glob
import timeit
from scipy import stats
import statsmodels.stats.api as sms
from matplotlib.collections import LineCollection
from tempfile import TemporaryFile
import matplotlib.gridspec as gridspec
import seaborn as sns
from statannot import add_stat_annotation
from scipy.signal import find_peaks
from pathlib import Path
from statannot import add_stat_annotation

    
from Locomotion import *

video_height = 800
video_width = 1920
fps = 250; # frame per second 
left_edge_x = 115 # one end of treadmil in the image reported in pixels
right_edge_x = 1565 # pix
treadmill_len = 37.6 # cm
elec_shock_len = 7.34 # electro shock section of the treadmil cm
scale_pix_to_cm = treadmill_len/(right_edge_x-left_edge_x) # one pix in cm
treadmill_velocity = 15 # default treadmill velocity in cm/s
interval_in_sec = 0.5 # duration of stimulation in sec
pre_stim_interval = 0.25 # pre stimulus interval for looking at x ,v and acceleration
pre_stim_interval_measure = 0.5 # pre stimulus interval for the laser-OFF epoch in seconds
post_stim_interval_measure = 0.5 # post stimulus interval for the laser-OFF epoch in seconds

n_timebin = 30 # number of frames for calculating velocity 
window_pos = 10 # moving average window for position smoothing
window_veloc = 5 # moving average window for velocity smoothing
max_speed = 28 # (cm/s) the max speed of the mouse derived from recovety of FoxP2 form laser trial


pre_direct = '/home/shiva/Desktop/Shiva_Behavior/' # ubuntu directory
pre_direct = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted' # ubuntu directory


min_t_between_stim = 1 * fps, #s minimum time between stimulations (used to distinguish beta stimulations)

accep_interval_range = (int((121.5/125) * interval_in_sec * fps), 
                        int((128/125) * interval_in_sec * fps)) # criteria for discarding the detected laser epochs

t_window_dict = {'fps' : fps, 
                 'n_timebin' : n_timebin, 
                 'window_pos' : window_pos, 
                 'window_veloc' : window_veloc}

exp_dict = {'cor_list' : np.array(['x', 'y']),
            'body_part_list' : np.array(['Tail', 'Nose', 'FL', 'HL']),
            'plot_param_list' : ['position' , 'velocity', 'acceleration']
           }

spont_trial_dict = {'max_distance' : int( 4.5 * fps), # max #n_timebines between sampled epochs for spontaneous
                     'min_distance' : int( 3 * fps), # min #n_timebines between sampled epochs for spontaneous
                     'n_trials_spont' : 25 # the average number of trials extracted from one spontaneous session
                    }

intervals_dict = {'pre_interval' : int(pre_stim_interval_measure*fps), # interval before laser onset
                  'interval' : int(interval_in_sec * fps), # number of timebins of stimulation
                  'post_interval' : int(post_stim_interval_measure*fps*2), # interval after laser onset
                   'pre_stim_inter' : int(pre_stim_interval*fps) }


pre_x_v_dict = {'back_front_boundary' : (treadmill_len - elec_shock_len) / 2 
                                        + elec_shock_len, # set the limit below which is considered back of the treadmill
                'v_threshold' : 1, # for normalized velocity 1 means moving with the treadmill velocity
                'pre_stim_inter' : int(pre_stim_interval*fps) # number of timebins in the pre-stimulus period 
                 }


misdetection_dict = { 'acc_deviance' : 2, # cm = acceptable deviance between right and left detections
                      'internal_ctr_dev' : 0.5,
                      'percent_thresh_align' : 0.8,
                      't_s' : 30, # number of time steps before and after to look at
                      'n_iter_jitter' : 1, # how many times go over data to find jitters and clear them out
                      'jitter_threshold' : max_speed / fps,
                    }

beta_cm, square_cm = create_colormaps_for_beta_square_pulses(beta = ['Oranges_r', 'Blues'],
                                                            square = ['Reds_r', 'Greens'])


pulse_cmap_dict = {'single_stim': {'betapulse' : beta_cm, 
                                   'squarepulse': square_cm},
                   
                   'double_stim': {'betapulse' : 'cividis', 
                                   'squarepulse': 'winter',
                                   'betapulse-squarepulse': 'Wistia'}}
c_spont = 'k'         

mouse_dict = {'D2' : {'ChR2': [156, 165, 166, 195, 196, 198, 199],
                      'Control': [172, 178]}, 
              'Vglut2': {'ChR2': [116, 117, 118, 119, 164, 165, 166],
                         'Control': [359, 360]}, 
              'FoxP2': {'ChR2': [8, 9, 10, 11, 24, 27],
                        'Control': [23, 26]},
              'Vglut2D2': {'ChR2': [55, 57, 58, 59, 60, 70, 71], 
                           'Control': [63, 73]}}

mouse_types = list(mouse_dict.keys())


mouse_color_opto_dict = {'ChR2': ['r', 'gold', 'darkorange', 'k', 'mediumspringgreen', 'dodgerblue', 'darkviolet'],
                          'Control': ['grey', 'lightcoral']}

mouse_color_dict = create_mouse_color_dict(mouse_dict, mouse_color_opto_dict)

protocol_dict = {
    'D2' : 
        {'ChR2': 
                {'STR': {
                        'squarepulse': 
                            ['squarepulse_0-5_mW',
                             'squarepulse_1_mW'],
                        'betapulse': 
                            ['betapulse_0-5_mW',
                              'betapulse_1_mW']}},
        'Control': 
                {'STR': {
                        'squarepulse': 
                            ['squarepulse_1_mW'],
                        'betapulse': 
                            ['betapulse_1_mW',
                              'betapulse_5_mW']}}},
    'Vglut2': 
        {'ChR2':
            {'STN': {
                 'squarepulse': 
                     ['squarepulse_0-5_mW',
                      'squarepulse_1_mW'],
                 'betapulse': 
                     ['betapulse_0-5_mW',
                      'betapulse_1_mW']},
            'GPe': {
                 'squarepulse': 
                     ['squarepulse_15_mW'],
                 'betapulse': []},
            'STN-GPe': {
                'squarepulse': 
                    ['squarepulse_1-15_mW'],
                 'betapulse': []}},
         'Control':
            {'STN': {
                 'squarepulse': 
                     ['squarepulse_1_mW'],
                 'betapulse': 
                     ['betapulse_1_mW',
                      'betapulse_5_mW']}}},
    
    'FoxP2': 
        {'ChR2':
            {'GPe': {
                'squarepulse': 
                    ['squarepulse_0-25_mW',
                     'squarepulse_0-5_mW',
                     'squarepulse_1_mW'],
                'betapulse': 
                    ['betapulse_0-5_mW',
                     'betapulse_1_mW',
                     'betapulse_2_mW',
                     'betapulse_5_mW']}},
        'Control':
            {'GPe': {
                'squarepulse': 
                    ['squarepulse_1_mW'],
                'betapulse': 
                    ['betapulse_1_mW',
                     'betapulse_5_mW']}}},
        
    'Vglut2D2': 
        {'ChR2':
            {'STN': {
                    'squarepulse': 
                        ['squarepulse_0-5_mW',
                         'squarepulse_0-75_mW',
                         'squarepulse_1_mW']},
                'STR': { 
                    'squarepulse': 
                        ['squarepulse_0-35_mW',
                         'squarepulse_0-5_mW',
                         'squarepulse_1_mW'],
                    'betapulse': 
                        ['betapulse_0-5_mW',
                         'betapulse_0-75_mW',
                         'betapulse_1_mW',
                         'betapulse_2-5_mW',
                         'betapulse_5_mW']},

                 'STR-STN': {
                       'squarepulse': 
                           ['squarepulse_0-5_mW',
                            'squarepulse_0-35-0-5_mW'],
                       'betapulse': 
                           ['betapulse_0-75_mW',
                            'betapulse_5_mW'],
                       'betapulse-squarepulse': 
                           ['betapulse-squarepulse_5-0-75_mW']}},
        'Control':
            {'STN': {
                    'squarepulse': 
                        ['squarepulse_0-5_mW']},
                'STR': { 
                    'squarepulse': 
                        ['squarepulse_0-5_mW'],
                    'betapulse': 
                        ['betapulse_0-5_mW',
                         'betapulse_0-75_mW',
                         'betapulse_1_mW',
                         'betapulse_2-5_mW']},

                 'STR-STN': {
                       'squarepulse': 
                           ['squarepulse_0-5_mW'],
                       'betapulse': 
                           ['betapulse_0-75_mW']}}}}
            
#%% run one inten


# study_param_dict = get_input_cor_body_part(**exp_dict) # decide to average over what and which coordinates
study_param_dict = {
                    'cor': 'x', 
                    'body_part': ['Tail', 'Nose'], 
                    'plot_param': 'velocity'
                    }
# opto_par_list=['ChR2','Control']
opto_par_list=['ChR2']

mouse_t_list = list( mouse_dict.keys()) # mouse lines are the keys of the mouse dict
mouse_t_list = ['Vglut2']
ylim = [-20,15]


stim_type = 'squarepulse_1_mW'
# stim_type = 'squarepulse_0-5_mW'
# stim_type = 'squarepulse_0-35_mW'
# stim_type = 'squarepulse_0-25_mW'

# stim_type = 'betapulse_0-5_mW'
# stim_type = 'betapulse_0-75_mW'
# stim_type = 'betapulse_1_mW'
# stim_type = 'betapulse_2_mW'
# stim_type = 'betapulse_2-5_mW'
# stim_type = 'betapulse_5_mW'

stim_loc = 'STR-STN'
stim_loc = 'STN'
# stim_loc = 'STR'
# stim_loc = 'GPe'

for mouse_type in mouse_t_list:  
    
    for opto_par in opto_par_list:
        
        run_one_intensity_save_data(pre_direct,
                                    scale_pix_to_cm,
                                    mouse_type, 
                                    mouse_dict[mouse_type],
                                    stim_loc,
                                    stim_type,
                                    opto_par,
                                    treadmil_velocity,
                                    ylim,
                                    spont_trial_dict,
                                    misdetection_dict,
                                    intervals_dict,
                                    t_window_dict,
                                    accep_interval_range,
                                    study_param_dict,
                                    **spont_trial_dict)
        
# plt.close('all')
#%% plot single session
file_name = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Control/Mouse_63/STR/betapulse_2-5mW/DLC/Vglut2D2Cre#63_betapulse_STR_2-5mW_15cm-s_r06_Stacked_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv'
df = read_DLC(file_name, scale_pix_to_cm)
fig, ax = plt.subplots()
ax.plot(df['rTail', 'x'])
ax.plot(df['lTail', 'x'])

#%% correction

path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2/ChR2/Mouse_119'
move_unwanter_files_out_of_folder(path)
directory = Directory(path)
directory.remove_spaces_dots()
get_sorted_laser_DLC_files(path)
rename_laser_files_according_to_DLC(path)

#%% correction
path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/D2' 
convert_all_xlsx_to_csv(path)   

path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/FoxP2'            
unify_protocol_names(path)



#%% superimpose

plt.close('all')
study_param_dict = {
                    'cor': 'x', 
                    'body_part': ['Tail', 'Nose'], 
                    'plot_param': 'velocity'
                    }
ylim = [-.5, 2]
x_label_list = [0, 0.5, 1, 1.5]
y_label_list = [-0.5, 0, 0.5, 1, 1.5, 2]
plot_spont = True
plot_spont = False


 

mouse_type = 'FoxP2'
# mouse_type = 'D2'

stim_loc_dict = {opto: {pulse: list(protocol_dict[mouse_type][opto].keys())
                                for pulse in ['squarepulse', 'betapulse']}
                     for opto in ['ChR2', 'Control']}


mouse_type = 'Vglut2'

stim_loc_dict = {'ChR2':{'squarepulse': list(protocol_dict[mouse_type]['ChR2'].keys()),
                          'betapulse':  ['STN']},
                 
                    'Control': {'squarepulse': ['STN'],
                                'betapulse':  ['STN']}}

# mouse_type = 'Vglut2D2'
# plot_spont = False
# stim_loc_dict = {'ChR2':{'squarepulse': list(protocol_dict[mouse_type]['ChR2'].keys()),
#                           'betapulse':   ['STR'],
#                           'betapulse-squarepulse':  ['STR-STN']
#                                             },
#                   'Control': { 'squarepulse':  list(protocol_dict[mouse_type]['Control'].keys()),
#                               'betapulse': ['STR'],
#           }}


opto_par_list  = ['ChR2']
# opto_par_list  = ['Control']
suptitle_y  = {'Control': 1.1,
               'ChR2': 0.94}
figs = superimpose_intensities(opto_par_list, pulse_cmap_dict, protocol_dict,  stim_loc_dict, 
                        pre_direct, scale_pix_to_cm, mouse_type, mouse_dict[mouse_type], treadmill_velocity,
                        ylim, spont_trial_dict, misdetection_dict, intervals_dict, t_window_dict, accep_interval_range, 
                        study_param_dict, **spont_trial_dict, suptitle_y = suptitle_y , plot_spont = plot_spont,
                        x_label_list = x_label_list, y_label_list = y_label_list)#, ylabel_x = 0.2, xlabel_y = 0.2)


# %% average all mice 

plt.close('all')
opto_par = 'ChR2'
label = 'intensity'

mouse_type = 'Vglut2D2'
stim_loc = 'STN'
stim_loc = 'STR'

# stim_loc = 'STR-STN'
# label = 'protocol'
not_contains = []
ylim = [0, 2]
x_label_list = [0, 0.5, 1, 1.5]
y_label_list = [-0.5, 0, 0.5, 1, 1.5, 2]

# mouse_type = 'FoxP2'
# stim_loc = 'GPe'
# min_y_pre_post = -25 ; max_y_pre_post =  20
# not_contains = []

# mouse_type = 'Vglut2'
# not_contains = ['D2']
# stim_loc = 'STN'
# min_y_pre_post = -15 ; max_y_pre_post =  10


# mouse_type = 'D2'
# not_contains = ['Vglut']
# stim_loc = 'STR'
# min_y_pre_post = -15 ; max_y_pre_post =  20



stim_type = 'squarepulse'
stim_type = 'betapulse'
        
path = os.path.join(pre_direct, 'data_npz', stim_loc)
files_list = find_files_containing_substring(path, ".npz")
summary_files_list = filter_based_on_substring(files_list, [stim_type, mouse_type, opto_par], not_contains = not_contains)


colors = choose_n_from_colormap(plt.get_cmap(pulse_cmap_dict['single_stim'][stim_type]), 
                                minval=0.0, maxval=1.0, n_colors = len(summary_files_list))
summary_files_list.sort()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
axvspan = True

for count, filepath in enumerate(summary_files_list):

    data = np.load(filepath)
    info = extract_info_from_npz_filename(os.path.basename(filepath))
    study_param_dict = {'cor' : data['cor'][0],
                        'body_part' : data['body_part'], 
                        'plot_param' : data['plot_param'][0]}
    print(info['mouse_type'])
    
    if count > 0:
        axvspan = False
    
    ax, figname = plot_pre_on_post(pre_direct, 
                                   info['mouse_type'], info['opto_par'], 
                                   info['stim_loc'], info['stim_type'], 
                                   data['epochs_all_mice'], data['epochs_spont_all_mice'], 
                                   treadmil_velocity,
                                   ylim,
                                   **t_window_dict,**study_param_dict,**intervals_dict,
                                   average = 'all_mice', save_fig = False,
                                   label = label, axvspan = axvspan,
                                   c_laser = colors[count], c_spont = c_spont,
                                   save_as_format = '.pdf', ax = ax,
                                   legend_fontsize = 14,
                                   legend_loc = 'upper left',
                                   bbox_to_anch_leg=(0., 1.),
                                   x_label_list = x_label_list, 
                                   y_label_list = y_label_list)

Directory.create_dir_if_not_exist(os.path.join(pre_direct, 'Subplots', stim_loc))
filepath = os.path.join(pre_direct, 'Subplots', 
                        stim_loc, stim_type + '_' + figname)
save_pdf_png(fig, filepath.split('.')[0], size = (6, 6))

# %% delta_v_vs_laser_intensity


measure = 'min'
# measure = 'mean'
ylim = (-1.75, 0.08)
experiment_dict = {'mouse_type': ['Vglut2D2', 'Vglut2D2', 'FoxP2', 'FoxP2', 'D2', 'D2', 'Vglut2', 'Vglut2'],
                   'opto_par': ['ChR2'] * 8,
                   'stim_loc': ['STR', 'STR', 'GPe', 'GPe', 'STR', 'STR', 'STN', 'STN'],
                   'pulse': ['square', 'beta', 'square', 'beta', 'square', 'beta', 'square', 'beta'],
                   'marker': ['o', 'o', '^', '^', 's', 's', '*', '*'],
                   'color': ['turquoise', 'gold', 'turquoise', 'gold','turquoise', 'gold','turquoise', 'gold']}


n_exp = len(experiment_dict['pulse'])
fig, ax = plt.subplots(figsize = (8,6))

ylabel = {'mean': r'$norm. \; (\overline{V}^{ON} - \overline{V}^{OFF}) $',
          'min': r'$norm. \; ({V}_{min}^{ON} - \overline{V}^{OFF})$'}
for i in range(n_exp):

    intensity, mean_dv, std_dv, n_trials = delta_v_vs_laser_intensity(pre_direct, intervals_dict,
                                                                      opto_par = experiment_dict['opto_par'][i],
                                                                      stim_loc = experiment_dict['stim_loc'][i],
                                                                      mouse_type = experiment_dict['mouse_type'][i],
                                                                      pulse = experiment_dict['pulse'][i],
                                                                      measure = measure)
    
    ax.errorbar(intensity, mean_dv, std_dv,
                marker = experiment_dict['marker'][i], 
                color =  experiment_dict['color'][i], 
                ms = 10, mec = 'k',
                capsize = 5, capthick = 2, 
                label = ' '.join([experiment_dict['mouse_type'][i], 
                                  experiment_dict['pulse'][i],
                                  experiment_dict['stim_loc'][i]]) \
                                  # experiment_dict['opto_par'][i]]) \
                        + ' (n = ' + str(n_trials) + ')'
                )
    remove_frame(ax)
    set_ticks(ax)
    ax.set_xlabel('Laser intensity (mW)').set_fontproperties(font_label)
    ax.set_ylabel(ylabel[measure]).set_fontproperties(font_label)
    ax.legend(fontsize = 12, frameon = False,  loc = 'upper left', bbox_to_anchor=(.8, 0.7))

ax.set_ylim(ylim)
ax.set_title(measure).set_fontproperties(font)
Directory.create_dir_if_not_exist(os.path.join(pre_direct, 'Subplots', 'Square_vs_beta'))
ax.tick_params(axis = 'both', which='major', labelsize = 16)

figname = 'Square_vs_Beta_' + measure + '_velocity_' + '_'.join(np.unique(experiment_dict['mouse_type']))
save_pdf_png(fig, 
             os.path.join(pre_direct, 'Subplots', 'Square_vs_beta', figname ), 
             size = fig.get_size_inches()*fig.dpi)



#%%  Violin Plot


opto_par = 'ChR2'
mouse_type = 'Vglut2D2'
stim_loc = 'STR'

# opto_par = 'ChR2'
# mouse_type = 'FoxP2'
# stim_loc = 'GPe'

# opto_par = 'ChR2'
# mouse_type = 'Vglut2'
# stim_loc = 'STN'

# opto_par = 'ChR2'
# mouse_type = 'D2'
# stim_loc = 'STR'

pulse = 'square'

summary_files_list = get_all_filepaths_different_intensities(pre_direct, pulse = pulse, 
                                                             opto_par = opto_par, 
                                                             stim_loc = stim_loc, 
                                                             mouse_type = mouse_type)
subplot_parameter = 'intensity_mW'

ax = violin_plot_laser_ON_OFF(pre_direct, intervals_dict,
                          opto_par, mouse_color_dict, mouse_type, 
                          subplot_parameter = subplot_parameter,
                          velocity_measure = 'norm_velocity (mean)', 
                          summary_files_list = summary_files_list)
#%%
plt.close('all')
stim_type = 'squarepulse_0-5_mW'
# stim_type = 'betapulse_0-75_mW'

opto_par = 'ChR2'
# opto_par = 'Control'

mouse_type = 'D2'
mouse_type = 'Vglut2D2'
# mouse_type = 'FoxP2'
# mouse_type = 'Vglut2'

stim_loc = 'STN'

 
file_path_list = find_filepaths_for_stim_type_and_loc_list_one_mouse_type([stim_loc], 
                                                                          pre_direct, 
                                                                          [stim_type], 
                                                                          opto_par,
                                                                          mouse_type)
result = create_df_from_data_summary(file_path_list, intervals_dict, 
                                          opto_par = opto_par, report_stats =False)
    
result = categorize_pre_x_and_v(result ,**pre_x_v_dict)
violin_plot_with_distiction(result, pre_direct, mouse_type, opto_par, stim_loc, 
                            stim_type, t_window_dict['fps'],**pre_x_v_dict,
                            x_distinction= True, save_as_format='.pdf', column = 'pre_accel_sign')
plot_velocity_phase_space(result, pre_direct, mouse_type, opto_par, stim_loc, stim_type, t_window_dict['fps'],**pre_x_v_dict,
                    save_as_format='.pdf')



for stim_loc, pulse_dict in protocol_dict[mouse_type][opto_par].items():
    for pulse_type, protocol_list in pulse_dict.items():
        for stim_type in protocol_list:
            
            plt.close('all')
            summary_files_list = find_filepaths_for_stim_type_and_loc_list_one_mouse_type([stim_loc], 
                                                                                pre_direct, 
                                                                                [stim_type], 
                                                                                opto_par,
                                                                                mouse_type)
             
            result = create_df_from_data_summary(summary_files_list, intervals_dict, 
                                                      opto_par = opto_par, report_stats =False)
                
            result = categorize_pre_x_and_v(result ,**pre_x_v_dict)
            
            violin_plot_with_distiction(result, pre_direct, mouse_type, opto_par, stim_loc, 
                                        stim_type, t_window_dict['fps'],**pre_x_v_dict,
                                        x_distinction= True, save_as_format='.pdf',column = 'pre_accel_sign')
            
            plot_velocity_phase_space(result, pre_direct, mouse_type, opto_par, stim_loc, 
                                      stim_type, t_window_dict['fps'],**pre_x_v_dict,
                                      save_as_format='.pdf')



# import seaborn as sns
# from statannotations.Annotator import Annotator


# tips = sns.load_dataset("tips")

# args = dict(x="sex", y="total_bill", data=tips, hue="smoker", hue_order=["Yes","No"], order=['Male', 'Female'])

# g = sns.catplot(edgecolor="black", errcolor="black", errwidth=1.5, capsize = 0.1, height=4, aspect=.7,alpha=0.5, kind="bar", ci = "sd", row="time", **args)
# g.map(sns.stripplot, args["x"], args["y"], args["hue"], hue_order=args["hue_order"], order=args["order"], palette=sns.color_palette(), dodge=True, alpha=0.6, ec='k', linewidth=1)

# pairs = [
#     (("Male", "Yes"), ("Female", "Yes"))]

# for ax_n in g.axes:
#     for ax in ax_n:
#         annot = Annotator(ax, pairs, **args)
#         annot.configure(test='Mann-Whitney', text_format='simple', loc='inside', verbose=2)
#         annot.apply_test().annotate()
