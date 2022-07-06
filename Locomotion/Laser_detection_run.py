#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:50:07 2022

@author: shiva
"""

from File_hierarchy import *
from Laser_detection import *

# # area_cal_method = 'contour'
# area_cal_method = 'pix_count'
# fps = 250 # frame per second of the video

# treadmil_length_in_cm = 37.6
# constrain_frame= True
# max_iteration = 20

# max_dev_in_cm = 1.7
# gauss_window = 10
# low_f = 1
# high_f = 50 
# filt_order = 10
# DLC_p_cutoff_ranges = 0.995
# enforce_use_laser_detection_only = True
# # enforce_use_laser_detection_only = False
# use_laser_detection_if_no_analogpulse = True
# reanalyze_existing = True

# crude_smooth_wind = 100
# stim_duration_dict = { 'beta': 118, 'square': 125, 'beta-square': 125}


# # RGB_blueLower = (50, 80, 60) ## HSV
# # RGB_blueUpper = (150, 255, 255)

# RGB_blueLower = (150, 10,60)
# RGB_blueUpper = (255, 120, 140)
# center_vicinity_h_thresh = 0.3
# start_end_h_thresh = 0.3

path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2'
laser_detection_path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2/LASER_DETECTION'

# directory = Directory(path)
# video_filepath_list = directory.get_spec_files( extensions= ['.avi', '.mp4', '.mov'])
# video_filepath_list = [v for v in video_filepath_list if 'beta' in v]
# for n, v in enumerate(video_filepath_list):

#     if 'o21' in v:
#         print(n) 
video_filepath_list = ['/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2/ChR2/Mouse_58/STR-STN/squarepulse_0-5_mW/Video/Vglut2D2Cre#58_SquarePulse_STR+STN_0-5mW_15cm-s_Stacked_d07.avi']
# video_filepath_list = SortedExpeiment.read_summary_csv(os.path.join(laser_detection_path, 'Problematic_files.csv'))

print( '{} experiment files found.'.format(len(video_filepath_list)) )

configpath = set_config_file(path)
SortedExpeiment.create_problematic_csv(os.path.join(laser_detection_path, 'Problematic_files.csv'))
SortedExpeiment.create_summary_csv(os.path.join(laser_detection_path, 'Analysis_summary.csv'))


for i, filepath in enumerate(video_filepath_list):

    plt.close( 'all' )
    print('{} from {} files.'.format(i + 1, len (video_filepath_list)))
    

    # try: 
    cfg = read_config(configpath)

    sorted_exp = SortedExpeiment(filepath, stim_duration_dict = cfg['stim_duration_dict'])

        
    if not sorted_exp.already_analyzed or cfg['reanalyze_existing']:
        print(cfg['pix_thresh_lower_bound'])
        laser_detector = LaserDetector( sorted_exp.video.path,  
                                        sorted_exp.DLC_path,
                                        thresh_method = cfg['thresholding_method'],
                                        area_cal_method = cfg['area_cal_method'],
                                        image_parts = ['upper', 'lower'],
                                        treadmil_length_in_cm = cfg['treadmil_length_in_cm'])

        areas = laser_detector.detect ( low_img_thresh = cfg['pix_thresh_lower_bound']['thresholding_method'], 
                                        high_img_thresh = cfg['pix_thresh_upper_bound']['thresholding_method'], 
                                        DLC_p_cutoff_ranges = cfg['DLC_p_cutoff_ranges'],
                                        nb_frames = None,
                                        constrain_frame= cfg['constrain_frame'],
                                        max_dev_in_cm = cfg['max_dev_in_cm'],
                                        DLC_body_label= cfg['DLC_body_label'])
        print('video file read.')
        
        
        analogpulse = AnalogPulse(sorted_exp.files['analogpulse'],
                                  stim_type = sorted_exp.stim_type)

        pulse = Pulse(areas, fs = cfg['fps'], 
                      enforce_use_laser_detection_only = cfg['enforce_use_laser_detection_only'],
                      use_laser_detection_if_no_analogpulse = cfg['use_laser_detection_if_no_analogpulse'],
                      true_duration = sorted_exp.stim_duration)
        
        # analogpulse.remove_pulses( -(np.arange(6)+1)) ## if analogpulse has more pulses input the indicies of the pulses to delete
        # pulse.cut_sig(11700)                           ## if there are problems in the signal cut it here
        

        pulse.find_events(gauss_window = cfg['gauss_window'], 
                          *cfg['bandpass_frequency'],
                          filt_order = cfg['filter_order'],
                          peak_heights = cfg['start_end_h_thresh'])
        
        pulse.verify_nb_with_analog(
                                    sorted_exp.video.name_base,
                                    laser_detection_path,
                                    analogpulse, 
                                    center_vicinity_h_thresh = cfg['center_vicinity_h_thresh'],
                                    max_iteration = cfg['max_iteration'],
                                    crude_smooth_wind = cfg['crude_smooth_wind'])
        
        pulse.determine_start_ends()
        # ax = pulse.plot_start_ends()
        pulse.remove_problematic_detections()
        pulse.find_centers()
        
        true_duration = pulse.find_duration(analogpulse)
            
        ax = pulse.plot_centers()
        
        starts, ends, centers = sorted_exp.get_laser_start_end( pulse, analogpulse, true_duration)

        # centers = remove_element(centers, 14)
        # starts = remove_element(starts, 14)
        # ends = remove_element(ends, 14)
        
        # centers = centers + 8
        # ends = ends + 8
        # starts = starts[:-1]
        # ends = ends[:-1]
        # centers = centers[:-1]
        ax_superimp = pulse.plot_superimposed(starts, ends, centers, true_duration)
        
        sorted_exp.save_laser_detections (starts, ends, 
                                          pulse.method, 
                                          pulse.shift_rel_to_analogpulse, 
                                          pulse.shift_rel_to_analogpulse_sd)
        sorted_exp.plot_laser_detections( ax, 
                                         centers, 
                                         pulse.normalize(pulse.smoothed_sig)[centers],
                                         title = pulse.method)


        filename = sorted_exp.video.name_base + '_constrained_'  + str(cfg['constrain_frame'])
        pulse.save_figs( ax, ax_superimp, 
                        [laser_detection_path, os.path.join(sorted_exp.exp_dir, 'Laser')], 
                        filename) 
        
        sorted_exp.add_file_to_csv(os.path.join(laser_detection_path, 
                                                      'Analysis_summary.csv'),
                                    pulse.method, 
                                    i, pulse.shift_rel_to_analogpulse, 
                                    pulse.shift_rel_to_analogpulse_sd)
        
        sorted_exp.remove_resolved_file_from_csv(os.path.join(laser_detection_path, 
                                                      'Problematic_files.csv'), filepath)
    else:
        
        continue
        
    # except Exception as error:
        
    #     print(error)
        
    #     sorted_exp.add_file_to_csv(os.path.join(laser_detection_path, 
    #                                                   'Problematic_files.csv'),
    #                                     error, i)
    #     continue
    
