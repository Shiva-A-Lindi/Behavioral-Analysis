#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:50:07 2022

@author: shiva
"""

from Sort_exp_files_into_hierarchy import *
from Laser_detection import *

# area_cal_method = 'contour'
area_cal_method = 'pix_count'

treadmil_length_in_cm = 37.6
constrain_frame= True
max_iteration = 50

# enforce_use_laser_detection_only = False
enforce_use_laser_detection_only = True

stim_duration = 125


# RGB_blueLower = (50, 80, 60) ## HSV
# RGB_blueUpper = (150, 255, 255)

RGB_blueLower = (150, 10,60)
RGB_blueUpper = (255, 120, 140)
center_vicinity_h_thresh = 0.3
start_end_h_thresh = 0.3
path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre'
laser_detection_path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/LASER_DETECTION'

directory = Directory(path)
# video_filepath_list = directory.get_spec_files( extensions= ['.avi', '.mp4', '.mov'])
video_filepath_list = ['/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/ChR2/Mouse_58/STR/squarepulse_0-5mW/Video/Vglut2D2Cre#58_SquarePulse_STR_0-5mW_15cm-s_Stacked_m01.avi']
# video_filepath_list = SortedExpeiment.read_summary_csv(os.path.join(laser_detection_path, 'Problematic_files.csv'))

print( '{} experiment files found.'.format(len(video_filepath_list)) )

SortedExpeiment.create_problematic_csv(os.path.join(laser_detection_path, 'Problematic_files.csv'))
SortedExpeiment.create_summary_csv(os.path.join(laser_detection_path, 'Analysis_summary.csv'))


for i, filepath in enumerate(video_filepath_list):

    plt.close( 'all' )
    print('{} from {} files.'.format(i, len (video_filepath_list)))
    

    try: 
        sorted_exp = SortedExpeiment(filepath)
        
        laser_detector = LaserDetector( sorted_exp.video.path,  
                                        sorted_exp.files['DLC'].path,
                                        thresh_method = 'rgb',
                                        area_cal_method = area_cal_method,
                                        image_parts = ['upper', 'lower'],
                                        treadmil_length_in_cm = treadmil_length_in_cm)
        
        areas = laser_detector.detect ( low_img_thresh = RGB_blueLower, 
                                        high_img_thresh = RGB_blueUpper, 
                                        p_cutoff_ranges = 0.995,
                                        nb_frames = None,
                                        constrain_frame= constrain_frame,
                                        max_dev_in_cm = 1.7)
        print('video file read.')
        smr = AnalogPulse(sorted_exp.files['smr'].path)
        # smr.remove_pulses( np.arange(-21,0,1)) ################ if smr has more pulses input the indicies of the pulses to delete
        pulse = Pulse(areas, fs = 250, 
                      enforce_use_laser_detection_only = enforce_use_laser_detection_only,
                      true_duration = stim_duration)
        # pulse.cut_sig(27560) ### if there are problems in the signal cut it here
        pulse.find_events(gauss_window = 10, 
                          low_f = 1, high_f = 50, 
                          filt_order = 10,
                          peak_heights = start_end_h_thresh)
        
        pulse.check_detected_nb(smr, 
                                center_vicinity_h_thresh,
                                sorted_exp.video.name_base,
                                laser_detection_path,
                                max_iteration = max_iteration)
        
        pulse.determine_start_ends()
        ax = pulse.plot_start_ends()
        pulse.remove_problematic_detections()
        pulse.find_centers()
        
        if not enforce_use_laser_detection_only:
            
            pulse.fill_missing_pulses(smr, plot = False, report = False)
            true_duration = smr.true_duration
            
        else:
            
            true_duration = pulse.true_duration
            
        ax = pulse.plot_centers()
        
        starts, ends, centers = sorted_exp.get_laser_start_end( pulse, smr)
        
        ax_superimp = pulse.plot_superimposed(starts, ends, centers, true_duration)
        
        sorted_exp.save_laser_detections (starts, ends, 
                                          pulse.method, 
                                          pulse.shift_rel_to_smr, 
                                          pulse.shift_rel_to_smr_sd)
        sorted_exp.plot_laser_detections( ax, 
                                         centers, 
                                         pulse.normalize(pulse.smoothed_sig)[centers],
                                         title = pulse.method)
        
        
        filename = sorted_exp.video.name_base + '_constrained_'  + str(constrain_frame)
        pulse.save_figs( ax, ax_superimp, 
                        [laser_detection_path, sorted_exp.files['smr'].dirpath], 
                        filename) 
        
        sorted_exp.add_file_to_csv(os.path.join(laser_detection_path, 
                                                      'Analysis_summary.csv'),
                                    pulse.method, 
                                    i, pulse.shift_rel_to_smr, 
                                    pulse.shift_rel_to_smr_sd)
        
        sorted_exp.remove_resolved_file_from_csv(os.path.join(laser_detection_path, 
                                                      'Problematic_files.csv'), filepath)
        
    except Exception as error:
        
        # sorted_exp.add_file_to_csv(os.path.join(laser_detection_path, 
        #                                               'Problematic_files.csv'),
        #                                 error, i)
        continue
    
