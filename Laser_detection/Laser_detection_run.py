#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:50:07 2022

@author: shiva
"""
import os, sys
current_path = os.getcwd()
sys.path.insert(1, os.path.join(os.path.dirname(current_path), 'Locomotion'))
from File_hierarchy import *
from Laser_detection import *

if __name__ == '__main__':


    # path = input('Input project path: \n')
    # project_name = input('Input the number corresponding to your project: \n \
    #                       1. Treadmill \n \
    #                       2. Open field \n')
    project_name = '2'                  
    path = '/media/shiva/LaCie/Lise_unsupervised_Behavior_class/'
    laser_detection_path = os.path.join(path,  'LASER_DETECTION')
    Directory.create_dir_if_not_exist(laser_detection_path)
    configpath = set_config_file(path, project_name, rewrite_existing = True)
    
    print('config file created at \n {}. You may adjust the parameters and resave.\n'.format(configpath))
    
    # stdin = input('Press ENTER if you wish to analyze all the video files in the project path directory, otherwise please input the path to a csv file containing the paths to your desired files.\n')
    
    # if stdin == '':
        
    #     directory = Directory(path)
    #     video_filepath_list = directory.get_spec_files( extensions= ['.avi', '.mp4', '.mov', '.MP4'])
            
    # elif os.path.exists(stdin):
        
    #     video_filepath_list = SortedExpeiment.get_videofile_paths_from_csv(stdin)
        
    # else:
        
    #     raise ValueError ('Invalid input. Please try again.')
    # print( '{} experiment files found.'.format(len(video_filepath_list)) )
    
    video_filepath_list = ['/media/shiva/LaCie/Lise_unsupervised_Behavior_class/FoxP2_31_140322_L-DOPA_GP_10ms_bottom.MP4']
    SortedExpeiment.create_problematic_csv(os.path.join(laser_detection_path, 'Problematic_files.csv'))
    SortedExpeiment.create_summary_csv(os.path.join(laser_detection_path, 'Analysis_summary.csv'))
    cfg = read_config(configpath)

    n_prob = 0
    for i, filepath in enumerate(video_filepath_list):
    
        plt.close( 'all' )
        print('Analyzing {} from {} files.'.format(i + 1, len (video_filepath_list)))
        
    
        try: 
        
            sorted_exp = SortedExpeiment(filepath, 
                                         extract_info_from_file= cfg['extract_info_from_file'],
                                         stim_duration_dict = cfg['stim_duration_dict'],
                                         stim_duration = cfg['stim_duration'],
                                         fps = cfg['fps'],
                                         inter_stim_interval = cfg['inter_stim_interval'],
                                         experiment = { '1': 'treadmill', '2': 'OF'}[project_name])
        
                
            if not sorted_exp.already_analyzed or cfg['reanalyze_existing']:
                
                laser_detector = LaserDetector( sorted_exp.video.path,  
                                                sorted_exp.DLC_path,
                                                thresh_method = cfg['thresholding_method'],
                                                area_cal_method = cfg['area_cal_method'],
                                                image_parts = ['upper', 'lower'],
                                                treadmill_length_in_cm = cfg['treadmill_length_in_cm'])
        
                areas, nb_frames = laser_detector.detect( low_img_thresh = tuple( cfg['pix_thresh_lower_bound'][
                                                                                cfg['thresholding_method']]), 
                                                        high_img_thresh = tuple(cfg['pix_thresh_upper_bound'][
                                                                                cfg['thresholding_method']]), 
                                                        DLC_p_cutoff_ranges = cfg['DLC_p_cutoff_ranges'],
                                                        nb_frames = None,
                                                        constrain_frame= cfg['constrain_frame'],
                                                        max_dev_in_cm = cfg['max_dev_in_cm'],
                                                        DLC_body_label= cfg['DLC_label'])
                print('video file read. Finding pulses...')
                
                
                analogpulse = AnalogPulse(sorted_exp.files['analogpulse'],
                                          stim_type = sorted_exp.stim_type)
                
                if cfg['experiment'] == 'locomotion treadmill':
                    
                    pulse = Pulse(areas, fs = cfg['fps'], nb_frames = nb_frames,
                                  enforce_use_laser_detection_only = cfg['enforce_use_laser_detection_only'],
                                  use_laser_detection_if_no_analogpulse = cfg['use_laser_detection_if_no_analogpulse'],
                                  true_duration = sorted_exp.stim_duration)
                    
                elif cfg['experiment'] == 'Open field':
                    
                    pulse = LongPulse(areas, fs = cfg['fps'], nb_frames = nb_frames,
                                      inter_stim_interval = sorted_exp.inter_stim_interval,
                                      enforce_use_laser_detection_only = cfg['enforce_use_laser_detection_only'],
                                      use_laser_detection_if_no_analogpulse = cfg['use_laser_detection_if_no_analogpulse'],
                                      true_duration = sorted_exp.stim_duration,
                                      min_acc_dev = cfg['min_acc_dev'], max_acc_dev = cfg['max_acc_dev'])
                else:
                    
                    raise ValueError ('Experiment must be either treadmill or open field!')
                    
                # analogpulse.remove_pulses(ind_pulses) ## if analogpulse has more pulses input the indicies of the pulses to delete
                # pulse.cut_sig(frame_to_cut)           ## if there are problems in the signal cut it here
                
                pulse.pre_process(gauss_window = cfg['gauss_window'], 
                                  low_f = cfg['bandpass_frequency'][0],
                                  high_f = cfg['bandpass_frequency'][1],
                                  filt_order = cfg['filter_order'])
                
                pulse.find_center_vicinities(
                                            sorted_exp.video.name_base,
                                            sorted_exp.exp_dir,
                                            laser_detection_path,
                                            analogpulse, 
                                            center_vicinity_h_thresh = cfg['center_vicinity_h_thresh'],
                                            max_iteration = cfg['max_iteration'],
                                            crude_smooth_wind = cfg['crude_smooth_wind'])
                

                pulse.determine_start_ends(thresh = cfg['start_end_h_thresh'])
                pulse.handle_problematic_detections()
                pulse.cal_centers()
                
                true_duration = pulse.find_duration(analogpulse)
                    
                ax = pulse.plot_centers()
                
                
                starts, ends, centers = sorted_exp.get_laser_start_end( pulse.starts, pulse.ends, pulse, 
                                                                       analogpulse, true_duration)
        
                # centers = remove_element(centers, 14) # to remove certain elements

                ax_superimp, title = pulse.plot_superimposed(starts, ends, centers, true_duration)
                
                sorted_exp.save_laser_detections (starts, ends, 
                                                  pulse.method, 
                                                  pulse.shift_rel_to_analogpulse, 
                                                  pulse.shift_rel_to_analogpulse_sd,
                                                  note = pulse.note,
                                                  nb_frames = nb_frames,
                                                  no_solid_detection = pulse.no_solid_detection)
                sorted_exp.plot_laser_detections( ax, 
                                                  centers, 
                                                  pulse.normalize(pulse.smoothed_sig)[centers],
                                                  title = title)
        
        
                filename = sorted_exp.create_figname(cfg['constrain_frame'])
                pulse.save_figs( ax, ax_superimp, 
                                [laser_detection_path, os.path.join(sorted_exp.exp_dir, 'Laser')], 
                                filename) 
                pulse.raise_err_if_no_solid_detection()
                sorted_exp.add_file_to_csv(os.path.join(laser_detection_path, 
                                                              'Analysis_summary.csv'),
                                            pulse.method, 
                                            i, pulse.shift_rel_to_analogpulse, 
                                            pulse.shift_rel_to_analogpulse_sd,
                                            note = pulse.note)
                
                sorted_exp.remove_resolved_file_from_csv(os.path.join(laser_detection_path, 
                                                              'Problematic_files.csv'), filepath)
            else:
                
                continue
            
        except Exception as error:
            
            print(error)
            
            sorted_exp.add_file_to_csv(os.path.join(laser_detection_path, 
                                                          'Problematic_files.csv'),
                                            error, i)
            n_prob += 1
            continue
        
    print('faild to analyze {} file(s). Refer to the report file "Problematic_files.csv" saved at \n {} for more information.'.format(n_prob, 
                                                                                                          laser_detection_path))