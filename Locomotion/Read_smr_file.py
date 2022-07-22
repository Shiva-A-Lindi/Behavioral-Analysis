#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:58:00 2021

@author: shiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import subprocess

# implement pip as a subprocess:

### first install quantities
# try:
#     import quantities
# except (ImportError, ModuleNotFoundError):
#     # !conda install -c conda-forge quantities
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'quantities'])
try:
    import neo
except (ImportError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'neo'])
    import neo
try:
    import sonpy
    
except (ImportError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sonpy'])
    import sonpy
try:
    import scipy
except (ImportError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
    import scipy

# import neo
# import scipy

def build_filePath_list(path, extensions):
    '''go over the directory tree in path and find all files with extensions included in <extensions>'''
    fname = []
    for (dirpath, dirnames, filenames) in os.walk(path):
    
        for f in filenames:

            fname.append(os.path.join(dirpath, f))
    file_path_list = [ fi for fi in fname if (fi.endswith( tuple(extensions) ))]

    return file_path_list

def save_list_to_txt(filepath, liste):
    
    # with open( filepath, 'w', newline='') as file:
    #     wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     wr.writerow(liste)
    with open(filepath, 'w') as f:
        for item in liste:
            f.write("%s\n" % item)

def find_which_channel(analog_signals):
    
    stds = []
    for c in range (len (analog_signals)):
        
        stds.append( np.std (analog_signals[c]) )
        
    return np.argmax ( np.array ( stds) )

def read_laser_file(filepath, neo_version_check):
    
    filename = os.path.basename(filepath)
    file_extension = filename.split('.') [-1]
    
    ## Depending on the file extension, the laser information is stored differently
    if file_extension == 'smrx':
        
        if not neo_version_check:
            
            enforce_neo_version_satisfaction()
            neo_version_check = True
            
        neo_obj = neo.CedIO(filepath)
        analogsignals = read_neo_file_return_analogsignals(filename, neo_obj)
        
        # keep the signal as a 16 bit float
        channel = find_which_channel(analogsignals)
        laser_series = np.float16(analogsignals[channel]) # the laser information is stored as the second analog signal

    else:
        
        neo_obj = neo.Spike2IO(filepath)
        analogsignals = read_neo_file_return_analogsignals(filename, neo_obj)
        
        # keep the signal as a 16 bit float
        laser_series = np.float16(analogsignals[1][:,1]) # the laser information is stored as the second column in the 
                                             # the second analog signal
        
    return laser_series    

def read_neo_file_return_analogsignals(filename, neo_obj):
    
    block = neo_obj.read()[0] # read the file 
    analogsignals = block.segments[0].analogsignals
    report_info_on_file(filename, block, analogsignals)
    
    return analogsignals

def report_info_on_file(filename, block, analogsignals):
    
    print("file = ", filename)
    print('number of segments = ', len(block.segments))
    print('number of analog signals = ', len(analogsignals))
    
    for i in range(len(analogsignals)):
        
        print('signal {} contains series of shape {} '.format(i+1, analogsignals[i].shape))
     
def laser_start_end(laser_series, laser_threshold = 4):
    
    peaks = (laser_series > laser_threshold).reshape(1,-1)[0]
    peaks = peaks * 1
    shifted_right = np.roll(peaks, 1)
    shifted_left = np.roll(peaks, -1)

    laser_start = np.where(peaks - shifted_right > 0)[0]
    laser_end = np.where(peaks - shifted_left > 0)[0]
    
    # plt.figure()
    # plt.plot(laser_series)
    # plt.plot(laser_start, np.ones_like(laser_start), 'x', c ='r')
    # plt.plot(laser_end, np.ones_like(laser_end), 'x', c ='g')
    
    

    return laser_start, laser_end

def laser_start_and_end_BetaPulse(laser_series, 
                                  peak_height_thresh = 1, 
                                  min_dist_bet_laser_coef = 1.5):
    
    laser_series = np.array(laser_series)
    laser_series = laser_series.reshape(-1,)
    peaks,_ = scipy.signal.find_peaks(laser_series, height = peak_height_thresh)
    peaks = np.array(peaks)
    dist_bet_peaks = np.diff(peaks)
    
    most_freq = np.argmax(np.bincount(dist_bet_peaks))
    
    laser_end = peaks[np.where(dist_bet_peaks > min_dist_bet_laser_coef * most_freq)]
    laser_start = peaks[np.where(dist_bet_peaks > 1.5 * most_freq)[0]+1]
    laser_start = np.insert(laser_start, 0, peaks[0] )
    laser_end = np.append(laser_end, peaks[-1])
    laser_end = np.array(laser_end).reshape(-1,)
    
    ## shift back the starts a quarter of a mini pulse
    quarter_wavelength = int((np.min(dist_bet_peaks)) / 4)
    laser_start = laser_start - quarter_wavelength
    
    ## shift forward the starts a quarter of a mini pulse
    half_wavelength = int((np.min(dist_bet_peaks)) / 2)
    laser_end = laser_end + quarter_wavelength
    
    print("{} stimulations".format(len(laser_start)))
    
    return laser_start, laser_end



def determine_stim_type( jumps, laser_series, fs = 1000, freq_thresh = 0.5) :

    pts = int (np.prod(laser_series.shape) ) # number of datapoints
    secs = pts / fs # recording length in seconds
    frequency_of_peaks = len( jumps )/ secs
    
    if frequency_of_peaks > freq_thresh : 
        
        stim_type = "BetaPulse"
        
    else: 
        
        stim_type = "SquarePulse"
    
    print("Signal is {}. (freq > {} Hz = BetaPulse) ".\
          format(stim_type, freq_thresh))
        
    return stim_type

def save_laser_stamps_to_csv(filepath, laser_start, laser_end):
    
    metadatas=[
                [filepath],
                ["framerate : "],
                ["csv_aligned :", 'NO'],
                ["first_laser_time : ", 'None']
               ]
    
    df = pd.DataFrame(np.concatenate((laser_start.reshape(-1,1), laser_end.reshape(-1,1)),
                                      axis = 1) , 
                      columns = ['ON', 'OFF'])

    resultFilePath = filepath.replace(os.path.splitext(filepath)[1], '.csv')
    
    with open(resultFilePath, 'w') as resultfile:

        csvResult=csv.writer(resultfile,delimiter=',', lineterminator='\n')
        csvResult.writerows(metadatas)

    df.to_csv(resultFilePath, mode = 'a', index = False)
    
def analyze_all_files(file_path_list, laser_threshold, neo_version_check):
    
    plotted = False
    
    for filepath in file_path_list:

        laser_series = read_laser_file(filepath, neo_version_check)
        
        #### first agnostically measure oscillations
        laser_start, laser_end = laser_start_end(laser_series, 
                                                 laser_threshold = laser_threshold)
        stim_type = determine_stim_type(laser_start, laser_series, freq_thresh = 0.5, fs = 1000 )
        
        if stim_type == 'SquarePulse':
            
            
            print("{} stimulations".format(len(laser_start)))
        else:
            
            laser_start, laser_end = laser_start_and_end_BetaPulse(laser_series, 
                                                                   peak_height_thresh = 1, 
                                                                   min_dist_bet_laser_coef = 1.5)
            
        save_laser_stamps_to_csv(filepath, laser_start, laser_end)
        
        if not plotted:
            
            plot_one_stim(laser_series, laser_start, laser_end)
            plotted = True
            
def get_laser_thresh():
    
    laser_threshold = input("Enter the laser threshold (microvolts) above which the laser is considered on ([4]/x).")
    
    if laser_threshold == "":
        
        laser_threshold = 4
        
    else:
        
        laser_threshold = int(laser_threshold)
        
    return laser_threshold

def enforce_neo_version_satisfaction():
    
    v1, v2, v3 = neo_version.split('.')
    
    if v1 == 0 and v2 < 10:
        
        path_to_package = input("neo version >= 0.10.0 is required in order to read .smrx files. \n \
                                Please download the latest release from : \
                                https://github.com/NeuralEnsemble/python-neo/releases/tag/0.10.0, \n \
                                Then, extract the zip and input the absolute path \
                                to the extracted folder here(e.g. \n /home/User-name/Downloads/python-neo-0.10.0): \n")
                                
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               path_to_package])
        
def plot_one_stim(laser_series, laser_start, laser_end, fs = 1000):
    
    pts = int(np.prod(laser_series.shape)) # number of datapoints
    secs = pts / fs # recording length in seconds
    time = np.linspace(0, secs, pts)
    
    miny = np.min(laser_series) ; maxy = np.max(laser_series)* 1.5
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(time , laser_series, 'b') 
    ax.set_xlabel('time (s)', fontsize=12)
    ax.set_ylabel('microvolts', fontsize=12)
    ax.axvspan(time[laser_start[0]], time[laser_end[0]],  facecolor='y', alpha=0.2)
    
    first_laser_duration = time[laser_end[0]] - time[laser_start[0]]
    
    ax.set_xlim(time[laser_start[0]] - first_laser_duration, time[laser_end[0]] + first_laser_duration)
    ax.set_ylim(miny, maxy)
    plt.tight_layout()
    
if __name__ == '__main__':
    
    global neo_version_check
    neo_version_check = False
    neo_version = neo.__version__
    
    plt.close('all')
    print( "Neo package version is : {} \n".format(neo_version) )
    
    while True:
        
        try:
            
            path = input("Enter the full path under which the .smr/.smrx file hierarchy resides:  \n")     
            
            if os.path.exists(path):
                
                break
            
            else:
                
                print("path doesn't exist. Try agin.\n")
                
        except KeyboardInterrupt:
            
            print('sth went wrong')
            
    # path = '/media/shiva/LaCie/Data_INCIA_Shiva/2021_02_19_newTrain2_Video_empty'
    # extension_ind = input("Enter the number corresponding to the right file extension: \n 1. .smr           2. .smrx : \n")  
    laser_threshold = get_laser_thresh()
    extensions = [".smr", ".smrx"]
    # extension = extensions[ int(extension_ind) -1 ]

    file_path_list = build_filePath_list(path, extensions)
    path_to_txt = os.path.join(path, "list_of_smr_files_to_read.txt")
    # save_list_to_txt( path_to_txt, 
    #                   file_path_list)
    
    while True:
        
        if_continue = input('{} files found. Continue with analysis of all ([y]/n)'.format(len(file_path_list)))
        
        if if_continue not in  ['y', 'n', 'Y', 'N' ,'']:
            
            print("Sorry, response must be 'y' or 'n'.")
            continue
        
        else: 
            
            if if_continue in ['', 'Y', 'y']:
                
                if_continue = True
                
            else:
                
                if_continue = False
                
            # we're happy with the value given.
            # we're ready to exit the loop.
            break
    
    if if_continue:
        
        analyze_all_files(file_path_list, laser_threshold, neo_version_check)
