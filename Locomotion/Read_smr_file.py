#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:58:00 2021

@author: shiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neo
import sys
import os
import csv
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'neo'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sonpy'])



def build_fileoPath_list(path, extensions):
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
        laser_series = np.float16(analogsignals[1]) # the laser information is stored as the second analog signal
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
     
def laser_start_and_end(laser_series, laser_threshold = 4):
    
    peaks = (laser_series > laser_threshold).reshape(1,-1)[0]
    peaks = peaks * 1
    shifted_right = np.roll(peaks, 1)
    shifted_left = np.roll(peaks, -1)

    laser_start = np.where(peaks - shifted_right > 0)[0]
    laser_end = np.where(peaks - shifted_left > 0)[0]
    return laser_start, laser_end

def save_laser_stamps_to_csv(filepath, laser_start, laser_end):
    
    metadatas=[[filepath],
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
        laser_start, laser_end = laser_start_and_end(laser_series, laser_threshold = laser_threshold)
        save_laser_stamps_to_csv(filepath, laser_start, laser_end)
        if not plotted:
            print("hey")
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
        
def plot_one_stim(laser_series, laser_start, laser_end):
    
    fs = 1000 # sampling rate
    pts = int(np.prod(laser_series.shape)) # number of datapoints
    secs = pts/fs # recording length in seconds
    time = np.linspace(0, secs, pts)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    miny = np.min(laser_series) ; maxy = np.max(laser_series)* 1.5
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
    print("Neo package version is : {}".format(neo_version) )
    path = input("Enter the full path under which the .smr/.smrx file hierarchy resides:  \n")     
    # extension_ind = input("Enter the number corresponding to the right file extension: \n 1. .smr           2. .smrx : \n")  
    laser_threshold = get_laser_thresh()
    extensions = [".smr", ".smrx"]
    # extension = extensions[ int(extension_ind) -1 ]

    file_path_list = build_fileoPath_list(path, extensions)
    path_to_txt = os.path.join(path, "list_of_smr_files_to_read.txt")
    save_list_to_txt( path_to_txt, 
                     file_path_list)
    while True:
        if_continue = input('List of all files found are saved at {}. Continue with analysis of all ([y]/n)'.format(path_to_txt))
        if if_continue not in  ['y', 'n', 'Y', 'N' ,'']:
            print("Sorry, response must be <y> or <n>.")
            continue
        else: 
            if if_continue in ['', 'Y', 'y']:
                if_continue = True
            else:
                if_continue = False
            #we're happy with the value given.
            #we're ready to exit the loop.
            break
    if if_continue:
        analyze_all_files(file_path_list, laser_threshold, neo_version_check)
