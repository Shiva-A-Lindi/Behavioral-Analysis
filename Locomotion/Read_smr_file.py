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

!python -m pip install sonpy
!python -m pip install neo

def build_fileoPath_list(path, extensions):
    '''go over the directory tree in path and find all files with extensions included in <extensions>'''
    fname = []
    for (dirpath, dirnames, filenames) in walk(path):
    
        for f in filenames:

            fname.append(os.path.join(dirpath, f))
    file_path_list = [ fi for fi in fname if (fi.endswith( tuple(extensions) ))]

    return file_path_list

def save_list_to_csv(filepath, liste):
    with open( filepath, 'w', newline='') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(liste)

def read_laser_file(filepath):
    filename = os.path.basename(filepath)
    file_extension = filename.split('.') [-1]
    
    ## Depending on the file extension, the laser information is stored differently
    if file_extension == 'smrx':
        neo_obj = neo.CedIO(filepath)
        analogsignals = read_neo_file_return_analogsignals(neo_obj)
        
        # keep the signal as a 16 bit float
        laser_series = np.float16(analogsignals[1]) # the laser information is stored as the second analog signal
    else:
        neo_obj = neo.Spike2IO(filepath)
        analogsignals = read_neo_file_return_analogsignals(neo_obj)
        
        # keep the signal as a 16 bit float
        laser_series = np.float16(analogsignals[1][:,1]) # the laser information is stored as the second column in the 
                                             # the second analog signal
        
    return laser_series    

def read_neo_file_return_analogsignals(neo_obj):
    
    block = neo_obj.read()[0] # read the file 
    analogsignals = block.segments[0].analogsignals
    report_info_on_file(block, analogsignals)
    
    return analogsignals

def report_info_on_file(block, analogsignals):
    
    print('number of segments = ', len(block.segments))
    print('number of analog signals = ', len(analogsignals))
    for i in range(len(analogsignals)):
        print('signal {} contains {} series'.format(i+1, analogsignals[i].shape))
     
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
    resultFilePath = filepath.replace(os.path.splitext(filename)[1], '.csv')
    with open(resultFilePath, 'w') as resultfile:

        csvResult=csv.writer(resultfile,delimiter=',', lineterminator='\n')
        csvResult.writerows(metadatas)

    df.to_csv(resultFilePath, mode = 'a', index = False)
    
def analyze_all_files(file_path_list, laser_threshold):
    for filepath in file_path_list:
        laser_series = read_laser_file(filepath)
        laser_start, laser_end = laser_start_and_end(laser_series, laser_threshold = laser_threshold)
        save_laser_stamps_to_csv(filepath, laser_start, laser_end)
        
def get_laser_thresh():
    laser_threshold = input("Enter the laser threshold (microvolts) above which the laser is considered on. ([4]/x)")
    if laser_threshold == "":
        laser_threshold = 4
    else:
        laser_threshold = int(laser_threshold)
    return laser_threshold

if __name__ == '__main__':
    
    neo_version = neo.__version__
    print("Neo package version is : {}".format(neo_version) )
    path = input("Enter the full path under which the .smr/.smrx file hierarchy resides:  \n")     
    extension_ind = input("Enter the number corresponding to the right file extension: \n 1. .smr           2. .smrx : \n")  
    laser_threshold = get_laser_thresh()
    extensions = [".smr", ".smrx"]
    extension = extensions[ int(extension_ind) -1 ]
    if extension == '.smrx' : 
        v1, v2, v3 = neo_version.split('.')
        if v1 == 0 and v2 < 10:
            path_to_package = input("neo version >= 0.10.0 is required in order to read .smrx files. \n Please download the latest release from : \
                                    https://github.com/NeuralEnsemble/python-neo/releases/tag/0.10.0, \n Then, extract the zip and input the absolute path \
                                        to the extracted folder here(e.g. \n /home/User-name/Downloads/python-neo-0.10.0): \n")
            !python -m pip install path_to_package
        # to do: check which version of neo is installed 
        
    file_path_list = build_fileoPath_list(path, extensions)
    path_to_csv = os.path.join(path, "list_of_smr_files_to_read.csv")
    save_list_to_csv( path_to_csv, 
                     file_path_list)
    while True:
        if_continue = input('List of all files found are saved at {}. Continue with analysis of all? [y/n]'.format(path_to_csv))
        if not in  ['y', 'n', 'Y', 'N']:
            print("Sorry, response must be <y> or <n>.")
            continue
        else:
            #we're happy with the value given.
            #we're ready to exit the loop.
            break
    if if_continue:
        analyze_all_files(file_path_list, laser_threshold)
