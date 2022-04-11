#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:40:00 2020

@author: shiva
"""
import os
from os import walk
import sys
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import timeit
import shutil

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

class Video:
    """class for videos
    """
    def __init__(self):
        self.file=None
        self.capture=None
        self.grabber=None
        #vidProp_dict['imageType']=image.dtype
        self.ToRGB=None #vidProp_dict['videoToRGB']=cap.get(16)
        self.width=None #vidProp_dict['videoWidth']=cap.get(3)
        self.height=None #vidProp_dict['videoHeight']=cap.get(4)      
        self.fps=None #vidProp_dict['fps']=cap.get(5)
        self.nbFrames=None #vidProp_dict['nbFrames']=cap.get(7)
        self.pos=None #vidProp_dict['framePos']= cap.get(1) -->position du frame suivant       
        self.playing=False
        self.currentFrame=None

def replace_space_with_underscore(path):
    '''go over the directory tree in path and find ll .avi files'''
    fname = []
    for (dirpath, dirnames, filenames) in walk(path):
    
        for f in filenames:
            os.rename(os.path.join(dirpath, f),os.path.join(dirpath, f.replace(' ', '_')))
            fname.append(os.path.join(dirpath, f))
        for i in range(len(dirnames)):
            new_name = dirnames[i].replace(' ', '_')
            os.rename(os.path.join(dirpath, dirnames[i]), os.path.join(dirpath, new_name))
            dirnames[i] = new_name
    # videofile_path = [ fi for fi in fname if fi.endswith(".avi") ]

    # return videofile_path


def create_path_csv(videofile_path_list):
    '''create the csv file for the report'''
    
    folders = [len(file_path.split(os.sep)) for file_path in videofile_path_list]
    col_str = []
    if len(folders) !=0 :
        cols = np.arange(max(folders))
        col_str = ['dir_'+str(i) for i in cols]
    col_str.insert(0,'filename')
    df = pd.DataFrame(columns = col_str)
    for i in range(len(videofile_path_list)):
        hierarchy = videofile_path_list[i].split(os.sep)
        hierarchy = hierarchy[::-1] # reverse the order of file directories
#        print(col_str[:len(hierarchy)])
        df.loc[i, col_str[:len(hierarchy)]] = hierarchy 
    df.insert(0, 'index',['']*len(videofile_path_list))
    df.insert(1, '%good_frames',['']*len(videofile_path_list))
    df.insert(2, '#total_frames_of_video',['']*len(videofile_path_list))
    df.insert(3, 'renamed',['n']*len(videofile_path_list))
    return df


def find_broken_indx(videoPath):
    '''open video and read frames, report either "non-readable" for no frames, "broken" for when finding corrupted frames or "correct" when the file is intact'''
    
    video=Video()
    video.capture = cv2.VideoCapture(videoPath)
    video.nbFrames=int(video.capture.get(7))
    print(video.nbFrames)
#    broken_indx_paths = []
    perc = 100
    state = 'correct'
    n_frames = video.nbFrames
    if video.nbFrames != 0:
        frame = 0
        while(video.capture.isOpened() and frame<n_frames):
            ret, video.currentFrame = video.capture.read()
            if ret:
                frame += 1
                # print(frame)
            else: 
                state = 'broken' ; perc =round(frame/video.nbFrames*100,1)
                print("state =",state, ' ', perc," %")
                break
    else: # file was not readable
        state = 'not_readable'
        perc = 0
    if state == 'correct': # lastly if the file is intact
        print("state =",state, ' ', perc," %")
    return state, perc, n_frames



def repair_broken_index(video_path_list):
    '''Do Not USE. This removes the corrupted frames of the videos'''
    for f in video_path_list:
        os.system('ffmpeg -i {0} -c copy {1}'.format(f,f))
        
def check_file_integrity(videofile_path_list,csv_path,df):
    
    
    i = 0
    for filename in videofile_path_list:
        start = timeit.default_timer()
        try:
            print(i, " from ", len(videofile_path_list))
            state, perc, n_frames = find_broken_indx(filename)
            df.loc[i,['index', '%good_frames', '#total_frames_of_video']] = state, perc, n_frames
            path = Path(filename)
            # filename_without_prefix = os.path.splitext(filename)[0]
            if not state in path.stem:
                new_filename = path.with_name(f"{path.stem}_{state}_{str(int(round(perc,0)))}{path.suffix}")
                try:
                    os.rename(filename,new_filename)
                    df.loc[i,'renamed'] = 'y'
                except OSError:
                    continue
        except Exception as e:
            print(e)
            break
        i+=1
        stop = timeit.default_timer()
        print("t = ", stop - start)
    df.to_csv(csv_path, sep=',',index=False)

    return df
  
def move_files_out_of_subfolder(dirpath,filename):
    '''if the dirpath ends with a subfolder with the same name of the objects within, move the objects one step up in the directory tree and remove empty folder'''
    last_sub_folder = dirpath.split(os.sep)[-1]
    filename_without_prefix = os.path.splitext(filename)[0]
    if last_sub_folder == filename_without_prefix:
        # print('haha',os.path.split(dirpath)[0])
        shutil.move(os.path.join(dirpath,filename), os.path.join(os.path.split(dirpath)[0],filename))

            
def list_video_files(path, move_files = False,remove_emp_files = False):
    '''go over the directory tree in path and find ll .avi files'''
    fname = []
    for (dirpath, dirnames, filenames) in walk(path):
    
        for f in filenames:
            fname.append(os.path.join(dirpath, f))
            if move_files :  # if you want to take out the files from isolated subfolders
                move_files_out_of_subfolder(dirpath,f)
            if remove_emp_files and len(os.listdir(dirpath)) == 0: # to remove empty folders after moving
                shutil.rmtree(dirpath)
        for i in range(len(dirnames)):
            new_name = dirnames[i].replace(' ', '_')
            dirnames[i] = new_name
    videofile_path = [ fi for fi in fname if (fi.endswith(".avi") or 
                                              fi.endswith(".mp4") or 
                                              fi.endswith(".mkv") or
                                              fi.endswith(".wmv") or
                                              fi.endswith(".mpeg"))]
    return videofile_path      

def extract_useful_info_from_filename(df_original):
    ''' strip the file names to form "Rat_<NUMBER>_<DATE>_<TIME>_<CAMERA>" '''
    # states = ['broken','correct','mot_readable']
    df = df_original.copy()
    
    for i in df.index.to_list():
        filename = os.path.splitext(df['filename'][i])[0]
        suffix = os.path.splitext(df['filename'][i])[1]
        if filename[3] != '_': filename = filename[:3]+'_' + filename[3:] # if its Rat12 instead of Rat_12, add the underscore
        # if any(x in filename for x in states):
        #     df['filename'][i] = "_".join(filename.split('_')[:-2])
        strips = filename.split('_')
        j = 0
        camera_ind_found = False
        for string in strips:
            if string.startswith('C00') and string.endswith('001'): 
                camera_ind_found = True
                ind = j
                break
            j +=1
        if not camera_ind_found:
            print('Camera details dont exist in:' , df['filename'])
            # sys.exit(1)
            continue
        df['filename'][i] = "_".join([strips[0],strips[1],strips[ind-2],strips[ind-1],strips[ind]])+suffix
    return df

def find_same_files_two_destinations(df1,df2):
    ''' look for the same files in both data frames and report back the overlaps'''
    mutual_df = pd.DataFrame(columns=df1.columns)
    parameter = 'filename'
    df1_stripped = extract_useful_info_from_filename(df1)
    df2_stripped = extract_useful_info_from_filename(df2)
    
    for filename in df1_stripped[parameter].values:

        entries = df2_stripped.loc[df2_stripped[parameter] == filename]
        if len(entries) != 0:
            mutual_df = pd.concat([mutual_df,df1.loc[df1_stripped[parameter] == filename]])
            mutual_df = pd.concat([mutual_df,df2.loc[df2_stripped[parameter] == filename]])
            mutual_df = mutual_df.append(pd.Series(name='space',dtype='float64'), ignore_index=False)
    return mutual_df
            
def find_missing_files_two_destinations(df1,df2):
    ''' report items in df1 that are not in df2'''
    difference_df = pd.DataFrame(columns=df1.columns)
    parameter = 'filename'
    df1_stripped = extract_useful_info_from_filename(df1)
    df2_stripped = extract_useful_info_from_filename(df2)
    for filename in df1_stripped[parameter].values:

        entries = df2_stripped.loc[df2_stripped[parameter] == filename]
        if len(entries) == 0:
            difference_df = pd.concat([difference_df,df1.loc[df1_stripped[parameter] == filename]])
    return difference_df

def save_df_of_comparison(csv_path1, csv_path2, merge_csv_path):
    ''' open two csv reports, save identical files history to another csv in "merge_csv_path" '''
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    df = find_same_files_two_destinations(df1,df2)
    df.to_csv(merge_csv_path, sep=',',index=False)
    
def save_missing_items_between_two_lists(csv_path1, csv_path2, diff_csv_path):
    ''' save the list of the files that exist in one file but not the other'''
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    find_missing_files_two_destinations(df1,df2).to_csv(os.path.splitext(diff_csv_path)[0]+'_1'+os.path.splitext(diff_csv_path)[1], sep=',',index=False)
    find_missing_files_two_destinations(df2,df1).to_csv(os.path.splitext(diff_csv_path)[0]+'_2'+os.path.splitext(diff_csv_path)[1], sep=',',index=False)
    
#%% MAIN
# 1. enter the path of the folder you want to scan
# path = '/home/shiva/smbshare/Master2019_Ana/Vidéos_old/Rat12/Mai' # run 
path = '/home/shiva/smbshare/BackUp_Filer_Rongeur/Nico/Master_Ana_NeuroBim_2019/Vidéos'

# 2. replace the spaces with underscores everywhere in your path
replace_space_with_underscore(path)

# 3. list the video file paths within this directory
videofile_path_list = list_video_files(path, move_files = False, remove_emp_files = False) # set "move_files = True" when you want to get rid of unecessary folders created by the camera and move their files outside


# 4. enter the directory and name of the csv file you want to save your report
csv_directory = path ; csv_filename  = path.split(os.sep)[-1]+'.csv'
csv_path = os.path.join(csv_directory,csv_filename)
# 5. Create the csv file (to be global not to lose if error is thrown)
df = create_path_csv(videofile_path_list)
# 6. Run to check all the videos
df = check_file_integrity(videofile_path_list,csv_path,df)

#%%
#%% If you explicitly want to compare two directories for duplicates look here:

# 1. specify the path of first folder
path1 = '/home/shiva/smbshare/BackUp_Filer_Rongeur/Nico/Master_Ana_NeuroBim_2019/Vidéos'
csv_directory = path1 ; csv_filename  = path1.split(os.sep)[-1]+'.csv'
csv_path1 = os.path.join(csv_directory,csv_filename)

# 1. specify the path of second folder
path2 = '/home/shiva/smbshare/Master2019_Ana/Video_All' # run 
csv_directory = path2 ; csv_filename  = path2.split(os.sep)[-1]+'.csv'
csv_path2 = os.path.join(csv_directory,csv_filename)

# 3. the comparisons csvs will be saved at the common path
csv_directory = os.path.commonpath([path1,path2])
merge_csv_path = os.path.join(csv_directory,'Maste_Ana_2019_BackUp_Filer_Rongeur_Nico_Vidéos_vs_Videos_all.csv')

# 4. find mutual files
save_df_of_comparison(csv_path1, csv_path2, merge_csv_path)

# 5. find differences between files in two directory
save_missing_items_between_two_lists(csv_path1, csv_path2, merge_csv_path)







