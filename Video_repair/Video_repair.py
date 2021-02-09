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
    videofile_path = [ fi for fi in fname if fi.endswith(".avi") ]

    return videofile_path

def list_video_files(path):
    '''go over the directory tree in path and find ll .avi files'''
    fname = []
    for (dirpath, dirnames, filenames) in walk(path):
    
        for f in filenames:
            fname.append(os.path.join(dirpath, f))
        for i in range(len(dirnames)):
            new_name = dirnames[i].replace(' ', '_')
            dirnames[i] = new_name
    videofile_path = [ fi for fi in fname if (fi.endswith(".avi") or 
                                              fi.endswith(".mp4") or 
                                              fi.endswith(".mkv") or
                                              fi.endswith(".mpeg"))]
    return videofile_path

def create_path_csv(videofile_path_list):
    folders = [len(file_path.split(os.sep)) for file_path in videofile_path_list]
    col_str = []
    if len(folders) !=0 :
        cols = np.arange(max(folders))
        col_str = ['dir_'+str(i) for i in cols]
    col_str.insert(0,'file name')
    df = pd.DataFrame(columns = col_str)
    for i in range(len(videofile_path_list)):
        hierarchy = videofile_path_list[i].split(os.sep)
        hierarchy = hierarchy[::-1]
#        print(col_str[:len(hierarchy)])
        df.loc[i, col_str[:len(hierarchy)]] = hierarchy 
    df.insert(0, 'index',['correct']*len(videofile_path_list))
    df.insert(1, '%good_frames',['100']*len(videofile_path_list))
    df.insert(2, '#frames',['']*len(videofile_path_list))

    return df

def find_broken_indx(videoPath):
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
        while(video.capture.isOpened()):
            ret, video.currentFrame = video.capture.read()
            try:  
                frame += 1
                a = np.sum(video.currentFrame)/np.count_nonzero(video.currentFrame)
            except TypeError:
#                broken_indx_paths.append(videoPath)
                state = 'broken' ; perc =round(frame/video.nbFrames*100,1)
                print("state =",state, ' ', perc," %")
                break
    else: # file was not readable
        state = 'not readable'
        perc = 0
    return state, perc, n_frames

def repair_broken_index(video_path_list):
    for f in video_path_list:
        os.system('ffmpeg -i {0} -c copy {1}'.format(f,f))
        
def check_file_integrity(videofile_path_list,csv_path):
    
    df = create_path_csv(videofile_path_list)
    i = 0
    for filename in videofile_path_list:
        print(i, " from ", len(videofile_path_list))
        state, perc, n_frames = find_broken_indx(filename)
        df.loc[i,['index', '%good_frames', '#frames']] = state, perc, n_frames
        if state == 'broken':
            path = Path(filename)
            new_filename = path.with_name(f"{path.stem}_{state}_{str(int(round(perc,0)))}{path.suffix}")
            os.rename(filename,new_filename)
        if state == 'not readable':
            path = Path(filename)
            new_filename = path.with_name(f"{path.stem}_{state}{path.suffix}")
            os.rename(filename,new_filename)
        i+=1
    df.to_csv(csv_path, sep=',',index=False)
    return df
        
#%%
# enter the path of the folder you want to scan
path = '/home/shiva/sambashare/Master2019_Ana/Video_All/Rat14/Juin/05-06-19_IntraLDOPA_10mM'
# enter the file path of the csv file
csv_path = '/home/shiva/sambashare/Master2019_Ana/Rat14.csv'
#filename = 'Rat 14 all head IntraLDOPA 10mM 05-06-19_20190605_135847_C001H002S0001.avi'

videofile_path_list = list_video_files(path)
df = check_file_integrity(videofile_path_list,csv_path)









#%% 
#broken_indx_paths = find_broken_indx(videofile_path_list[0])
#repair_broken_index(broken_indx_paths)   
#def check_video_integrity(videofile_path_list):
#    for filepath in videofile_path_list:
##        print(os.system('ffmpeg -v 5 -i {0} -f null - 2>&1'.format(filepath)))
#        t = os.popen('ffmpeg -v 5 -i "%s" -f null - 2>&1' % filepath).read()
#        t = re.sub(r"frame=.+?\r", "", t)
#        t = re.sub(r"\[(.+?) @ 0x.+?\]", "[\\1]", t)
#        print(t)
#check_video_integrity(videofile_path_list)
#
#os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 output%d.png".format(filename))
#
#t = re.sub(r"frame=.+?\r", "", t)
#t = re.sub(r"\[(.+?) @ 0x.+?\]", "[\\1]", t)
#print(t)

