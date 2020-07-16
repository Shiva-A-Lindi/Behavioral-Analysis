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
#    for i in videofile_path:
#        print(i)
    return videofile_path


def find_broken_indx(videoPath):
    video=Video()
    video.capture = cv2.VideoCapture(videoPath)
    video.nbFrames=int(video.capture.get(7))
    print(video.nbFrames)
    broken_indx_paths = []
    if video.nbFrames != 0:
        frame = 0
        while(video.capture.isOpened()):
            ret, video.currentFrame = video.capture.read()
            try:  
                frame += 1
                a = np.sum(video.currentFrame)/np.count_nonzero(video.currentFrame)
            except TypeError:
                broken_indx_paths.append(videoPath)
                print("err ",frame)
                break
    return broken_indx_paths

def repair_broken_index(video_path_list):
    for f in video_path_list:
        os.system('ffmpeg -i {0} -c copy {1}'.format(f,f))
path = '/media/shiva/LaCie/VideoRat_Sophie/videos_Rat14/22-05-19/Rat 14  6OHDA all head 22-05-19_20190522_134611_C001H001S0001'
#videofile_path_list = replace_space_with_underscore('/media/shiva/LaCie/VideoRat_Sophie') 
#broken_indx_paths = find_broken_indx(videofile_path_list[0])
#print(broken_indx_paths)        
repair_broken_index(broken_indx_paths)   
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

