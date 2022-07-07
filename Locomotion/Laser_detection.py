#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:11:14 2022

@author: shiva
"""


   
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import pickle
import timeit, time
from threading import Thread

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
 
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue
    
from scipy.signal import butter, sosfilt, sosfreqz, spectrogram, sosfiltfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from pdf2image import convert_from_path

from File_hierarchy import *
import yaml
from ruamel.yaml import YAML

##################### Config file  ##########################################################################################
def create_config_template():
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """

    yaml_str = """\
    # Project definitions (do not edit)
        \n
        experiment: # experiment paradigm 
        \n
    # Project path (change when moving around)
        \n
        project_path: 
        \n
    # Experiment parameters
        \n
        fps : # frame per second of recorded video
        treadmil_length_in_cm : # float, Full treadmill length in cm
        stim_duration_dict : # stim durations (in ms) for different pulses
        stim_duration : # stim duration time (in ms), not necessary if stim type is specified in name and the above dict is specified.
        duration_off_per_cycle : # duration of laser off (in ms)per pulse (normal beta would be 25 ms)
        extract_info_from_file : # True only for locomotion and only if filenames comply with the common naming rules
    # Frame analysis parameters
        \n
        thresholding_method : # RGB or HSV
        pix_thresh_upper_bound : #  (R, G, B) and  (H, S, V)
        pix_thresh_lower_bound : # (R, G, B) and (H, S, V) 
        area_cal_method: # pix_count for counting nb of laser pixels, or contour for analyzing contours around laser (suitable for HSV)

        \n
    # Laser detection parameters
        \n
        use_laser_detection_if_no_analogpulse : # true or false
        enforce_use_laser_detection_only : # true if you don't want to use the analogpulse
        reanalyze_existing: # true if you wish to reanalyze already analyzed videos 
        \n
    # Individual Pulse detection parameters
        \n
        crude_smooth_wind : # gassian kernel window for crudly smoothing 
        center_vicinity_h_thresh: # peak threshold value for crudly smoothed signal 
        gauss_window : # gaussina kernel window for smoothing signal before bandpass filtering
        bandpass_frequency : # (low_f, high_f) tuple specifying the freq band to filter for detecting start end of pulse
        filter_order : # bandpass filter order
        start_end_h_thresh: # peak threshold value for start end event detection using above specified bandpass filtered sig
        max_iteration : # maximum number of iterations for recursive adjustment of peak threshold
        \n
    # DLC Aid parameters
        \n
        constrain_frame : # bool, whether or not constrain searched pixels with DLC analysis
        DLC_label : # str, DLC label to use to constrain pixels
        DLC_p_cutoff_ranges: # float, cutoff label detection liklihood to use to constrain the vertical wherabout of laser
        max_dev_in_cm : # float, maximum deviation in cm around the provided label to constrain pixels.
        
    """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
    
    return cfg_file, ruamelFile
    
def read_config(configpath):
    """
    Reads structured config file defining a project.
    """
    
    ruamelFile = YAML()
    if os.path.exists(configpath):
        try:
            # with open(configpath, "r") as f:
            #     cfg = ruamelFile.load(f)
            a_yaml_file = open(configpath)
            cfg = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
            return cfg
                # curr_dir = os.path.dirname(configname)
                # if cfg["project_path"] != curr_dir:
                #     cfg["project_path"] = curr_dir
                #     write_config(configname, cfg)
        except Exception as err:
            
            if len(err.args) > 2:
                if ( err.args[2]
                    == "could not determine a constructor for the tag '!!python/tuple'"):
                    with open(path, "r") as ymlfile:
                        
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                    return cfg
                else:
                    raise

    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
                                )
    


def write_config(configname, cfg):
    """
    Write structured config file.
    """
#     write_plainconfig(configname, cfg)
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        ruamelFile.dump(cfg_file, cf)

def edit_config(configname, edits):
    """
    Convenience function to edit and save a config file from a dictionary.
    Parameters
    ----------
    configname : string
        String containing the full path of the config file in the project.
    """
    cfg = read_plainconfig(configname)
    
    for key, value in edits.items():
        cfg[key] = value
    write_plainconfig(configname, cfg)
    
    return cfg

def read_plainconfig(configname):
    
    if not os.path.exists(configname):
        raise FileNotFoundError(
            f"Config {configname} is not found. Please make sure that the file exists."
        )
    with open(configname) as file:
        return YAML().load(file)

def write_plainconfig(configname, cfg):
    with open(configname, "w") as file:
        YAML().dump(cfg, file)
        
def set_config_file(path, n_exp, rewrite_existing = False): 

    if n_exp == '1': 
        cfg = set_config_file_locomotion(path)
        
    elif n_exp == '2':
        cfg = set_config_file_openfield(path)
        
    else:
        raise ValueError ('Invalid input. Try again. You must input 1 or 2.')
        
    configpath = os.path.join(path, os.path.basename(os.path.normpath(path)) + '_config.yaml')
    
    if not os.path.exists(configpath) or rewrite_existing:
        write_config(configpath, cfg)
        
    return configpath

def set_config_file_locomotion(path):
    
    cfg = {
        'area_cal_method': 'pix_count', # str, {'pix_count': counts nb of laser pixels, 'contour': creates contour around laser (suitable for HSV)}, 
        'experiment': 'Locomotion treadmil',# experiment paradigm 
        'project_path': path,
        'fps' : 250,# frame per second of recorded video
        'treadmil_length_in_cm' : 37.6,# float, Full treadmill length in cm
        'stim_duration_dict' : { 'beta': 500, 'square': 500, 'beta-square': 500} , # dictionary, corresponding stim durations (in ms) for different pulses
        'stim_duration' : None, # stim duration time, not necessary if stim type is specified in name and the above dict is specified.
        'duration_off_per_cycle' : 25,
        'extract_info_from_file' : True,
        'thresholding_method' : 'RGB',# str, {'RGB', 'HSV'}
        'pix_thresh_upper_bound' : {'RGB': (255, 120, 140),
                                  'HSV': (150, 255, 255)}, # dictionary , {'HSV': (H, S, V), 
                                                                          # 'RGB' : (R, G, B)}
        'pix_thresh_lower_bound' :  {'RGB': (150, 10,60),
                                  'HSV': (50, 80, 60)},# dictionary , {'HSV': (H, S, V), 
                                                                     # 'RGB' : (R, G, B)}

        'use_laser_detection_if_no_analogpulse' : True,# bool
        'enforce_use_laser_detection_only' : False,# bool, True if you don't want to use the analogpulse
        'reanalyze_existing': False,# bool, True if you wish to reanalyze already analyzed videos 
        'crude_smooth_wind' : 100,# gassian kernel window for crudly smoothing 
        'center_vicinity_h_thresh': 0.3,# peak threshold value for crudly smoothed signal 
        'gauss_window' : 10,# gaussina kernel window for smoothing signal before bandpass filtering
        'bandpass_frequency' : (1, 50),# (low_f, high_f) tuple specifying the freq band to filter for detecting start end of pulse
        'filter_order' : 6,# bandpass filter order
        'start_end_h_thresh': 0.3,# peak threshold value for start end event detection using above specified bandpass filtered sig
        'max_iteration' : 20, 
        'constrain_frame' : True,# bool, whethers or not constrain searched pixels with DLC analysis
        'DLC_label' : 'Nose',# str, DLC label to use to constrain pixels
        'DLC_p_cutoff_ranges': 0.995,# float, cutoff label detection liklihood to use to constrain the vertical wherabout of laser
        'max_dev_in_cm' : 1.7# float, maximum deviation in cm around the provided label to constrain pixels.
        
        }
    return cfg

def set_config_file_openfield(path):
    
    cfg = {
        'area_cal_method': 'pix_count', # str, {'pix_count': counts nb of laser pixels, 'contour': creates contour around laser (suitable for HSV)}, 
        'experiment': 'Open field',# experiment paradigm 
        'project_path': path,
        'fps' : 60, # frame per second of recorded video
        'stim_duration_dict' : {'beta': 30000, 
                                'square': 30000} , # dictionary, corresponding stim durations (in ms) for different pulses
        'stim_duration' : 30000, # stim duration time, not necessary if stim type is specified in name and the above dict is specified.
        'duration_off_per_cycle' : 40,
        'extract_info_from_file' : False,
        'thresholding_method' : 'RGB',# str, {'RGB', 'HSV'}
        'pix_thresh_upper_bound' : {'RGB': (255, 120, 140),
                                  'HSV': (150, 255, 255)}, # dictionary , {'HSV': (H, S, V), 
                                                                          # 'RGB' : (R, G, B)}
        'pix_thresh_lower_bound' :  {'RGB': (150, 10,60),
                                  'HSV': (50, 80, 60)},# dictionary , {'HSV': (H, S, V), 
                                                                     # 'RGB' : (R, G, B)}

        'use_laser_detection_if_no_analogpulse' : True,# bool
        'enforce_use_laser_detection_only' : True,# bool, True if you don't want to use the analogpulse
        'reanalyze_existing': False,# bool, True if you wish to reanalyze already analyzed videos 
        'crude_smooth_wind' : 100,# gassian kernel window for crudly smoothing 
        'center_vicinity_h_thresh': 0.3,# peak threshold value for crudly smoothed signal 
        'gauss_window' : 10,# gaussina kernel window for smoothing signal before bandpass filtering
        'bandpass_frequency' : (1, 50),# (low_f, high_f) tuple specifying the freq band to filter for detecting start end of pulse
        'filter_order' : 6,# bandpass filter order
        'start_end_h_thresh': 0.3,# peak threshold value for start end event detection using above specified bandpass filtered sig
        'max_iteration' : 20, 
        'constrain_frame' : False,# bool, whether or not constrain searched pixels with DLC analysis
        'DLC_label' : 'Nose',# str, DLC label to use to constrain pixels
        'DLC_p_cutoff_ranges': 0.995,# float, cutoff label detection liklihood to use to constrain the vertical wherabout of laser
        'max_dev_in_cm' : 1.7# float, maximum deviation in cm around the provided label to constrain pixels.
        
        }
    return cfg

class MouseLocation:
    

    def __init__( self, DLC_filepath, pos_in_vid = 'upper', p_cutoff = 0.995, 
                 treadmil_length_in_cm = 33, treadmil_length_in_pix = 1000,
                 max_dev_in_cm = 2, body_part = 'Nose'):
        
        
        self.DLC_filepath = DLC_filepath
        self.x = None
        self.y = None
        self.p = None
        self.side = None
        self.x_range = None
        self.y_range = None
        self.pos_in_vid = pos_in_vid
        self._set_side_from_pos(pos_in_vid)
        self.get_location(body_part = body_part)
        self.max_dist_nose_to_laser = self.cal_max_deviat_nose_to_laser(treadmil_length_in_cm, 
                                                                        treadmil_length_in_pix, 
                                                                        max_dev_in_cm = max_dev_in_cm)
        self.get_cor_range( p_cutoff = p_cutoff )
        self.shift_rel_to_analogpulse_sd = None
        
    def cal_max_deviat_nose_to_laser(self, 
                                     treadmil_length_in_cm, 
                                     treadmil_length_in_pix, 
                                     max_dev_in_cm = 2):
        
        return max_dev_in_cm * treadmil_length_in_pix / treadmil_length_in_cm

        
        
    def _set_side_from_pos (self, pos_in_vid):
        
        if pos_in_vid == 'upper':
            
            self.side = 'r'
            
        else:
            
            self.side = 'l'
            
    def read_DLC(self):

        '''Read DeepLabCut Data.'''

        df = pd.read_csv( self.DLC_filepath, header=[1,2], skiprows=0)
    
        return df
    
    def get_location( self, body_part = 'Nose'):
        
        df = self. read_DLC()
        col = self.side + body_part
        
        self.x = df[ ( col, 'x') ]
        self.y = df[ ( col, 'y') ]
        self.p = df[ ( col, 'likelihood') ]

    def get_cor_range(self, p_cutoff = 0.8):
        
        high_likelohood_ind = self. p > p_cutoff
        
        self.x_range = np.array([ 
                            max( np.min( self.x[ high_likelohood_ind ]) -self.max_dist_nose_to_laser , 0),
                            np.max( self.x[ high_likelohood_ind ])]).astype(int)
        
        self.y_range = np.array([ max(np.min( self.y[ high_likelohood_ind ]) - self.max_dist_nose_to_laser, 0),
                                 np.max( self.y[ high_likelohood_ind ]) - self.max_dist_nose_to_laser /5 ]).astype(int)  
        
        print( ' x range = ', self.x_range, 
               ' y range = ', self.y_range)
        
    def plot_coordinate(self, ax = None, cor = 'x', p_cutoff = 0):
        
        if cor == 'x' : 
            
            param = self.x; 
            cor_range = self.x_range
            
        elif cor == 'y' : 
            
            param = self.y
            cor_range = self.y_range
            
        ax = ax or plt.subplots() [1]
        ax.plot(param [self.p > p_cutoff])
        ax.set_xlabel('time (frame)', fontsize = 15)
        ax.set_ylabel( cor + ' (pix)', fontsize = 15)
        ax.fill_between( np.arange( len(param)), *cor_range, alpha = 0.1)
        
        return ax
    
    def plot_x(self, ax = None,p_cutoff = 0):
        
        ax = self.plot_coordinate(cor = 'x', ax = ax,  p_cutoff = p_cutoff)

        return ax
    
    def plot_y(self, ax = None, p_cutoff = 0):
        
        ax = self.plot_coordinate(cor = 'y', ax = ax ,p_cutoff = p_cutoff)
        
        return ax
    
    def plot_likelihood(self, ax = None):
        
        ax = ax or plt.subplots() [1]
        ax.plot( self.p )
        ax.set_xlabel('time (frame)', fontsize = 15)
        ax.set_ylabel(' tracking likelihood', fontsize = 15)
     
class VideoStream:
    
    """
    restrict frame grabbing to a thread and then queue a stack of frames
    """
    
    def __init__(self, filepath, queueSize = 300, nb_frames = None):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        
        (grabbed, frame) = cv2.VideoCapture(filepath).read()
        self.treadmil_length_in_pix = frame.shape[1]  
        
        self.streaming_video = cv2.VideoCapture(filepath)
        self.stopped = False
        self.nb_frames = nb_frames or self.get_nb_frames()
        self.current_frame_no = 0
        
        print('number of frames = {}'.format(self.nb_frames))

        self.Q = Queue(maxsize=queueSize) # initialize the queue used to store frames read from the video file

        
    def get_current_frame_no(self) :
        
        self.current_frame_no = int ( self.streaming_video.get(1) )
        return self.current_frame_no
        
    def get_nb_frames(self) :
        
        return int( self.streaming_video.get(7) )
        
    def start(self) :
        
        ''' start a thread to read frames from the file video stream
        '''
        
        t = Thread( target = self.create_frame_queue )
        t.daemon = True
        # self.streaming_video.set(1,nframe) ## ?
        t.start()
        
        return self
    
    def create_frame_queue(self):
        
        ''' Add frames to the queue until the queue is stopped or full
        '''
        
        while self.current_frame_no < self.nb_frames :
			
            if self.stopped: # if the thread indicator variable is set, stop the thread
            
                return
            
			
            elif not self.Q.full(): # otherwise, ensure the queue has room in it
            
				
                (grabbed, frame) = self.streaming_video.read() # read the next frame from the file
                
                self.current_frame_no = self.get_current_frame_no()
                
                if not grabbed: # if the `grabbed` boolean is `False`, then we have
                				# reached the end of the video file
                    self.stop()
                    
                    return
            
                self.Q.put(frame) # add the frame to the queue

        
        
    def read(self):
        
        ''' return the next frame in the queue
        '''
        return self.Q.get()
    
    def more(self):
        
        ''' return True if there are still frames in the queue 
        '''
        return self.Q.qsize() > 0
    
    def stop(self):
        
        ''' indicate that the thread should be stopped
        '''
        self.stopped = True     
    
    
class LaserDetector :
    
    def __init__ (self, 
                  video_filepath, 
                  DLC_filepath,
                  thresh_method = 'RGB',
                  area_cal_method = 'pix_count',
                  image_parts = ['upper', 'lower'],
                  treadmil_length_in_cm = 33):
        
        self.video_filepath = video_filepath
        self.DLC_filepath = DLC_filepath
        self.thresh_method = thresh_method
        self.area_cal_method = area_cal_method
        self.image_parts = image_parts
        self.treadmil_length_in_cm = treadmil_length_in_cm 
        
    def set_laser_boundaries(self, 
                             p_cutoff = 0.995, 
                             treadmil_length_in_pix = 1000,
                             max_dev_in_cm = 2, 
                             plot_bounds = False,
                             body_part = 'Nose'):
        
        mloc_lower = MouseLocation( self.DLC_filepath , 'lower', p_cutoff = p_cutoff,
                                    treadmil_length_in_cm = self.treadmil_length_in_cm, 
                                    treadmil_length_in_pix = treadmil_length_in_pix,
                                    max_dev_in_cm = max_dev_in_cm,
                                    body_part = body_part)
        
        mloc_upper = MouseLocation( self.DLC_filepath , 'upper', p_cutoff = p_cutoff, 
                                    treadmil_length_in_cm = self.treadmil_length_in_cm, 
                                    treadmil_length_in_pix = treadmil_length_in_pix,
                                    max_dev_in_cm = max_dev_in_cm,
                                    body_part = body_part)
        
        if plot_bounds:
            
            ax = mloc_lower.plot_y(p_cutoff = p_cutoff)
            ax = mloc_upper.plot_y(p_cutoff = p_cutoff, ax = ax)
            
        x_range = { mloc.pos_in_vid : 
                   mloc.x_range for mloc in [mloc_lower, mloc_upper] }
        
        y_range = { mloc.pos_in_vid : 
                   mloc.y_range for mloc in [mloc_lower, mloc_upper] }

        return x_range, y_range 
    
    
    
    def detect(self, 
                  low_img_thresh = (150, 10,60), 
                  high_img_thresh = (255, 120, 140),
                  nb_frames = None,
                  DLC_p_cutoff_ranges = 0.995,
                  constrain_frame = True,
                  max_dev_in_cm = 2,
                  DLC_body_label = 'Nose'):
        
        start = timeit.default_timer()
        
        vidstr = VideoStream( self.video_filepath , nb_frames = nb_frames). start()
        
        
        if constrain_frame:
            x_range, y_range = self.set_laser_boundaries( p_cutoff = DLC_p_cutoff_ranges, 
                                                         treadmil_length_in_pix = vidstr.treadmil_length_in_pix,
                                                         max_dev_in_cm = max_dev_in_cm,
                                                         body_part = DLC_body_label)
        else:
            x_range, y_range = None, None

        laser = Laser ( low_img_thresh, high_img_thresh )
                    
        laser.detect(vidstr, 
                     nb_frames = nb_frames,
                     x_range = x_range, 
                     y_range = y_range,
                     area_cal_method = self.area_cal_method,
                     thresh_method = self.thresh_method,
                     image_parts = self.image_parts)
        
        stop = timeit.default_timer()
        try:
            print('runtime =', time.strftime("%H:%M:%S", time.gmtime(round(stop - start, 2)))) 
        except:
            pass
        return laser.area_list
    
    def all_video(self):
        
        pass
  
    @staticmethod
    def pickle_obj(obj, filepath):
    
        with open(filepath, "wb") as file_:
            pickle.dump(obj, file_)  # , -1)
        
    @staticmethod
    def load_pickle(filepath):
    
        return pickle.load(open(filepath, "rb"))  # , -1)
    
class Frame :
    
    """class for frames
    """
    
    def __init__(self, vidstr,  x_range = None, y_range = None) :
        
        self.image = vidstr.read()
        self.no = None # vidstr.get_current_frame_no()
        self.height, self.width = self.image.shape[:2]
        self.x_range = x_range
        self.y_range = y_range
        
        self.constrain_frame = self.whether_constrain_frame()
        self.detection_thresh_methods = {
                                        'HSV' : self.mask_hsv, 
                                        'RGB' : self.mask_rgb}
        
        self.detection_area_cal_methods = {
                                        'contour' : self.cal_contour_areas,
                                        'pix_count': self.count_threshold_passed_pix}
                                    
        
    def whether_constrain_frame(self):
        
        if self.x_range == None:
            
            return False
            
        else:
            
            return True

        
    def mask_hsv(self, low_img_thresh, high_img_thresh):
        
        self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        masked = cv2.inRange(hsv, low_img_thresh, high_img_thresh)
    
        return masked
    
    def mask_rgb (self,low_img_thresh, high_img_thresh):
        
        image = cv2.GaussianBlur(self.image, (3, 3), 0)
        masked = cv2.inRange(image, low_img_thresh, high_img_thresh)

        return masked
    
    def cal_contour_areas(self, masked, image_parts = ['upper', 'lower']):
        
        contours, _ = cv2.findContours(masked.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        laser_area = 0
        
        if len(contours) > 1 :
            
            contour_max = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            laser_area = cv2.contourArea(contour_max) 
        
        return laser_area
    
    def count_threshold_passed_pix(self, masked, image_parts = ['upper', 'lower']):
        
        if self.constrain_frame:
            
            laser_area = 0
            
            for part in image_parts:
                
                laser_area += np.count_nonzero(
                            masked[ self.y_range[part][0] :  self.y_range[part][1],
                                    self.x_range[part][0] :  self.x_range[part][1]]
                                        )
            
            
        else:
            
            laser_area = np.count_nonzero(masked)

        return laser_area

    def detect_laser(self,  
                     low_img_thresh, 
                     high_img_thresh, 
                     thresh_method = 'rgb',
                     area_cal_method = 'pix_count',
                     image_parts = ['upper', 'lower']):
        
        masked = self.detection_thresh_methods[ thresh_method] (low_img_thresh, high_img_thresh)
        laser_area = self.detection_area_cal_methods[ area_cal_method] (masked, image_parts = image_parts)

        return masked, laser_area
    
class Laser :
    
    def __init__(self, 
                 low_img_thresh : tuple, 
                 high_img_thresh : tuple) :
         
        self.area_list = []
        self.low_img_thresh = low_img_thresh
        self.high_img_thresh = high_img_thresh
        

        
    def plot_mask(self, mask, frame_no, image, contours = None):
        
        while True:
    	
            cv2.imshow('frame # {}'.format(frame_no), image) # display the image and wait for a keypress
            cv2.imshow('mask', mask)
            # cv2.drawContours(image, contours, -1, (0,0, 255), 3)

            if cv2.waitKey(1) == 27 :
                break
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        
    def detect(self, vidstr, 
               nb_frames = None, 
               x_range = None, 
               y_range = None,
               thresh_method = 'rgb',
               area_cal_method = 'pix_count',
               image_parts = ['upper', 'lower']):
        
        nb_frames = nb_frames or vidstr.nb_frames 
        self.area_list = np.zeros(nb_frames + 1)
        
        n = 0
        while n < nb_frames:
            
            
            frame = Frame (vidstr, x_range = x_range, y_range = y_range)
            
            n += 1
            frame.no = n
            
            # print(frame.no)
            if int(frame.no) % 1000 == 0:
                print('frame :', frame.no)
                
            
            masked, self.area_list [frame.no - 1] = frame.detect_laser( self.low_img_thresh, 
                                                                        self.high_img_thresh, 
                                                                        thresh_method = thresh_method,
                                                                        area_cal_method = area_cal_method,
                                                                        image_parts = image_parts )
            

            
            # if frame.no == 14918 :
                
            #     self. plot_mask(masked, frame.no, frame.image)
           
    @staticmethod                
    def moving_average_array(X, n):
        
        '''Return the moving average over X with window n without changing dimesions of X'''
        z2= np.cumsum(np.pad(X, (n,0), 'constant', constant_values=0))
        
        z1 = np.cumsum(np.pad(X, (0,n), 'constant', constant_values=X[-1]))
        return (z1-z2)[(n-1):-1]/n
                    
            
        
class Pulse :
    
    def __init__( self, sig, fs = 250, low_f = 1, high_f = 50, 
                 enforce_use_laser_detection_only = False, 
                 use_laser_detection_if_no_analogpulse = False,
                 true_duration = None):
        
        self.signal = sig
        self.raw_signal = sig.copy()
        self.low_f = low_f
        self.high_f = high_f
        self.fs = fs
        self.events = []
        self.centers = []
        self.starts = []
        self.ends = []
        self.smoothed_sig = []
        self.center_vicinities = []
        self.shift_rel_to_analogpulse = None
        self.method = None
        self.missing_start_end = [] # indices of lasers that the center vicinity is detected but either of start end stamps are not
        self.output_centers = None
        self.center_vicinity_h_thresh = None
        self.enforce_use_laser_detection_only = enforce_use_laser_detection_only
        self.use_laser_detection_if_no_analogpulse = use_laser_detection_if_no_analogpulse
        self.true_duration = true_duration
        
    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        
        return sos
    
    @staticmethod
    def butter_bandpass_filter(sig, lowcut, highcut, fs, order=5):
        
        ''' filtfilt is the forward-backward filter. It applies the filter twice, 
            once forward and once backward, resulting in zero phase delay.
        '''
        sos = Pulse.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, sig)
        return y
    
    def gauss_filter(self, gauss_window = 5):
        
        self.signal = gaussian_filter1d(self.signal, gauss_window)
        
    def low_pass_filter(self, low_f=1, high_f =50, fs = 250, filt_order= 10):
        
        self.signal = Pulse.butter_bandpass_filter(self.signal, low_f, high_f , fs, order= filt_order)
        
    def find_peaks(self, height = 40):
        
        ''' signal should be low pass filtered so that the peaks are negative values'''
        
        self.events, _ = find_peaks( -self.signal, height = height)
    
    
    def find_centers(self):
        
        self.centers = np.average( np.column_stack( (self.starts, 
                                                     self.ends)) , 
                                  axis = 1)
    def find_events(self, gauss_window = 5, 
                           low_f = 1, high_f = 50, 
                           filt_order = 10,
                           peak_heights = 0.3):
        """
        Filter laser detection signal between the low_f and high_f
        and then find the events (peaks in filtered sig) based on the peak heights threshold

        Parameters
        ----------
        gauss_window : int, optional
            window for gaussian kernel. The default is 5.
        low_f : float, optional
            low bound for frequency find_peaks. The default is 1.
        high_f : float, optional
            upper bound for frequency find_peaks. The default is 50.
        filt_order : int, optional
            filter order. The default is 10.
        peak_heights : float, optional
            sig peak height threshold. The default is 0.3.

        Returns
        -------
        None.

        """
        self.pre_process( gauss_window = gauss_window, 
                        low_f = low_f, high_f = high_f, 
                        filt_order = filt_order)
        
        self.find_peaks(height = peak_heights)
        
    def smooth_sig(self, crude_smooth_wind):
        
        self.smoothed_sig = gaussian_filter1d( self.raw_signal.copy(), crude_smooth_wind)

    def find_center_vicinities(self, center_vicinity_h_thresh = 0.3, crude_smooth_wind = 100):
        """
        Find approximate time for the pulse center by detectinf the peaks of
        crudly smoothed laser detection signal.

        Parameters
        ----------
        center_vicinity_h_thresh : float, optional
            peak threshold for detecting the pulse center. The default is 0.3
        crude_smooth_wind : int, optional
            Gaussian kernel smoothing window to have a crude laser detection envelope. The default is 100.

        Returns
        -------
        None.

        """
        self.smooth_sig(crude_smooth_wind)
        self.center_vicinities,_ = find_peaks(self.normalize(self.smoothed_sig), 
                                              height = center_vicinity_h_thresh)

    def determine_start_ends(self):

        derivative_at_events = np.diff( self.smoothed_sig ) [self.events]
        self.starts = self.events[ derivative_at_events > 0]
        self.ends = self.events[ derivative_at_events < 0]
        

        

    def thresh(self, laser_area_thresh = 180):
        
        return np.where(self.signal > laser_area_thresh) [0]
    
    def detect_start_by_thresh (self, ind_thr_pass, jump_by = 5):
        
        signal_shift_fwd = Pulse.shift( self.signal, 1, 0)
        ind_start_jump = np.where( self.signal  > jump_by * signal_shift_fwd ) [0]
        
        self.laser_starts = np.intersect1d (ind_thr_pass, ind_start_jump) - 1
        
    def detect_ends_by_thresh (self, ind_thr_pass, jump_by = 5):
        
        signal_shift_bck = Pulse.shift( self.signal, -1, 0)
        ind_end_jump = np.where( self.signal  > jump_by * signal_shift_bck ) [0]
        
        self.laser_ends = np.intersect1d (ind_thr_pass, ind_end_jump) + 1
        
    def detect_laser_start_end(self,  laser_area_thresh = 180, jump_by = 5):
        
        ind_thr_pass = self.thresh( laser_area_thresh = laser_area_thresh )
        
        self.detect_start_by_thresh ( ind_thr_pass, jump_by = jump_by)
        self.detect_ends_by_thresh (ind_thr_pass, jump_by = jump_by)
        
    def pre_process(self,
                    gauss_window = 5, 
                    low_f = 1, high_f = 50, 
                    filt_order = 10,
                    peak_heights = 40):
        
        self.gauss_filter(gauss_window = gauss_window)
        self.low_pass_filter(low_f = low_f, high_f = high_f, fs = self.fs, filt_order = filt_order)
        self.normalize_filtered_sig()
        
    def cut_sig(self, ind_cut):
        
        self.signal = self.signal[:ind_cut]
        self.raw_signal = self.signal[:ind_cut].copy()

    
    def remove_problematic_detections(self):
        
        ''' remove detected laser start and ends where there is not 
        only and only one detected for each per peak
        '''
        
        between_pulses = np.average( np.column_stack( (self.center_vicinities, 
                                                     np.roll(self.center_vicinities, -1))),
                                    axis = 1) [:-1]
        self.missing_start_end = np.zeros_like(self.center_vicinities, dtype = bool)
        
        pulse_boundaries = np.zeros(len(between_pulses) + 2)
        pulse_boundaries[1 : -1] = between_pulses
        pulse_boundaries[-1] = len(self.raw_signal)
        
        n_to_remove = 0
        
        for i, (left, right) in enumerate(zip(pulse_boundaries, pulse_boundaries[1:])):
            
            # plt.axvline(x = left, ls ='--', c = 'k')
            this_starts = np.logical_and ( self.starts > left, self.starts < right)
            this_ends = np.logical_and ( self.ends > left, self.ends < right)
            
            if sum(this_starts ) !=  1 or sum( this_ends ) != 1 :
                
                self.starts = np.delete( self.starts, this_starts)
                self.ends = np.delete( self.ends, this_ends)
                self.missing_start_end[i] = True
                n_to_remove += 1 
                
        print(' {} pulses flagged as problematic'.format( n_to_remove) )
        
    def use_laser_detection_only(self, analogpulse):
        
        base_on_laser_only = False
        run_values, _, run_lengths = Pulse.find_runs(self.missing_start_end)
        
        if not self.enforce_use_laser_detection_only:
            
            if len(self.centers_aligned) == len (analogpulse.centers) \
                and (len(self.center_vicinities) == len( analogpulse.centers ) # or if single pulses are missed
                      and np.sum(run_lengths[run_values] > 1) == 0 ): # if all pulses detected
    
                self.method = 'smr file with all-pulse alignement'            
                self.centers = self.centers_aligned
                self.cal_shift_rel_to_analogpulse_all_pulses( analogpulse )
                        
    
            elif (np.sum(run_lengths[run_values] > 1) > 0 
                and not self.check_first_missing):  # consequtive pulses missed in laser detection
                                                                                            # but first laser detected
    
                self.method = 'smr file with first-pulse alignement'                                      
                self.cal_shift_rel_to_analogpulse_first_pulse_only( analogpulse)
            
            else: # all methods fail, only include the definitive laser detections instead of alignment
                
                self.method = 'laser detections only'
                self.cal_exact_start_ends(analogpulse.true_duration)
                base_on_laser_only = True
                self.shift_rel_to_analogpulse = np.nan
                self.shift_rel_to_analogpulse_sd = np.nan
                
        else:
            
            self.method = 'laser detections only'
            self.cal_exact_start_ends(self.true_duration)
            base_on_laser_only = True
            self.shift_rel_to_analogpulse = np.nan
            self.shift_rel_to_analogpulse_sd = np.nan
            
        print('Method:', self.method)
        return base_on_laser_only
            
    def cal_exact_start_ends(self, true_duration):
        
        print('analogpulse laser durations:', true_duration)
        
        self.starts = (self.centers - true_duration / 2 ).astype(int)
        self.ends = (self.centers + true_duration / 2 ).astype(int)
        
    def cal_shift_rel_to_analogpulse_first_pulse_only(self, analogpulse):
        
        '''  calculate shift from the first pulse only '''
        
        self.shift_rel_to_analogpulse = self.centers[0] - analogpulse.centers [0]
        print('shift between laser and analogpulse: \n', 
              int(self.shift_rel_to_analogpulse))
        
    def cal_shift_rel_to_analogpulse_all_pulses(self, analogpulse):
        
            
        all_shifts = self.centers_aligned - analogpulse.centers
        all_shifts_without_misses = all_shifts[ self.centers_aligned != 0]
        self.shift_rel_to_analogpulse = int( np.average( all_shifts_without_misses ) )
        self.shift_rel_to_analogpulse_sd = round(np.std( all_shifts_without_misses ))
        
        print('shift between laser and analogpulse: \n', 
              int(self.shift_rel_to_analogpulse), u"\u00B1",
              self.shift_rel_to_analogpulse_sd,
              ' frames')

    def verify_nb_with_analog(self, 
                              video_filename, path, 
                              analogpulse,
                              center_vicinity_h_thresh = 0.3,
                              change_coef = 0.2,
                              max_iteration = 20,
                              crude_smooth_wind = 100):
        
        self.enf_if_no_analpulse(analogpulse)        

        if not self.enforce_use_laser_detection_only:
            
            self.find_center_through_recursion(center_vicinity_h_thresh, 
                                               video_filename, path,
                                               analogpulse,
                                               change_coef = change_coef,
                                               max_iteration = max_iteration,
                                               crude_smooth_wind = crude_smooth_wind)
                
            
        else:
            
            self.find_center_vicinities(center_vicinity_h_thresh = h_thresh, crude_smooth_wind = crude_smooth_wind)
            
    def find_center_through_recursion(self, 
                                      center_vicinity_h_thresh, 
                                      video_filename, path,
                                      analogpulse,
                                      change_coef = 0.2,
                                      max_iteration = 20,
                                      crude_smooth_wind = 100):
        """
        In order to get all the center vicinities when knowing the actual number of pulses through the
        analog signal, we recursively change the peak detection threshold of the center vicinites till 
        we get it right.

        Parameters
        ----------
        center_vicinity_h_thresh : float
            starting peak threshold value.
        video_filename : str
            filename.
        path : str
            poth.
        analogpulse : AnalogPulse
            analogopulse class instance.
        change_coef : float, optional
            coefficient by which the threshold is scaled in each iteration. The default is 0.2.
        max_iteration : int, optional
            maximum number of iterations to reset the threshold. The default is 20.
        crude_smooth_wind : int, optional
            the gaussian kernel window to smooth the laser detection signal with . The default is 100.

        Returns
        -------
        None.

        """
        it = 0
        raise_err = True
        h_thresh_1_before = center_vicinity_h_thresh
        h_thresh_2_before = 1
        message = 'zero iterations for threshold adjustment'
        while it < max_iteration:
            
            self.find_center_vicinities(center_vicinity_h_thresh = center_vicinity_h_thresh, crude_smooth_wind = crude_smooth_wind)

            
            if len(self.center_vicinities) > len(analogpulse.centers):
                
                message = "Extra pulses are detected"
                center_vicinity_h_thresh = h_thresh_1_before + change_coef * abs(h_thresh_2_before - h_thresh_1_before)

            elif len(self.center_vicinities) < len(analogpulse.centers):
                
                message = "Not all pulses are detected"
                 
                center_vicinity_h_thresh = h_thresh_1_before - change_coef * abs(h_thresh_1_before - h_thresh_2_before)


            else:
                
                raise_err = False
                print('working threshold = {} found in {} iterations'.format(center_vicinity_h_thresh, it))
                print(os.path.join(path, 'JPEG', 'Problematic'),
                      video_filename + '_unsuccessful.jpg')
                File.rm_if_exist(os.path.join(path, 'JPEG', 'Problematic'), video_filename + '_unsuccessful.jpg')
                
                break
            
            print(message, 'new threshold = ', center_vicinity_h_thresh)

            h_thresh_2_before = h_thresh_1_before
            h_thresh_1_before = center_vicinity_h_thresh
            it += 1
        
        if raise_err:
            self.find_center_vicinities(center_vicinity_h_thresh, crude_smooth_wind = crude_smooth_wind)
            self. _raise_err_out_plot(analogpulse, message, video_filename, path, center_vicinity_h_thresh)
            
        return 
    
    # def find_center_without_recursion(self,h_thresh, 
    #                                  video_filename, path,
    #                                  analogpulse,
    #                                  change_coef = 0.2,
    #                                  max_iteration = 20,
    #                                  crude_smooth_wind = 100)
    
    #     self.find_center_vicinities(h_thresh, crude_smooth_wind = crude_smooth_wind)
    
        
    #     if len(self.center_vicinities) > len(analogpulse.centers):
            
    #         message = "Extra pulses are detected"
    #         h_thresh = h_thresh_1_before + change_coef * abs(h_thresh_2_before - h_thresh_1_before)
    
    #     elif len(self.center_vicinities) < len(analogpulse.centers):
            
    #         message = "Not all pulses are detected"
             
    #         h_thresh = h_thresh_1_before - change_coef * abs(h_thresh_1_before - h_thresh_2_before)
    
    
    #     else:
            
    #         raise_err = False
    #         print('working threshold = {} found in {} iterations'.format(h_thresh, it))
    #         print(os.path.join(path, 'JPEG', 'Problematic'),
    #               video_filename + '_unsuccessful.jpg')
    #         File.rm_if_exist(os.path.join(path, 'JPEG', 'Problematic'), video_filename + '_unsuccessful.jpg')
            
                
    def enf_if_no_analpulse(self, analogpulse):
        
        if analogpulse.value == None:
            
            if  not self.use_laser_detection_if_no_analogpulse:
                
                raise ValueError("Analogpulse file does not exist. \n \
                                 Either provide, or set 'use_laser_detection_if_no_analogpulse' to True.")
            else:
                
                self.enforce_use_laser_detection_only = True
                
    def _raise_err_out_plot(self, analogpulse, message, video_filename, path, h_thresh):
        
        ax = self.plot_start_ends()
        ax.plot(analogpulse.centers, np.ones_like(analogpulse.centers), 'o', c = 'k', label = 'analogpulse centers')
        ax.axhline(y = h_thresh, 
                   xmin = 0, xmax = len(self.raw_signal), 
                   ls = '--', c = 'k', label = 'peak thresh')
        ax.legend(frameon = False, loc = 'lower right', fontsize = 12, bbox_to_anchor=(1.25, 0.15))
        ax.set_title(message, fontsize = 12)
        fig = ax.get_figure()
        fig.set_size_inches((10,4), forward=False)
        path_to_save = os.path.join(path, 'JPEG', 'Problematic', message.replace(' ', '_'))
        Directory.create_dir_if_not_exist(path_to_save)
        fig.savefig(os.path.join(path_to_save, video_filename + '_unsuccessful.jpg'), 
                    dpi = 250, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        
        raise ValueError ("Did not manage to analyze.")
        
    def fill_missing_pulses(self, analogpulse, plot = False, report = False):
            
        n_missing = len( analogpulse.centers) - len( self.centers) # number of missing laser detections
        
        if n_missing > 0 :

            # second method counting on the fact that all pulse center vicinites are detected
            self.centers_aligned = np.zeros_like(analogpulse.centers)
            self.centers_aligned[self.missing_start_end] = 0
            self.centers_aligned[~ self.missing_start_end] = self.centers
            print(' {} pulses missing in video detection'.format(n_missing))
            
        else:
            
            self.centers_aligned = self.centers.copy()
        
    def find_duration(self, analogpulse):
        
        if not self.enforce_use_laser_detection_only:
            
            self.fill_missing_pulses(analogpulse, plot = False, report = False)
            true_duration = analogpulse.true_duration
            
        else:
            
            true_duration = self.true_duration
            
        return true_duration
    
    def check_first_missing(self):
        
        if self.centers[0] - self.center_vicinities[0] > 100 :
            
            return True
        
        else:
            
            return False
        
    def compare_methods(self, gauss_window = 5, 
                        low_f = 1, high_f = 50, 
                        filt_order = 10,
                        peak_heights = 40):
        
        sig_copy = self.signal.copy()
        ax = self.plot( signal = self.raw_signal, label = 'true signal', ax = None )
        
        # self.low_pass_filter(low_f = low_f, high_f = high_f, fs = self.fs, filt_order = filt_order)
        # ax = self.plot(label = 'low pass filtered signal', ax = ax)
        # self.signal = sig_copy.copy()

        self.gauss_filter(gauss_window = gauss_window)
        ax = self.plot(label = 'gaussian filtered signal', ax = ax)
        
        self.low_pass_filter(low_f = low_f, high_f = high_f , fs = self.fs, filt_order= filt_order)
        ax = self.plot(label = 'low pass filtered smoothed signal', ax = ax)
        
        self.find_peaks(height = peak_heights)
        ax = self.plot_events(ax = ax)

        self.signal = sig_copy

        return ax
    

    
    def normalize_filtered_sig(self):
        
        """ normalize filtered signal by the maximum deflection value"""
        max_sig = max( abs ( self.signal[ self.signal < 0]) )
        self.signal = self.signal / max_sig

    def normalize(self, sig):
        
        max_sig = max( abs ( sig) )
        return sig / max_sig
        
    
    def plot( self, signal = [], label = 'true signal', ax = None , c = 'darkgrey') :
        
        if len(signal) == 0:
            signal = self.signal
            
        ax = ax or plt.subplots()[1]
        
        ax.plot(signal, '-o', ms = 0.1, label = label, c = c, lw = 1)
        ax.set_ylabel('# blue pixels in frame', fontsize = 15)
        ax.set_xlabel('# frame', fontsize = 15)
        ax.legend(frameon = False, loc = 'lower right', fontsize = 12, bbox_to_anchor=(1.25, 0.15))
        
        return ax
    
    def plot_events(self, ax = None):
        
        ax = ax or plt.subplots()[1]
        
        ax = self.plot(ax = ax, label = 'filtered')
        ax.plot( self.normalize(self.smoothed_sig), label = 'smoothed' , c = 'orange', lw = 1)
        ax.plot( self.events, self.signal [self.events] , 'x', ms = 10, c = 'k', label = 'events')
        ax.legend(frameon = False, loc = 'lower right', fontsize = 12, bbox_to_anchor=(1.25, 0.15))
        
        return ax
    
    def plot_start_ends(self, ax = None):
        
        ax = ax or plt.subplots() [1]
        
        ax = self.plot(ax = ax, label = 'filtered')
        ax = self.plot(signal = self.normalize(self.raw_signal), ax = ax, c = 'teal')

        ax.plot( self.normalize(self.smoothed_sig), label = 'smoothed' , c = 'orange', lw = 1)
        ax.plot( self.center_vicinities, self.normalize(self.smoothed_sig)[self.center_vicinities], 'o', c = 'y', ms = 7,
                label = 'center appx' )
        ax.plot( self.starts, self.signal[ self.starts], 'o', c = 'blue', label = 'det start')
        ax.plot( self.ends, self.signal[ self.ends], 'o', c = 'r', label = 'det end')
        ax.legend(frameon = False, loc = 'lower right', fontsize = 12, bbox_to_anchor=(1.25, 0.15))
        ax.set_ylabel('normalized # blue pixels in frame', fontsize = 15)

        return ax
    
    def plot_centers(self, ax = None):
           
        ax = ax or plt.subplots()[1]

        ax = self.plot_start_ends( ax = ax)
        ax.scatter(self.centers, self.normalize(self.smoothed_sig)[ self.centers.astype(int) ] , s=120, 
                   facecolor = 'none', ec = 'k', lw = 3, label = 'center exact')
        ax.legend(frameon = False, loc = 'lower right', fontsize = 12, bbox_to_anchor=(1.25, 0.15))

        return ax

    def plot_superimposed(self, starts, ends, centers, true_duration):
        
        fig, ax = plt.subplots()
        
        half_duration = int(true_duration / 2)
        x_s = np.linspace(-half_duration - 20, half_duration + 20, num = 2 * half_duration + 40)
        
        for start, end, center in zip(starts, ends, centers):
            
            line, = ax.plot(x_s, self.raw_signal[center - half_duration - 20:
                                    center + half_duration + 20], 
                    '-o', ms = 3, lw = 0.5)
            ax.axvline(x = (start-center), ls = '--', c = line.get_color())
            ax.axvline(x = (end - center), ls = '--', c = line.get_color())
            
        ax.axvline(x = 0, ls = '--', c = 'k')
        ax.set_ylabel('laser intensity', fontsize = 12)
        ax.set_xlabel('# frame', fontsize = 12)
        ax.set_title(self.method, fontsize = 12)

        return ax
    
    @staticmethod
    def find_runs(x):
        """Find runs of consecutive items in an array."""

        # ensure array
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError('only 1D array supported')
        n = x.shape[0]

        # handle empty array
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        else:
            # find run starts
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]

            # find run values
            run_values = x[loc_run_start]

            # find run lengths
            run_lengths = np.diff(np.append(run_starts, n))

            return run_values, run_starts, run_lengths

    @staticmethod
    def shift(arr, num, fill_value=np.nan):
        
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
            
        return result
    
    @staticmethod
    def save_figs(ax, ax_superimp, path_list, figname, size = (10,4)):
        
        fig = ax.get_figure() ; fig_superimp = ax_superimp.get_figure()
        fig.set_size_inches(size, forward=False)
        fig_superimp.set_size_inches((5,4), forward=False)
        for path in path_list:
            fig.savefig(os.path.join( path, figname + '.pdf'), dpi = 500, facecolor='w', edgecolor='w',
                            orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
            
        Directory.create_dir_if_not_exist( os.path.join( 
                                                path_list[0],
                                                'JPEG'))
        fig.savefig(os.path.join(path_list[0], 'JPEG', figname + '.jpg'), dpi = 250, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
        fig_superimp.savefig(os.path.join(path_list[0], 'JPEG', figname + '_superimp.jpg'), dpi = 250, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)

# class LongPulse(Pulse):
    
#     def __init__( self, sig, fs = 250, low_f = 1, high_f = 50, 
#                  enforce_use_laser_detection_only = False, 
#                  use_laser_detection_if_no_analogpulse = False,
#                  true_duration = None):
        
#         Pulse.__init__(self, sig, fs = fs, low_f = low_f, high_f = high_f, 
#                      enforce_use_laser_detection_only = enforce_use_laser_detection_only, 
#                      use_laser_detection_if_no_analogpulse = use_laser_detection_if_no_analogpulse,
#                      true_duration = true_duration)
        
#     def determine_start_ends(self):
        

class AnalogPulse : 
    
    def __init__(self, file, stim_type):
        
        self.stim_type = stim_type
        
        if isinstance(file, File):
            
            self.starts = []
            self.ends = []
            self.centers = []
            self.true_duration = None
            self.read_data_from_csv(file.path)
            self.cal_centers()
            self.value = 'exists'
            
        else:
            
            self.value = None
        
    def read_data_from_csv(self, filepath):
        
        df = pd.read_csv(filepath, skiprows = 4, header = [0])
        
        self.starts = (df['ON'].values / 4).astype(int)
        self.ends = (df['OFF'].values / 4).astype(int)
        self.cal_true_duration()
                
        
    def cal_centers(self):
        
        self.centers = np.average( np.column_stack( (self.starts, 
                                                     self.ends) ), 
                                  axis = 1).astype(int) 


    def cal_true_duration(self):
        
        self.true_duration = int( np.average( self.ends - self.starts) )
        
    def align_to_video(self, shift_rel_to_analogpulse):
        
        self.starts += shift_rel_to_analogpulse
        self.ends += shift_rel_to_analogpulse
        
    def remove_pulses(self, ind):
        
        self.starts = np.delete(self.starts, ind)
        self.ends = np.delete(self.ends, ind)
        self.centers = np.delete(self.centers, ind)
        self.cal_true_duration()
        
class SortedExpeiment(Experiment) :
    
    def __init__(self, video_filepath, 
                 stim_duration = None,
                 stim_duration_dict = { 'beta': 500, 'square': 500, 'beta-square': 500}, 
                 extract_info_from_file = True,
                 fps = 250,
                 duration_off_per_cycle = 25):
        
        Experiment.__init__(self, video_filepath)
        
        self.files = {'DLC': None, 'analogpulse': None}
        self.stim_duration = stim_duration
        self.check_func = {'DLC': self.right_DLC, 
                           'analogpulse': self.right_analogpulse}
        self.fps = fps
        if extract_info_from_file:
            self.extract_info_from_video_filename()
        
        self.find_stim_duration(fps, stim_duration_dict, duration_off_per_cycle)
        
        
        self.exp_dir = os.path.dirname( self.video.dirpath )
        
        self.find_analogpulse_DLC_csv()
        print('Video: ', self.video.name)

        self.get_DLC_path()
        self.already_analyzed = False
        self.prob_csv_path= None
        self.result_filename = self.get_result_filename()
        
    def get_DLC_path(self):
        
                
        if isinstance(self.files['DLC'], File):
            self.DLC_path = self.files['DLC'].path
        else:
        
            self.DLC_path = None
        
            
    def find_stim_duration(self, fps, stim_duration_dict, duration_off_per_cycle):
        
        """get the single pulse duration in terms of frames. If pulse is beta, the last off period of the cycle is 
        accounted for and removed from the time"""
        

        if self.stim_duration != None:
            self.stim_duration  = int(self.stim_duration / 1000 * fps)

        else:
            beta = 'beta' in self.stim_type.lower()
            square = 'square' in self.stim_type.lower()
            if beta and not square:
                self.stim_duration = int((stim_duration_dict['beta'] - duration_off_per_cycle) / 1000 * fps)
            
            elif square:
                self.stim_duration = int(stim_duration_dict['square'] / 1000 * fps)
                
            else:
                print('stim duration is unknown!')
                self.stim_duration = None
            
    def check_if_already_analyzed(self):
        
        directory = Directory( os.path.join( self.exp_dir, 'Laser'))
        
        if isinstance(self.files['DLC'], str):
            supposed_analysis_filepath = os.path.join(self.exp_dir, 'Laser', 
                                                      self.files['DLC'].name_base +
                                                      '_Laser.csv')
            if '.csv' in directory.filepath_list:
                if supposed_analysis_filepath in directory.filepath_list['.csv']:
                    
                    self.already_analyzed = True
                    print('File already analyzed!')
            
    def right_analogpulse (self, file):
        
        condition_1 = self.day_tag in file.name_base
        condition_2 = '_Laser' not in file.name_base 
    
        return condition_1 and condition_2 
    
    def right_DLC (self, file):

        condition_1 = self.day_tag in file.name_base
        condition_2 = ('modif' not in file.name_base)
        
        return condition_1 and condition_2 
    
            
    def find_csv_file(self, folder = 'Laser', d_type = 'analogpulse') :
        
        directory = Directory( os.path.join( self.exp_dir, folder))
        corres_files = []
        
        if '.csv' in directory.filepath_list:
            
            for path in directory.filepath_list['.csv']:
            
                file = File ( path )
                
                if self.check_func[ d_type] (file):
                    
                    corres_files.append(file)
                    
            if len( corres_files ) == 1: # excatly one file found, great!
    
                self.files[d_type] = corres_files [0]   
                
            elif len( corres_files ) > 1:
                
                [ print(f.path) for f in corres_files]
                raise ValueError(" multiple" + d_type + " files with the same day tag!")
            
            if isinstance(self.files[d_type], File):
                print(d_type, ' : ', self.files[d_type].name )
        # else:

        #    raise ValueError( d_type + " file doesn't exist!")
           
    def find_analogpulse_DLC_csv(self):
        
        self.find_csv_file( folder = 'Laser', d_type = 'analogpulse')
        self.find_csv_file( folder = 'DLC', d_type = 'DLC')
        self.check_if_already_analyzed()
        
    def save_laser_detections (self, starts, ends, method, mean_shift, sd_shift):

        if isinstance(self.files['analogpulse'], File):
            
            analog_path = self.files['analogpulse'].path
            
        else:
            
            analog_path = 'No analog pulse provided.'
            
        metadatas=[
            [self.video.path],
            [analog_path],
            [method],
            [str(mean_shift) + '+/-' + str(sd_shift)]
            ]
    
        df = pd.DataFrame( np.column_stack(( starts, ends)),
                           columns = ['ON', 'OFF'])
    
        resultFilePath = os.path.join(self.exp_dir, 'Laser', 
                                      self.result_filename)
        
        with open(resultFilePath, 'w') as resultfile:
    
            csvResult=csv.writer(resultfile,delimiter=',', lineterminator='\n')
            csvResult.writerows(metadatas)
            
        df.to_csv(resultFilePath,  mode = 'a', index = False)
    
    def get_result_filename(self):
        
        if isinstance(self.files['DLC'], File):
            return self.files['DLC'].name_base + '_Laser.csv'
        
        else:
            
            return self.video.name_base +  '_Laser.csv'
        
    def get_laser_start_end(self, pulse, analogpulse, true_duration):
        
        if pulse.use_laser_detection_only(analogpulse) or pulse.enforce_use_laser_detection_only :
            
            starts, ends = pulse.starts, pulse.ends
            
        else:
            
            analogpulse.align_to_video(pulse.shift_rel_to_analogpulse)
            starts, ends = analogpulse.starts, analogpulse.ends
            
        centers = np.average( np.column_stack( (starts, ends)), axis = 1).astype(int)
        
        
        return starts, ends, centers
    

    def plot_laser_detections(self, ax, centers, y_plot, title = None):
        
        for center, y in zip( centers[:-1], y_plot[:-1]) :
            
            ax.axvline( x = center, ymin = y, ymax = 1, ls = '--', c = 'limegreen', zorder= 0)
        
        ax.axvline( x = centers[-1], ymin = y_plot[-1], ymax = 1, ls = '--', c = 'limegreen', zorder= 0, label = 'output')

        ax.set_title(title, fontsize = 12)      
        ax.legend(frameon = False, loc = 'lower right', fontsize = 12, bbox_to_anchor=(1.25, 0.15))

    def remove_if_already_exists(self, filepath):
        
        df = pd.read_csv(filepath, header = [0])
        
        if self.video.path in df['path'].values:
            print('reanalyzed')
            df = df.drop(np.where(df['path'] == self.video.path)[0])
            
        df.to_csv(filepath, index = False)
        
    def add_file_to_csv(self, filepath, info, file_no, shift_mean = None, shift_sd = None):
        
        self.remove_if_already_exists(filepath)
        
        if  shift_mean != None:
            
            data = [self.video.path, info, shift_mean, shift_sd, file_no, self.day_tag]
            
        else:
            
            data = [self.video.path, info, file_no, self.day_tag]
            
        file = open(filepath, 'a+', newline ='')
          
        with file:    
            
            write = csv.writer(file)
            write.writerows([data])

    @staticmethod 
    def get_videofile_paths_from_csv(csv_filepath):
        
        df = pd.read_csv(csv_filepath, header = [0])
        filepath_list = df[df.columns[0]].values
    
        return filepath_list        
    
    @staticmethod
    def remove_resolved_file_from_csv(csv_filepath, filepath):
        
        df = pd.read_csv(csv_filepath, header = [0])
        if filepath in df['path'].values :
            
            df = df.drop( np.where(df['path'] == filepath)[0])
            df.to_csv(csv_filepath, index = False)
            
    @staticmethod
    def create_problematic_csv(filepath):
        
        if not os.path.exists(filepath):
            file = open(filepath, 'w', newline ='')
      
            with file:
                # identifying header  
                header = ['path', 'Problem', 'file_no', 'trial']
                writer = csv.DictWriter(file, fieldnames = header)
                writer.writeheader()  
    
    @staticmethod
    def create_summary_csv(filepath):
        
        if not os.path.exists(filepath):
            file = open(filepath, 'w', newline ='')
      
            with file:
                # identifying header  
                header = ['path', 'method', 'shift_mean', 'shift_sd', 'file_no', 'trial']
                writer = csv.DictWriter(file, fieldnames = header)
                writer.writeheader()
                
    @staticmethod
    def pdf_saveas_jpg(path =  '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/LASER_DETECTION'):
        
        directory = Directory(path)
        pdf_list = directory.get_spec_files(extensions=['.pdf'])

        for pdf in pdf_list:
            
            images = convert_from_path(pdf)
            dirname = os.path.join( os.path.dirname(pdf), 'JPEG' )
            name = os.path.splitext( os.path.basename(pdf) )[0]
            jpg_path = os.path.join( dirname, name + '.jpg')
            
            if not os.path.exists(jpg_path):
                images[0].save(jpg_path , 'JPEG')
    

class ProblematicFileAnalysis:
    
    def __init__(self, csv_filepath, center_vicinity_h_thresh):
        
        self.csv_filepath = csv_filepath
        self.filepath_dict = {}
        self.issues = None
        self.read_prob_csv()
        self.filepath_list = []
        self.df = None
        self.organize_files(csv_filepath)
        self.new_thresh = {}
        self.set_new_thresh(center_vicinity_h_thresh)
        
    def set_new_thresh(self, original_thresh):
        
        self.new_thresh['Not all pulses are detected'] = original_thresh * 0.8 
        self.new_thresh['Extra pulses are detected'] = original_thresh * 1.2

    def read_prob_csv(self):
        
        self.df = pd.read_csv(self.csv_filepath, header = [0])
        self.filepath_list = df['path']
        return self.filepath_list
        
    
    def organize_files(self):
        

        self.issues = list( set(df['Problem']))

        self.filepath_dict = {issue:  df['path'][ df['Problem'] == issue] 
                                  for issue in self.issues}



def remove_element(arr, ind):
    
    return np.hstack((arr[:ind], arr[ind+1:]))