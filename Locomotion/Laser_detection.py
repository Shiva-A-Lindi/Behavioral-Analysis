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
from threading import Thread
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
 
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue
    
from scipy.signal import butter, sosfilt, sosfreqz, spectrogram, sosfiltfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
        
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
    
    
class Analyze :
    
    def __init__ (self, 
                  video_path_list, 
                  DLC_path_list,
                  thresh_method = 'rgb',
                  area_cal_method = 'pix_count',
                  image_parts = ['upper', 'lower'],
                  treadmil_length_in_cm = 33):
        
        self.video_path_list = video_path_list
        self.DLC_path_list = DLC_path_list
        self.thresh_method = thresh_method
        self.area_cal_method = area_cal_method
        self.image_parts = image_parts
        self.treadmil_length_in_cm = treadmil_length_in_cm 
        
    def set_laser_boundaries(self, file_no = 0, 
                             p_cutoff = 0.995, 
                             treadmil_length_in_pix = 1000,
                             max_dev_in_cm = 2, 
                             plot_bounds = False):
        
        DLC_filepath= self.DLC_path_list [file_no]
        video_filepath = self.video_path_list [file_no]
        
        mloc_lower = MouseLocation( DLC_filepath , 'lower', p_cutoff = p_cutoff,
                                    treadmil_length_in_cm = self.treadmil_length_in_cm, 
                                    treadmil_length_in_pix = treadmil_length_in_pix,
                                    max_dev_in_cm = max_dev_in_cm )
        
        mloc_upper = MouseLocation( DLC_filepath , 'upper', p_cutoff = p_cutoff, 
                                    treadmil_length_in_cm = self.treadmil_length_in_cm, 
                                    treadmil_length_in_pix = treadmil_length_in_pix,
                                    max_dev_in_cm = max_dev_in_cm )
        
        if plot_bounds:
            
            ax = mloc_lower.plot_y(p_cutoff = p_cutoff)
            ax = mloc_upper.plot_y(p_cutoff = p_cutoff, ax = ax)
            
        x_range = { mloc.pos_in_vid : 
                   mloc.x_range for mloc in [mloc_lower, mloc_upper] }
        
        y_range = { mloc.pos_in_vid : 
                   mloc.y_range for mloc in [mloc_lower, mloc_upper] }

        return x_range, y_range 
    
    
    
    def one_video(self, 
                  file_no = 0,   
                  low_img_thresh = (150, 10,60), 
                  high_img_thresh = (255, 120, 140),
                  nb_frames = None,
                  p_cutoff_ranges = 0.995,
                  constrain_frame = True,
                  max_dev_in_cm = 2):
        
        vidstr = VideoStream( self.video_path_list[ file_no ] , nb_frames = nb_frames). start()
        
        x_range, y_range = None, None
        
        if constrain_frame:
            x_range, y_range = self.set_laser_boundaries(file_no = file_no, 
                                                         p_cutoff = p_cutoff_ranges, 
                                                         treadmil_length_in_pix = vidstr.treadmil_length_in_pix,
                                                         max_dev_in_cm = max_dev_in_cm )
            
        laser = LaserDetector ( low_img_thresh, high_img_thresh )
                    
        laser.detect(vidstr, 
                     nb_frames = nb_frames,
                     x_range = x_range, 
                     y_range = y_range,
                     area_cal_method = self.area_cal_method,
                     thresh_method = self.thresh_method,
                     image_parts = self.image_parts)
            
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
        # self.laser_area = 0
        self.height, self.width = self.image.shape[:2]
        self.x_range = x_range
        self.y_range = y_range
        
        self.constrain_frame = self.whether_constrain_frame()
        self.detection_thresh_methods = {
                                        'hsv' : self.mask_hsv, 
                                        'rgb' : self.mask_rgb}
        
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
    
class LaserDetector :
    
    def __init__(self, 
                 low_img_thresh : tuple, 
                 high_img_thresh : tuple) :
         
        self.area_list = []
        self.color = None
        self.location_list = None
        self.first_laser_found = False
        self.low_img_thresh = low_img_thresh
        self.high_img_thresh = high_img_thresh
        

        
    def plot_contour(self, mask, frame_no, image, contours = None):
        
        while True:
    	
            cv2.imshow('contours frame # {}'.format(frame_no), image) # display the image and wait for a keypress
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
            
            if int(frame.no) % 250 == 0:
                print('frame :', frame.no)
                
            
            masked, self.area_list [frame.no - 1] = frame.detect_laser( self.low_img_thresh, 
                                                                        self.high_img_thresh, 
                                                                        thresh_method = thresh_method,
                                                                        area_cal_method = area_cal_method,
                                                                        image_parts = image_parts )
            

            
            # if frame.no == 2175 :
                
            #     self. plot_contour(masked, frame.no, frame.image)
           
    @staticmethod                
    def moving_average_array(X, n):
        
        '''Return the moving average over X with window n without changing dimesions of X'''
        z2= np.cumsum(np.pad(X, (n,0), 'constant', constant_values=0))
        
        z1 = np.cumsum(np.pad(X, (0,n), 'constant', constant_values=X[-1]))
        return (z1-z2)[(n-1):-1]/n
                    
            
        
class Pulse :
    
    def __init__( self, sig, fs = 250, low_f = 1, high_f = 50):
        
        self.signal = sig
        self.low_f = low_f
        self.high_f = high_f
        self.fs = fs
        self.events = []
        self.centers = []
        self.laser_starts = []
        self.laser_ends = []
        
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
        
    def plot_events(self, ax = None):
        
        ax = ax or plt.subplots()[1]

        ax.plot( self.events, self.signal [self.events] , 'x', ms = 10, c = 'k')

        return ax
    
    def plot_centers(self, ax = None):
           
        ax = ax or plt.subplots()[1]

        ax.plot(self.centers, np.full(len(self.centers), -0.1) , 'x', ms = 10, c = 'r')

        return ax
    
    def find_laser_centers(self,
                           gauss_window = 5, 
                           low_f = 1, high_f = 50, 
                           filt_order = 10,
                           peak_heights = 40):
        
        self.pre_process( gauss_window = gauss_window, 
                        low_f = low_f, high_f = high_f, 
                        filt_order = filt_order)
        
        self.find_peaks(height = peak_heights)

        try:

            self.centers = np.average( np.column_stack((self.events[::2], self.events[1::2])), axis = 1 )

        except (ValueError, ZeroDivisionError): 
            print("couldn't find centers")
        
    def plot( self, signal = None, label = 'true signal', ax = None ) :
        
        signal = signal or self.signal
        ax = ax or plt.subplots()[1]
        
        ax.plot(signal, '-o', ms = 1, label = label)
        ax.set_ylabel('# blue pixels in frame', fontsize = 15)
        ax.set_xlabel('# frame', fontsize = 15)
        ax.legend(frameon = False, fontsize = 10)
        
        return ax
    

    def thresh(self, laser_area_thresh = 180):
        
        
        return np.where(self.signal > laser_area_thresh) [0]
    
    def detect_laser_start (self, ind_thr_pass, jump_by = 5):
        
        signal_shift_fwd = Pulse.shift( self.signal, 1, 0)
        ind_start_jump = np.where( self.signal  > jump_by * signal_shift_fwd ) [0]
        
        self.laser_starts = np.intersect1d (ind_thr_pass, ind_start_jump) - 1
        
    def detect_laser_ends (self, ind_thr_pass, jump_by = 5):
        
        signal_shift_bck = Pulse.shift( self.signal, -1, 0)
        ind_end_jump = np.where( self.signal  > jump_by * signal_shift_bck ) [0]
        
        self.laser_ends = np.intersect1d (ind_thr_pass, ind_end_jump) + 1
        
    def detect_laser_start_end(self,  laser_area_thresh = 180, jump_by = 5):
        
        ind_thr_pass = self.thresh( laser_area_thresh = laser_area_thresh )
        
        self.detect_laser_start ( ind_thr_pass, jump_by = jump_by)
        self.detect_laser_ends (ind_thr_pass, jump_by = jump_by)
        

    def plot_start_ends(self):
        
        ax = self.plot()
        ax.plot( self.laser_starts, self.signal[ self.laser_starts], 'x', c = 'magenta')
        ax.plot( self.laser_ends, self.signal[ self.laser_ends], 'x', c = 'r')

    def compare_methods(self, gauss_window = 5, 
                        low_f = 1, high_f = 50, 
                        filt_order = 10,
                        peak_heights = 40):
        
        sig_copy = self.signal.copy()
        ax = self.plot( label = 'true signal', ax = None )
        
        # self.low_pass_filter(low_f = low_f, high_f = high_f, fs = self.fs, filt_order = filt_order)
        # ax = self.plot(label = 'low pass filtered signal', ax = ax)

        self.gauss_filter(gauss_window = gauss_window)
        ax = self.plot(label = 'gaussian filtered signal', ax = ax)
        
        self.low_pass_filter(low_f = low_f, high_f = high_f , fs = self.fs, filt_order= filt_order)
        ax = self.plot(label = 'low pass filtered smoothed signal', ax = ax)
        
        self.find_peaks(height = peak_heights)
        ax = self.plot_events(ax = ax)
        

        self.signal = sig_copy

        return ax
    
    def pre_process(self,
                    gauss_window = 5, 
                    low_f = 1, high_f = 50, 
                    filt_order = 10,
                    peak_heights = 40):
        
        self.gauss_filter(gauss_window = gauss_window)
        self.low_pass_filter(low_f = low_f, high_f = high_f, fs = self.fs, filt_order = filt_order)
        self.normalize_filtered_sig()
        
    
    def normalize_filtered_sig(self):
        
        
        max_sig = max( abs ( self.signal[ self.signal < 0]) )
        self.signal = self.signal / max_sig

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
    def save_pdf(fig, figname, size = (8,6)):
        
        fig.set_size_inches(size, forward=False)
        fig.savefig(figname + '.pdf', dpi = 500, facecolor='w', edgecolor='w',
                        orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)


class MouseLocation:
    

    def __init__( self, DLC_filepath, pos_in_vid = 'upper', p_cutoff = 0.995, 
                 treadmil_length_in_cm = 33, treadmil_length_in_pix = 1000,
                 max_dev_in_cm = 2):
        
        
        self.DLC_filepath = DLC_filepath
        self.x = None
        self.y = None
        self.p = None
        self.side = None
        self.x_range = None
        self.y_range = None
        self.pos_in_vid = pos_in_vid
        self._set_side_from_pos(pos_in_vid)
        self.get_location()
        self.max_dist_nose_to_laser = self.cal_max_deviat_nose_to_laser(treadmil_length_in_cm, 
                                                                        treadmil_length_in_pix, 
                                                                        max_dev_in_cm = max_dev_in_cm)
        self.get_cor_range( p_cutoff = p_cutoff )
        
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
        
        print( ' x range bounds = ', self.x_range, 
               ' y range bounds = ', self.y_range)
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


plt.close( 'all' )
filepath_list  = [
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_55/STR/squarepulse_0-35mW/Video/Vglut2D2Cre#55_SquarePulse_STR_0-35mW_15cm-s_Stacked_f17.avi' ,
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_58/STR/squarepulse_0-35mW/Video/Vglut2D2Cre#58_SquarePulse_STR_0-35mW_15cm-s_Stacked_f08.avi',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_70/STR/squarepulse_0-35mW/Video/Vglut2D2Cre#70_SquarePulse_STR_0-35mW_15cm-s_Stacked_f14.avi',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-35mW/Video/Vglut2D2Cre#59_SquarePulse_STR_0-35mW_15cm-s_Stacked_f05.avi',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-5mW/Video/Vglut2D2Cre#59_SquarePulse_STR_0-5mW_15cm-s_q11_Stacked.avi',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/betapulse_0-5mW/Video/Vglut2D2Cre#59_betapulse_STR_0-5mW_15cm-s_q12_Stacked.avi']

filepath_list_DLC = [
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_55/STR/squarepulse_0-35mW/DLC/Vglut2D2Cre#55_SquarePulse_STR_0-35mW_15cm-s_Stacked_f17_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_58/STR/squarepulse_0-35mW/DLC/Vglut2D2Cre#58_SquarePulse_STR_0-35mW_15cm-s_Stacked_f08_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_70/STR/squarepulse_0-35mW/DLC/Vglut2D2Cre#70_SquarePulse_STR_0-35mW_15cm-s_Stacked_f14_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-35mW/DLC/Vglut2D2Cre#59_SquarePulse_STR_0-35mW_15cm-s_Stacked_f05_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-5mW/DLC/Vglut2D2Cre#59_SquarePulse_STR_0-5mW_15cm-s_q11_Stacked_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
    '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/betapulse_0-5mW/DLC/Vglut2D2Cre#59_betapulse_STR_0-5mW_15cm-s_q12_Stacked_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv']

n = 5

filepath_DLC = filepath_list_DLC [ n ]
filepath = filepath_list [ n ]

# area_cal_method = 'contour'
area_cal_method = 'pix_count'

treadmil_length_in_cm = 33
constrain_frame= True

RGB_blueLower = (150, 10,60)
RGB_blueUpper = (255, 120, 140)

# RGB_blueLower = (50, 80, 60) ## HSV
# RGB_blueUpper = (150, 255, 255)

analysis = Analyze (filepath_list, filepath_list_DLC,                   
                    thresh_method = 'rgb',
                    area_cal_method = area_cal_method,
                    image_parts = ['upper', 'lower'],
                    treadmil_length_in_cm = treadmil_length_in_cm)


areas = analysis.one_video( file_no = n, 
                            low_img_thresh = RGB_blueLower, 
                            high_img_thresh = RGB_blueUpper, 
                            p_cutoff_ranges = 0.995,
                            nb_frames = 5000,
                            constrain_frame= constrain_frame,
                            max_dev_in_cm = 1.5)

pulse = Pulse(areas, fs = 250)
# pulse.detect_laser_start_end(laser_area_thresh = 180, jump_by = 4)
# pulse.plot_start_ends()
# ax = pulse.compare_methods(gauss_window = 5, 
#                             low_f = 1, high_f = 30, 
#                             filt_order = 10,
#                             peak_heights = 40)

ax = pulse.plot()
pulse.find_laser_centers(
                       gauss_window = 5, 
                       low_f = 1, high_f = 50, 
                       filt_order = 10,
                       peak_heights = 0.4)

ax = pulse.plot()
pulse.plot_centers(ax = ax)
pulse.plot_events(ax = ax)
pulse.save_pdf( ax.get_figure(), os.path.basename(filepath).replace('.avi', '_constrained_'  + str(constrain_frame)) )


# Analyze.pickle_obj({'area' : areas}, filepath.replace('avi', 'pkl') )\
# data = Analyze.load_pickle(filepath.replace('avi', 'pkl') )
# plt.plot(data['area'])






# image = cv2.imread('/home/shiva/Desktop/test/frames_f17/output_00785.png')
# image = cv2.GaussianBlur(image, (3, 3), 0)
# mask = cv2.inRange(image, RGB_blueLower, RGB_blueUpper)
# print(np.count_nonzero(mask))
# # print(np.nonzero(mask))
# while True:
 	
#     cv2.imshow('785', image)
#     cv2.imshow('785', mask)
#     if cv2.waitKey(1) == 27 :
#         break
    
# cv2.destroyAllWindows()
# cv2.waitKey(1)


# s = pd.Series(areas)
# fig, ax = plt.subplots()
# ax.plot(s.rolling(20).std())