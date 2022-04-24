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
    restretic frame grabbing to a thread and then queue a stack of frames
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
    
    def __init__ (self, video_path_list, DLC_path_list):
        
        self.video_path_list = video_path_list
        self.DLC_path_list = DLC_path_list
        
        
        
    def set_laser_boundaries(self, file_no = 0, 
                             p_cutoff = 0.995, 
                             treadmil_length_in_cm = 33, 
                             treadmil_length_in_pix = 1000):
        
        DLC_filepath= self.DLC_path_list [file_no]
        video_filepath = self.video_path_list [file_no]
        
        mloc_lower = MouseLocation( DLC_filepath , 'lower', p_cutoff = p_cutoff,
                                    treadmil_length_in_cm = treadmil_length_in_cm, 
                                    treadmil_length_in_pix = treadmil_length_in_pix)
        
        mloc_upper = MouseLocation( DLC_filepath , 'upper', p_cutoff = p_cutoff, 
                                    treadmil_length_in_cm = treadmil_length_in_cm, 
                                    treadmil_length_in_pix = treadmil_length_in_pix)
        
        ax = mloc_lower.plot_y(p_cutoff = p_cutoff)
        ax = mloc_upper.plot_y(p_cutoff = p_cutoff, ax = ax)
        
        x_range = { mloc.pos_in_vid : 
                   mloc.x_range for mloc in [mloc_lower, mloc_upper] }
        
        y_range = { mloc.pos_in_vid : 
                   mloc.y_range for mloc in [mloc_lower, mloc_upper] }

        return x_range, y_range 
    
    
    
    def one_video(self, 
                  file_no = 0,   
                  low_hsv_thresh = (150, 50, 100), 
                  high_hsv_thresh = (190, 255, 255),
                  low_RGB_thresh = (150, 10,60), 
                  high_RGB_thresh = (255, 120, 140),
                  nb_frames = None,
                  treadmil_length_in_cm = 33,
                  p_cutoff_ranges = 0.995):
        
        vidstr = VideoStream( self.video_path_list[ file_no ] , nb_frames = nb_frames). start()
        x_range, y_range = self.set_laser_boundaries(file_no = file_no, 
                                                     p_cutoff = p_cutoff_ranges, 
                                                     treadmil_length_in_cm = treadmil_length_in_cm, 
                                                     treadmil_length_in_pix = vidstr.treadmil_length_in_pix)
        
        laser = LaserDetector ( low_RGB_thresh, high_RGB_thresh )
                    
        laser.detect_laser_starts(vidstr, nb_frames = nb_frames,
                                  x_range = x_range, y_range = y_range)
            
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
        self.n_pix_in_range = 0
        self.height, self.width = self.image.shape[:2]
        self.x_range = x_range
        self.y_range = y_range
        self.constrain_frame = self.whether_constrain_frame()
        # self.x_range = x_range or { obj: [0, self.width] for obj in ['upper', 'lower']}
    
    def whether_constrain_frame(self):
        
        if self.x_range == None:
            
            return False
            
        else:
            
            return True

        
    def find_contours_with_hsv(self, low_hsv_thresh, high_hsv_thresh):
        
        blurred = cv2.GaussianBlur(self.image, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low_hsv_thresh, high_hsv_thresh)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        if len(contours) > 1 :
            
            contour_max = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            countour_max_area = cv2.contourArea(contour_max) 
        
        return contours , countour_max_area,  mask
    
    def in_range_RGB(self,  low_RGB_thresh, high_RGB_thresh):
        
        image_blr = cv2.GaussianBlur(self.image, (3, 3), 0)
        masked = cv2.inRange(image_blr, low_RGB_thresh, high_RGB_thresh)
        self.count_threshold_passed_pix(masked, image_parts = ['upper', 'lower'])

        return masked

    def count_threshold_passed_pix(self, masked, image_parts = ['upper', 'lower']):
        
        if self.constrain_frame:
            
            self.n_pix_in_range = 0
            
            for part in image_parts:
                
                self.n_pix_in_range += np.count_nonzero(
                                                masked[ self.y_range[part][0] :  self.y_range[part][1],
                                                        self.x_range[part][0] :  self.x_range[part][1]]
                                                        )
            
        else:
            
            self.n_pix_in_range = np.count_nonzero(masked)
        
class LaserDetector :
    
    def __init__(self, 
                 low_RGB_thresh : tuple, 
                 high_RGB_thresh : tuple,
                 low_hsv_thresh = None, 
                 high_hsv_thresh = None) :
         
        self.area_list = []
        self.color = None
        self.location_list = None
        self.first_laser_found = False
        self.low_RGB_thresh = low_RGB_thresh
        self.high_RGB_thresh = high_RGB_thresh
        self.low_hsv_thresh = low_hsv_thresh
        self.high_hsv_thresh = high_hsv_thresh

        
    def plot_contour(self, mask, frame_no, image, contours = None):
        
        while True:
    	
            cv2.imshow('contours frame # {}'.format(frame_no), image) # display the image and wait for a keypress
            cv2.imshow('mask', mask)
            # cv2.drawContours(image, contours, -1, (0,0, 255), 3)

            if cv2.waitKey(1) == 27 :
                break
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        
    def detect_laser_starts(self, vidstr, 
                            nb_frames = None, 
                            x_range = None, 
                            y_range = None):
        
        nb_frames = nb_frames or vidstr.nb_frames 
        self.area_list = np.zeros(nb_frames + 1)
        
        n = 0
        while n < nb_frames:
            
            
            frame = Frame (vidstr, x_range = x_range, y_range = y_range)
            
            n += 1
            frame.no = n
            
            if int(frame.no) % 250 == 0:
                print('frame :', frame.no)
                
            
            # contours, countour_max_area, mask = frame.find_contours_with_hsv(self.low_hsv_thresh,self. high_hsv_thresh)
            # self.area_list [frame.no - 1] = countour_max_area
            
            mask = frame.in_range_RGB(self.low_RGB_thresh, 
                                      self.high_RGB_thresh)
            
            self.area_list [frame.no - 1] =  frame.n_pix_in_range
            
            if frame.no == 2414 :#or frame.no == 908:
                
                self. plot_contour(mask, frame.no, frame.image)
                # print('frame :', frame.no, 'max area = ',  self.area_list [frame.no - 1] )
           
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
        self.filtered_signal = None
        self.events = []
        self.centers = []
        
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
        
        self.events, _ = find_peaks( -self.signal, height = 40)
        
    def plot_events(self, ax = None):
        
        ax = ax or plt.subplots()[1]

        ax.plot( self.events, self.signal [self.events] , 'x', ms = 10, c = 'k')

        return ax
    
    def plot_centers(self, ax = None):
           
        ax = ax or plt.subplots()[1]

        ax.plot(self.centers, np.full(len(self.centers), -150) , 'x', ms = 10, c = 'g')

        return ax
    
    def find_centers(self):
        
        self.centers = np.average( np.column_stack((self.events[::2], self.events[1::2])), axis = 1 )
        
    def plot( self, signal = None, label = 'true signal', ax = None ) :
        
        signal = signal or self.signal
        ax = ax or plt.subplots()[1]
        
        ax.plot(signal,'-o', ms = 3, label = label)
        ax.set_ylabel('# blue pixels in frame', fontsize = 15)
        ax.set_xlabel('# frame', fontsize = 15)
        ax.legend(frameon = False, fontsize = 10)
        
        return ax
    
    def compare_methods(self, gauss_window = 5, 
                        low_f = 1, high_f = 50, 
                        filt_order = 10,
                        peak_heights = 40):
        
        sig_copy = self.signal.copy()
        ax = self.plot( label = 'true signal', ax = None )
        
        self.low_pass_filter(low_f = low_f, high_f = high_f, fs = self.fs, filt_order = filt_order)
        ax = self.plot(label = 'low pass filtered signal', ax = ax)

        self.signal = sig_copy
        self.gauss_filter(gauss_window = gauss_window)
        ax = self.plot(label = 'gaussian filtered signal', ax = ax)
        
        self.low_pass_filter(low_f = low_f, high_f = high_f , fs = self.fs, filt_order= filt_order)
        ax = self.plot(label = 'low pass filtered smoothed signal', ax = ax)
        
        self.find_peaks(height = peak_heights)
        ax = self.plot_events(ax = ax)
        
        
        self.find_centers()
        self.plot_centers(ax = ax)

        self.signal = sig_copy

    # self.gauss_filter(gauss_window = gauss_window)
    # self.low_pass_filter(low_f = low_f, high_f = high_f, fs = self.fs, filt_order = filt_order)
    # self.find_peaks(height = peak_heights)
    # self.find_centers()


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
        
        self.x_range = np.array([ np.min( self.x[ high_likelohood_ind ]),
                         np.max( self.x[ high_likelohood_ind ])]).astype(int)
        
        self.y_range = np.array([ np.min( self.y[ high_likelohood_ind ]) - self.max_dist_nose_to_laser,
                         np.max( self.y[ high_likelohood_ind ])]).astype(int)  
        
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
filepath_list  = ['/home/shiva/Desktop/test/Vglut2D2Cre#55_SquarePulse_STR_0-35mW_15cm-s_Stacked_f17.avi' ,
                     '/media/shiva/LaCie/Data_INCIA_Shiva/2021_02_25_newTrain2_ok/Vglut2D2Cre#58_STRalone_0.35mW_15cm-s_Stacked_f08.avi',
                     '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_70/STR/squarepulse_0-35mW/Video/Vglut2D2Cre#70_SquarePulse_STR_0-35mW_15cm-s_Stacked_f14.avi',
                     '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-35mW/Video/Vglut2D2Cre#59_SquarePulse_STR_0-35mW_15cm-s_Stacked_f05.avi',
                     '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-5mW/Video/Vglut2D2Cre#59_SquarePulse_STR_0-5mW_15cm-s_q11_Stacked.avi']

filepath_list_DLC = ['/home/shiva/Desktop/test/Vglut2D2Cre#55_STNalone_0.5mW_15cm-s_Stacked_f15DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
                '/media/shiva/LaCie/Data_INCIA_Shiva/2021_02_25_newTrain2_ok/Vglut2D2Cre#58_STRalone_0.35mW_15cm-s_Stacked_f08DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
                '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_70/STR/squarepulse_0-35mW/DLC/Vglut2D2Cre#70_SquarePulse_STR_0-35mW_15cm-s_Stacked_f14_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
                '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-35mW/DLC/Vglut2D2Cre#59_SquarePulse_STR_0-35mW_15cm-s_Stacked_f05_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv',
                '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/Mouse_59/STR/squarepulse_0-5mW/DLC/Vglut2D2Cre#59_SquarePulse_STR_0-5mW_15cm-s_q11_Stacked_DLC_resnet_50_treadmillOptoJun3shuffle1_1030000.csv']

n = 3
filepath_DLC = filepath_list_DLC [ n ]
filepath = filepath_list [ n ]



analysis = Analyze (filepath_list, filepath_list_DLC)

RGB_blueLower = (150, 10,60)
RGB_blueUpper = (255, 120, 140)
areas = analysis.one_video( file_no = n, 
                           low_RGB_thresh = RGB_blueLower, 
                           high_RGB_thresh = RGB_blueUpper, 
                           treadmil_length_in_cm = 33,
                           p_cutoff_ranges = 0.995,
                           nb_frames = 5000)


Analyze.pickle_obj({'area' : areas}, filepath.replace('avi', 'pkl') )

# data = Analyze.load_pickle(filepath.replace('avi', 'pkl') )
# plt.plot(data['area'])

pulse = Pulse(areas, fs = 250)
    
pulse.compare_methods(gauss_window = 5, low_f = 1, high_f = 50, 
                      filt_order = 10,
                      peak_heights = 40)






def recons_by_direct_reversion():

    from numpy.fft import fft, ifft 
    from scipy import signal as sig
    from scipy import stats, real
    
    N = 2000
    a =- 0.97
    L = 50
    spos  = stats.bernoulli.rvs(loc = 0, p = 0.6, size = int(N / L))
    s = np.kron(spos, np.ones(L))
    d = np.zeros(N)
    d[0] = 1
    
    h = sig.lfilter( [1, 0.5, 0.95] , [1, a] , d)
    
    H = fft(h, N)
    X = fft (s) * H
    x = real (ifft (X))
    fig, ax = plt.subplots()
    ax.plot(x)
    
    x_rec = real( ifft (X / H))
    ax.plot(x_rec)
    





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