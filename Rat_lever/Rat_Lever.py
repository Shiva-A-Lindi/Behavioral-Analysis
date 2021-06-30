import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
import pandas as pd
from pandas import read_excel
from matplotlib.font_manager import FontProperties
from scipy.ndimage.interpolation import shift
import os
import glob
import timeit
from scipy import stats
import statsmodels.stats.api as sms
from matplotlib.collections import LineCollection
from tempfile import TemporaryFile
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from statannot import add_stat_annotation
# import numpy_indexed as npi
#print(os.path.dirname(__file__))
import itertools
import pickle
from pathlib import Path
import yaml
from ruamel.yaml import YAML
from os.path import dirname as up
import shutil
import fnmatch
flatten = itertools.chain.from_iterable

def set_ticks(ax):
    ax.get_xaxis().set_tick_params(direction='out',labelsize = 20)
    ax.xaxis.set_ticks_position('bottom')
    ax.get_yaxis().set_tick_params(direction='out',labelsize = 20)
    ax.yaxis.set_ticks_position('left')
##################### Config file  ##########################################################################################
def create_config_template():
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """

    yaml_str = """\
    # Project definitions (do not edit)
        \n
        Task:
        date:
        experimentor:
        \n
    # Project path (change when moving around)
        \n
        project_path: '/home/shiva/Desktop/Rat_Lever_Analysis'
        \n
    # Annotation data set configuration 
        \n
        body_part_list:
        exp_par: # optogentics, pharmacology, ...
        optogenetic_manip: # True, False
        r_or_l: # 'R' or 'L' the hand performing the task
        fp_trial: # number of frames per trial
        fps: # frame per second of video
        laser_duration: # ms. Laser duration in partial-MT protocol. 
        laser_delay: # delay of laser onset relative to pad-off
        laser_protocol: # all-RT, all-MT, prtial-MT,...
        laser_pulse: # beta, square,...
        laser_intensity: # mW
        \n
    # Plotting configuration 
        \n
        laser_on_color:
        laser_off_color:
        markersize:
        alphavalue:
        n_grid: # number of grids in x for averaging y among trials
        \n
    # Analysis configuration
        \n
        n_timebin:
        pad_length_cm: # length of pad in cm used to scale videos
        velocity_mask: # an arbitrary value to mask the unwanted timebins of trials
        \n
    # Cropping Parameters (varies according to croppings made during DLC analysis)
        \n
        frame_height: # pix
        \n
    # if Refinement
        \n
        if_refine:
        \n
    # Refinement configuration 
        \n
        max_speed: # cm/s. The max speed estimated for the animal 
        jitter_proximity_thresh_rho: #cm. the acceptable jitter in abs(r) vector to be considered the same jump
        jitter_proximity_thresh_phi: #rad. the acceptable jitter in the angel of r vector to be considered the same jump
        thresh_adjust: # if there are 5 jumped points the threshold is <jitter_proximity_thresh_rho>
        t_s: # number of timebins before and after for reference in jitter removal
        p_cutoff: # the DLC likelihood threshold for considering detections
        n_good_points_ratio:
        max_delta_r_per_frame: # cm/4ms
        cor_t_prox_thresh:  #number of frames more than which the low likelihood points are not allowd to be connected
        pad_thresh:
        window: # moving average window for position
        window_veloc: # moving average window for velocity
        """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile
    
def read_config(configname):
    """
    Reads structured config file defining a project.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = os.path.dirname(configname)
                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                    err.args[2]
                    == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg

def write_config(configname, cfg):
    """
    Write structured config file.
    """
#     write_plainconfig(configname, cfg)
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable for backward compatibility.
        if not "skeleton" in cfg.keys():
            pass
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
        
def set_config_file(folder,path): 
    cfg = {'Task': 'Rat_lever_press',
        'date':'01-05-2020',
        'body_part_list': ['finger0',  'finger1base', 'finger1joint', 'finger2base', 'finger2joint',
                           'forearm', 'elbow'],
        'laser_on_color': 'lightskyblue',
        'laser_off_color': 'k',
        'markersize': 5,
        'alphavalue': 0.5,
        'n_grid': 20,
        'n_timebin': 10,
        'pad_length_cm': 2,
        'fps': 250,
        'frame_height': 400,
        'if_refine': True,
        'max_speed': 100,
        'jitter_proximity_thresh_rho': 0.2,
        'jitter_proximity_thresh_phi': 30/180*np.pi,
        'thresh_adjust': 1/5,
        't_s': 4,
        'p_cutoff': 0.8,
        'n_good_points_ratio':1/4,
        'cor_t_prox_thresh':100,
        'pad_thresh':5,
        'window':10,
        'window_veloc':5,
        'velocity_mask': 2000,
        'laser_duration': 250,
        'laser_delay': 25}

    edits = {'max_delta_r_per_frame': cfg['max_speed']/cfg['fps']}


    configname = os.path.join(path, folder, 'config_'+folder+'.yaml')
    if not os.path.exists(configname):
        write_config(configname, cfg)
        edit_config(configname, edits)
    return configname


################################################## Classes ##################################################################

class Trials:
    ''' stores the mean and errors of the trajectory for any number of trials'''
    
    def __init__(self,n_grid):
        self.x = np.zeros((n_grid)) * np.nan
        self.y = np.zeros((n_grid)) * np.nan
        self.err_x = np.zeros((n_grid)) * np.nan
        self.conf_inter_x = np.zeros((n_grid, 2)) * np.nan
        self.conf_inter_y = np.zeros((n_grid, 2)) * np.nan
        self.err_y = np.zeros((n_grid)) * np.nan
        self.max_time = None
        self.min_time = None
        
    def remove_nans(self):
        ''' if the grid is too fine some windows wouldn't have any data, here we discard them'''
        ind = ~np.isnan(self.x) 
        self.x = self.x[ind]
        self.y = self.y[ind]
        self.err_x = self.err_x[ind]
        self.err_y = self.err_y[ind]
        self.conf_inter_x = self.conf_inter_x[ind]
        self.conf_inter_y = self.conf_inter_y[ind]
        
class Session:
    ''' class storing information for each video session'''
    
    def __init__(self,folder):
        self.rat_no = None
        self.path = None
        self.folder = folder
        self.fp_trial = None
        self.pad_left_x = None
        self.pad_right_x = None
        self.pad_y = None
        self.lever_x = None
        self.lever_y = None
        self.n_pad_miss_detection = None
        self.n_trials = None
        self.n_failed = None
        self.n_succeeded = None
        self.epochs_x = None
        self.epochs_y = None
        self.likelihood = None # likelihood reported by DLC
        self.pad_off_t = None # stores times when paw left the pad relative to start of trial (-1 if didn't)
        self.got_reward_t = None # stores times of lever press in each trial (-1 for failed)
        self.pad_miss_detection = None # set true for mis detected pad trials
        self.starts = None # start of each trial in the session time line
        self.ends = None # stop of each trial in the session time line
        self.failure = None # True for trials where either paw didn't leave the pad or didn't press the lever
        self.to_keep = None # False for trials to be discarded
        self.laser_trial = None # True for trials that contain laser stimulation
        self.laser_ind = None # stores the indices for laser-on time points 
        self.n_laser_trials = None
        self.laser_duration = None
        self.laser_start = None
        self.laser_pulse = None # Square continous or beta wave 
        self.laser_intensity = None
        self.laser_protocol = None # all-RT, all-MT, partial-MT 
        self.velocity_r = None # velocity 
        self.distance = None
        self.tortuosity = None
        self.MT = None # movement time
        self.steps = None # the length of steps taken throughout the trial
        self.scale_pix_to_cm = None
        
    def set_epochs(self, x, y , likelihood,cfg):
         # set the epochs provided by csv files
        n_trials = int(len(x)/cfg['fp_trial'])
        self.epochs_x = x.reshape(n_trials,cfg['fp_trial'])
        self.epochs_y = y.reshape(n_trials,cfg['fp_trial'])
        self.likelihood = likelihood.reshape(n_trials,cfg['fp_trial'])

        
    def set_properties(self, path, df, cfg):
        
        ''' find when the paw left the pad and when reached the lever
            determine failed and successful trials and trials to be discarded due to errors
        
        '''
        self.fp_trial = cfg['fp_trial']
        # read LEDs from the LED df 
        pad = np.copy(df[cfg['r_or_l']+'_pad'].values).reshape( int(len(df.index)/cfg['fp_trial']),
                                                                cfg['fp_trial'])
        reward = np.copy(df['reward'].values).reshape( int( len(df.index)/cfg['fp_trial']), 
                                                      cfg['fp_trial'])
#         print(cfg['r_or_l'])
        # coef to change pix to cm
        self.scale_pix_to_cm = cfg['pad_length_cm'] / abs(df['x'][df['point']=='pad_right'].values.copy()-
                                df['x'][df['point']=='pad_left'].values.copy()) # coef to change pix to cm
        
#         self.epochs_x = self.epochs_x*self.scale_pix_to_cm # scale into cm
#         self.epochs_y = self.epochs_y*self.scale_pix_to_cm
        
        self.n_trials = pad.shape[0]
        self.pad_off_t = np.zeros((self.n_trials),dtype = int)
        self.MT = np.zeros((self.n_trials),dtype = int)

        self.pad_left_x = ( np.full( (self.n_trials), 
                                  df['x'][ df['point'] == 'pad_left'].values.copy()) * 
                           self.scale_pix_to_cm )
        
        self.pad_right_x = ( np.full( (self.n_trials),
                                   df['x'][ df['point'] == 'pad_right'].values.copy()) * 
                            self.scale_pix_to_cm )
        self.pad_y = ( np.full( (self.n_trials), 
                             cfg['frame_height'] - df['y'][df['point'] == 'pad_left'].copy()) * 
                    self.scale_pix_to_cm )
        
        self.lever_x = ( np.full( (self.n_trials), 
                                 df['x'][df['point'] == 'lever'].copy()) *
                        self.scale_pix_to_cm )
        self.lever_y = ( np.full( (self.n_trials), 
                               cfg['frame_height'] - df['y'][df['point'] == 'lever'].copy()) * 
                        self.scale_pix_to_cm )
        self.got_reward_t = np.zeros((self.n_trials), dtype = int)
        self.starts = np.arange(0,len(df.index),cfg['fp_trial'])
        self.ends = np.arange(0,len(df.index),cfg['fp_trial'])
        self.pad_miss_detection = np.array( [ False ] * self.n_trials)
        self.failure = np.array([ False ] * self.n_trials)
        self.path = np.array( [ path ] * self.n_trials, dtype='object')
        self.rat_no = np.array([Path(path).parts[-5]]* self.n_trials, dtype='object')
        
        for i in range (self.n_trials):

            temp_reward, = np.where(reward[i,:] == 1) # times where reward LED is on
            temp_pad_on, = np.where(pad[i,:] == 1) # times where reward corresponding pad LED is on
            temp_pad_off, = np.where(pad[i,:] == 0) # times where reward corresponding pad LED is off
            
            if len(temp_reward) == 0: #----------------------# failed trial
                if len(temp_pad_on) == 0: #-------------------------# hand already left the paw
                    self.pad_off_t[i] = int(0)
                else:
                    if len(temp_pad_off) != 0:
                        self.pad_off_t[i] = int(np.min(temp_pad_off)) # first time hand left pad
                    else: self.pad_off_t[i] = int(0)
                self.got_reward_t[i]  = -1 #-----------------# coding the failures
                self.failure[i] = True
                
            else: #------------------------------------------# successful trial
                self.got_reward_t[i]  = int(np.min(temp_reward)) # earliest time paw pressed lever
                if len(temp_pad_on) == 0: #-------------------------# hand already left the paw
                    self.pad_off_t[i] = int(0)
                else:
#                     print("pad ",temp_pad_on,temp_reward)
#                     print("reward", temp_reward)
#                     print(self.starts[i])
#                     print("reward after pad ",temp_pad[temp_pad < self.got_reward_t[i]])
                    if len(temp_pad_on[temp_pad_on < self.got_reward_t[i]]) == 0:
                        self.pad_off_t[i] = int(0)
                    elif len(temp_pad_off[temp_pad_off < self.got_reward_t[i]]) == 0:#--------------------------------------# store last frame before pressing the lever the paw was on the pad
                        self.pad_off_t[i] = -1 #-----------------# code pad was wrongly activated
                        print("pad not off before reward")
                        self.pad_miss_detection[i] = True
                    else:
                        self.pad_off_t[i] = int(np.min(temp_pad_off[temp_pad_off < self.got_reward_t[i]]))

#                     print("reward ", self.got_reward_t[i])
#                     print("pad left at:" , self.pad_off_t[i])
                ## -------------- if MT is less than a threshold there was pad miss detection (or lever)
                if self.got_reward_t[i] - self.pad_off_t[i] < cfg['pad_thresh']: ###################### to do
                    print("< pad thresh", self.pad_off_t[i], self.got_reward_t[i])
                    self.pad_off_t[i] = -1 #-----------------# code pad was wrongly activated
                    self.pad_miss_detection[i] = True
       
        self.to_keep = np.invert(self.pad_miss_detection) # remove trials with pad mis detection
        self.n_failed = sum(self.failure)
        self.n_succeeded = self.n_trials - self.n_failed
        self.n_pad_miss_detection = sum(self.pad_miss_detection)
        self.MT = self.got_reward_t - self.pad_off_t # movement time
        print("to discard - pad misdetection",self.n_pad_miss_detection)
    
    def apply_pad_constraint(self,cfg):
        
        ''' whatever the situation the paw starts from the pad. Here considering from the x 
            of the first detection
        '''
        self.epochs_x = np.roll(self.epochs_x, 1, axis=1) # shift to add the pad constraint even for ones that are leaving at 0
        self.epochs_y = np.roll(self.epochs_y, 1, axis=1)
        self.epochs_y[:,0] = self.epochs_y[:,1]
        self.epochs_x[:,0] = self.epochs_x[:,1]
        self.likelihood = np.roll(self.likelihood, 1, axis=1) # shift to add the pad constraint even for ones that are leaving at 0
        self.likelihood[:,0] = 1
        discard = 0
        for i in range (self.n_trials):
            high_likelihood, = np.where(self.likelihood[i] > cfg['p_cutoff'])
            if cfg['r_or_l'] == 'L':
                above_pad, = np.where((self.epochs_x[i] > self.pad_left_x[0]) & 
                                       (self.epochs_x[i] < self.pad_right_x[0]+self.pad_right_x[0]/3))
            else:
                above_pad, = np.where((self.epochs_x[i] > (self.pad_left_x[0] - self.pad_left_x[0]/3)) & 
                                       (self.epochs_x[i] < self.pad_right_x[0]))
#             print("above pad",above_pad)
#             print("h likelihood", high_likelihood[high_likelihood > self.pad_off_t[i]])
            supposed_x_on_pad_ind = np.intersect1d(high_likelihood[high_likelihood > self.pad_off_t[i]], above_pad)
#             print("intersect",supposed_x_on_pad_ind, i)
            if len(supposed_x_on_pad_ind != 0 ):
                supposed_x_on_pad = self.epochs_x[i,supposed_x_on_pad_ind[0]]
#                 print("join pad",supposed_x_on_pad)
                width  = (self.pad_right_x[0] - self.pad_left_x[0])
                if self.epochs_y[i,supposed_x_on_pad_ind[0]] < self.pad_y[0] -width or self.epochs_y[i,supposed_x_on_pad_ind[0]] > self.pad_y[0]+width:
                    discard +=1
                    self.to_keep[i] = False 
                self.epochs_x[i,:self.pad_off_t[i]+1] = supposed_x_on_pad
                self.epochs_y[i,:self.pad_off_t[i]+1] = self.pad_y[0]
                self.likelihood[i,:self.pad_off_t[i]+1] = 1
            else: # if there are no high probability detections above the pad, this trial better be discarded
                discard +=1
                self.to_keep[i] = False 
        print("to discard - no solid detection above pad", discard)

    def mark_unreasonable_n_acc_likelihood(self,cfg):
        
        ''' mark trials that have less than n_good_points with likelihood 
            above cfg['p_cutoff'] as unusable
        '''
        
        good = np.sum(self.likelihood > cfg['p_cutoff'],axis = 1) 
        ind, = np.where(good < cfg['n_good_points_ratio']*self.fp_trial)
        self.to_keep[ind] = False
        print("to discard - low likelihoods ", len(ind))
        
    def correct_small_likelihoods(self,cfg):
        
        ''' put equidistant points from ends instead of detections with 
            likelihood less than cfg['p_cutoff']
        '''
        
        for i in range (self.n_trials):
            low_p_ind, = np.where(self.likelihood[i] < cfg['p_cutoff'])
            goods = (self.likelihood[i] > cfg['p_cutoff']).astype(int)
            good_shift = shift(goods.copy(), -1 ,cval = 1)
            bounds_start, = np.where((goods - good_shift) == 1) 
            good_shift = shift(goods.copy(), 1 ,cval = 1)
            bounds_end, = np.where(( goods - good_shift) == 1) 
            ind = np.unique(np.concatenate((bounds_start,bounds_end),0))

            for j in range(len(ind)-1):
                this = ind[j] ; next_ = ind[j+1]
                if next_-this > cfg['cor_t_prox_thresh']: # if the points are too far apart
                    continue
                p1 = np.array([self.epochs_x[i,this],self.epochs_y[i,this]])
                p2 = np.array([self.epochs_x[i,next_],self.epochs_y[i,next_]])
#                 if i == m: 
#                     print("low p", low_p_ind)
#                     print('end',bounds_end)
#                     print('start',bounds_start)
#                     print("edges",ind)
#                     print(this,next_,issubset(np.arange(ind[j]+1,ind[j+1]),low_p_ind))
                    
                if next_-this-1 > 0 and issubset(np.arange(ind[j]+1,ind[j+1]),low_p_ind):
                    x,y = equidistant_points_between(p1, p2, next_-this-1)
#                     print(points)
                    self.epochs_x[i,this+1:next_] = x
                    self.epochs_y[i,this+1:next_] = y
                    
    def mask_beginnings_and_end_of_trials(self, cfg, beginning, end):
        
        ''' mask values for before leaving the pad and after lever press '''
        
        
        successful_trials, = np.where(~self.failure)
        for i in successful_trials: # go over successfull trials
            
            if end == True: ## after reward delivary is not important --> mask it
                
                self.epochs_x[i,self.got_reward_t[i]:] = -1 ; self.epochs_y[i,self.got_reward_t[i]:] = -1
                self.velocity_r[i,self.got_reward_t[i]:] = cfg['velocity_mask'] ; self.velocity_r[i,self.got_reward_t[i]:] = cfg['velocity_mask']
                
            if beginning == True:## before pad off is not important --> mask it
                
                if self.pad_off_t[i] != 0 and self.pad_off_t[i] != -1:
                    self.epochs_x[i,:self.pad_off_t[i]] = -1 ; self.epochs_y[i,:self.pad_off_t[i]] = -1
                    self.velocity_r[i,:self.pad_off_t[i]] = cfg['velocity_mask'] ; self.velocity_r[i,:self.pad_off_t[i]] = cfg['velocity_mask']
        
    def set_laser_properties(self, df, cfg):
        ''' read laser onset and offsets from df and set the properties corresponding to laser stimulation'''
        
        laser = np.copy(df['laser'].values).reshape(int(len(df.index)/cfg['fp_trial']),cfg['fp_trial'])
        self.laser_trial = np.array([False]* self.n_trials)
        self.laser_duration = np.zeros((self.n_trials))
        self.laser_start = np.zeros((self.n_trials))
        
        self.laser_pulse = np.array([cfg['laser_pulse']]* self.n_trials, dtype='object')        
        self.laser_intensity = np.array([cfg['laser_intensity']]* self.n_trials, dtype='object')
        self.laser_protocol = np.array([cfg['laser_protocol']]* self.n_trials, dtype='object')
        for i in range (self.n_trials):

            temp_laser, = np.where(laser[i,:] == 1) # times where laser was on
            
            if len(temp_laser) == 0: #-------------------------# not a laser trial
                continue
                
            else: #--------------------------------------------#  laser trial
                self.laser_trial[i] = True 
                laser_start = min(temp_laser) ; laser_end = max(temp_laser) 
#                 if self.got_reward_t[i] < laser_end: # if the detection of reward happens before laser end 
#                     # since we know the laser tends to turn off with lever press
#                     laser_end = self.got_reward_t[i]
#                 print("discontinuity = ", sum(laser[i,laser_start:laser_end] == 0))
                self.laser_duration[i] = laser_end - laser_start
                self.laser_start[i] = laser_start

        self.n_laser_trials = sum(self.laser_trial)
    
    def ave_LED_info_based_on_laser_protocol(self, cfg):
        ''' averages over laser and pad off detections based on protocol. This is for when you are 
            sure of all LED detections equally and want to increase accuracy by averaging.
            Note: to be called after discard function.'''
        
        if self.laser_protocol[0] == "partial-MT":

            start = (self.pad_off_t[self.laser_trial]+cfg['laser_delay']+self.laser_start[self.laser_trial])/2
            self.pad_off_t[self.laser_trial] = (start-cfg['laser_delay']).astype(int); self.laser_start[self.laser_trial] = start.astype(int)
            self.pad_off_t[self.pad_off_t < 0] = 0
            self.laser_duration[self.laser_trial] = start.astype(int) + int(cfg['laser_duration']/(1000/cfg['fps'])) # assuming stimulation is applied in a specific duration
            ind = np.logical_and(self.got_reward_t < (self.laser_duration+self.laser_start),self.got_reward_t != -1) # if the detection of reward happens before laser end 
           # since we know the laser tends to turn off with lever press
            self.laser_duration[ind] = self.got_reward_t[ind]- self.laser_start[ind]
            self.MT = self.got_reward_t - self.pad_off_t
            
        if self.laser_protocol[0] == "all-MT": # laser onset aligned to pad off and laser off to reward delivery
            start = (self.pad_off_t[self.laser_trial]+ self.laser_start[self.laser_trial])/2
            end = (self.got_reward_t[self.laser_trial]+ self.laser_start[self.laser_trial]+
                            self.laser_duration[self.laser_trial])/2
            self.pad_off_t[self.laser_trial] = start.astype(int) ; self.laser_start[self.laser_trial] = start.astype(int)
            self.got_reward_t[self.laser_trial] = end.astype(int) ; self.laser_duration[self.laser_trial] = end.astype(int) - start.astype(int)
            self.MT = self.got_reward_t - self.pad_off_t
            
        if self.laser_protocol[0] == "all-RT": # leaving the pad is aligned with laser off
            pad_off = (self.pad_off_t[self.laser_trial]+ self.laser_start[self.laser_trial]+self.laser_duration[self.laser_trial])/2
            self.pad_off_t[self.laser_trial] =pad_off.astype(int); self.laser_duration[self.laser_trial] = pad_off.astype(int)-self.laser_start[self.laser_trial]
            self.MT = self.got_reward_t - self.pad_off_t
            
    def refine_pad_LED_based_on_laser_protocol(self, cfg):
        ''' since the pad detector LED is more prone to error, 
            it is corrected by the laser and reward LED and the knowlegde of the protocol. 
            Note: to be called after discard function to have no negatives for pad and reward times'''
        
        if self.laser_protocol[0] == "partial-MT":
            self.pad_off_t[self.laser_trial] = (self.laser_start[self.laser_trial]-cfg['laser_delay']).astype(int)
            self.pad_off_t[self.pad_off_t < 0] = 0
            self.laser_duration[self.laser_trial] = self.laser_start[self.laser_trial] + int(cfg['laser_duration']/(1000/cfg['fps'])) # assuming stimulation is applied in a specific duration
            ind = np.logical_and(self.got_reward_t < (self.laser_duration+self.laser_start),self.got_reward_t != -1) # if the detection of reward happens before laser end 
           # since we know the laser tends to turn off with lever press
            self.laser_duration[ind] = self.got_reward_t[ind]- self.laser_start[ind]
            self.MT = self.got_reward_t - self.pad_off_t
            
        if self.laser_protocol[0] == "all-MT": # laser onset aligned to pad off and laser off to reward delivery

            self.pad_off_t[self.laser_trial] = (self.laser_start[self.laser_trial]-cfg['laser_delay']).astype(int)
            self.pad_off_t[self.pad_off_t < 0] = 0
            end = (self.got_reward_t[self.laser_trial]+ self.laser_start[self.laser_trial]+
                            self.laser_duration[self.laser_trial])/2
            self.got_reward_t[self.laser_trial] = end.astype(int) ; self.laser_duration[self.laser_trial] = end.astype(int) - start.astype(int)
            self.MT = self.got_reward_t - self.pad_off_t
            
        if self.laser_protocol[0] == "all-RT": # leaving the pad is aligned with laser off
            self.pad_off_t[self.laser_trial] = (self.laser_start[self.laser_trial]+self.laser_duration[self.laser_trial])
            self.MT = self.got_reward_t - self.pad_off_t
            
    def calculate_velocity(self, cfg):
        
        ''' calculate velocity over cfg['n_timebin']s. 
            Note: pads the epochs on both ends to be able to calculate 
            the velocity for the boundaries as well
        '''
        
        conc_t = int(cfg['n_timebin']/2)
        r = np.sqrt(np.power(self.epochs_x,2) + np.power(self.epochs_y,2))
        shifted_forward_r = np.concatenate((np.repeat(r[:,0].reshape(-1,1),conc_t,axis = 1),
                                            r[:,:-conc_t]),axis = 1) # repeat boundaries to avoid boundary condition
        shifted_back_r = np.concatenate((r[:,conc_t:],np.repeat(r[:,-1].reshape(-1,1),conc_t,axis = 1)),axis = 1) # repeat boundaries to avoid boundary condition
        
        self.velocity_r = ( shifted_back_r - shifted_forward_r ) / (2 * conc_t / cfg['fps'] )
#         self.velocity_r = self.steps*cfg['fps']

    def calculate_steps_traveled(self, cfg):
        
        ''' clculate the d^2 = delta(x)^2+ delta(y)^2 
            and replace the unacceptable jumps with the mean
        '''
        
        x = self.epochs_x.copy()
        y = self.epochs_y.copy()
        conc_t = 1
        shifted_forward_x = np.concatenate((np.repeat(x[:,0].reshape(-1,1),conc_t,axis = 1),x),axis = 1) # repeat boundaries to avoid boundary condition
        shifted_back_x = np.concatenate((x,np.repeat(x[:,-1].reshape(-1,1),conc_t,axis = 1)),axis = 1) # repeat boundaries to avoid boundary condition
        delta_x_2 = np.power((shifted_back_x - shifted_forward_x)[:,:-1],2)
        shifted_forward_y = np.concatenate((np.repeat(y[:,0].reshape(-1,1),conc_t,axis = 1),y),axis = 1) # repeat boundaries to avoid boundary condition
        shifted_back_y = np.concatenate((y,np.repeat(y[:,-1].reshape(-1,1),conc_t,axis = 1)),axis = 1) # repeat boundaries to avoid boundary condition
        delta_y_2 = np.power((shifted_back_y - shifted_forward_y)[:,:-1],2)   
        self.steps = np.sqrt(delta_x_2 + delta_y_2)
        
        for i in range(self.steps.shape[0]):
            ind, = np.where(self.steps[i] != 0)
            
            mean_distance = np.average(self.steps[i,ind])# average over timebins where there was movement
#             ind, = np.where(self.steps[i] > cfg['max_delta_r_per_frame'])
#             if self.failure[i] == False and self.laser_trial[i] == False: to see where jitters are detected in laser sessions
#                 print("jitter ", ind-self.pad_off_t[i])
#             self.steps[i,ind] = mean_distance # replace with mean in case of crazy jumps
        self.distance = np.zeros((self.n_trials))
        failed, = np.where(self.failure)
        successful, = np.where(np.invert(self.failure))
        for i in failed:
#             ind, = np.where(self.steps[i] != 0)
#             mean_distance = np.average(self.steps[i,ind])
#             ind, = np.where(self.steps[i] > cfg['max_delta_r_per_frame'])
#             self.steps[i,ind] = mean_distance
            self.distance[i] = np.sum(self.steps[i,int(self.pad_off_t[i]):])
            x2 = np.power((self.lever_x-self.epochs_x[np.arange(self.n_trials),self.pad_off_t.astype(int)]),2)
            y2 = np.power((self.lever_y - self.pad_y),2)
            straight_trajectory_length = (x2 + y2)**0.5 # paw on pad to lever
            self.tortuosity = (self.distance / straight_trajectory_length).astype(float)
        for i in successful:
#             ind, = np.where(self.steps[i] != 0)
#             mean_distance = np.average(self.steps[i,ind])
#             ind, = np.where(self.steps[i] > cfg['max_delta_r_per_frame'])
#             self.steps[i,ind] = mean_distance
            self.distance[i] = np.sum(self.steps[i,int(self.pad_off_t[i]):int(self.got_reward_t[i])])
            x2 = np.power((self.lever_x-self.epochs_x[np.arange(self.n_trials),self.pad_off_t.astype(int)]),2)
            y2 = np.power((self.lever_y - self.pad_y),2)
            straight_trajectory_length = (x2 + y2)**0.5 # paw on pad to lever
            self.tortuosity = (self.distance / straight_trajectory_length).astype(float)
           

    def discard_unacceptable_trials(self):
        
        ''' adjust all of the session object variables after discarding the unacceptable trials
        '''
        
        print("discarded: ",self.n_trials - sum(self.to_keep))
        self.epochs_x = self.epochs_x[self.to_keep]
        self.epochs_y = self.epochs_y[self.to_keep]
        self.likelihood =self.likelihood[self.to_keep]
        self.velocity_r = self.velocity_r[self.to_keep]
        self.tortuosity = self.tortuosity[self.to_keep]
        self.distance = self.distance[self.to_keep]
        self.pad_off_t = self.pad_off_t[self.to_keep]
        self.got_reward_t = self.got_reward_t[self.to_keep]
        self.pad_miss_detection = self.pad_miss_detection[self.to_keep]
        self.starts = self.starts[self.to_keep]
        self.ends = self.ends[self.to_keep]
        self.failure = self.failure[self.to_keep]
        self.MT = self.MT[self.to_keep]
        self.path = self.path[self.to_keep]
        self.rat_no = self.rat_no[self.to_keep]
        self.steps = self.steps[self.to_keep]
        self.n_trials = len(self.starts)
        self.n_failed = sum(self.failure)
        self.n_succeeded = self.n_trials - self.n_failed
        self.n_pad_miss_detection = sum(self.pad_miss_detection)
        self.pad_left_x = self.pad_left_x[self.to_keep]
        self.pad_right_x = self.pad_right_x[self.to_keep]
        self.pad_y = self.pad_y[self.to_keep]
        self.lever_x = self.lever_x[self.to_keep]
        self.lever_y = self.lever_y[self.to_keep]
        if self.n_laser_trials != None : # if this concerns optogenetic manipulation        
            self.laser_trial = self.laser_trial[self.to_keep] 
            self.laser_duration = self.laser_duration[self.to_keep] 
            self.laser_start = self.laser_start[self.to_keep]
            self.laser_pulse = self.laser_pulse[self.to_keep] 
            self.laser_intensity = self.laser_intensity[self.to_keep]
            self.laser_protocol = self.laser_protocol[self.to_keep]
            self.n_laser_trials = sum(self.laser_trial)
        self.to_keep = self.to_keep[self.to_keep]
        
class Failed(Session):
    ''' storing only failed trials
         Note: this is the last class to be called to categorize sessions 
    '''

    def __init__(self, session):

        self.n_trials = session.n_failed
        self.path = session.path[session.failure]
        self.rat_no = session.rat_no[session.failure]
        self.folder = session.folder
        self.fp_trial = session.fp_trial
        self.pad_left_x = session.pad_left_x[session.failure]
        self.pad_right_x = session.pad_right_x[session.failure]
        self.pad_y = session.pad_y[session.failure]
        self.lever_x = session.lever_x[session.failure]
        self.lever_y = session.lever_y[session.failure]
        self.epochs_x = session.epochs_x[session.failure]
        self.epochs_y = session.epochs_y[session.failure]
        self.likelihood = session.likelihood[session.failure]
        self.velocity_r = session.velocity_r[session.failure]
        self.pad_off_t = session.pad_off_t[session.failure]
        self.got_reward_t = session.got_reward_t[session.failure]
        self.pad_miss_detection = session.pad_miss_detection[session.failure]
        self.starts = session.starts[session.failure]
        self.ends = session.ends[session.failure] 
        self.steps = session.steps[session.failure] 
        self.MT = session.MT[session.failure] 
        self.n_pad_miss_detection = sum(self.pad_miss_detection)
        self.distance = session.distance[session.failure] # the distace travelled during MT.
        self.tortuosity = session.tortuosity[session.failure]  # ratio of the traveled distance to the straight line from pad to lever
        
        if session.n_laser_trials != None : # if this concerns optogenetic manipulation
            self.laser_pulse = session.laser_pulse[session.failure] 
            self.laser_intensity = session.laser_intensity[session.failure] 
            self.laser_protocol = session.laser_protocol[session.failure] 
            self.laser_trial = session.laser_trial[session.failure] 
            self.laser_start = session.laser_start[session.failure] 
            self.laser_duration = session.laser_duration[session.failure] 
             
class Successful(Session):
    ''' storing only successful trials
        Note: this is the last class to be called to categorize sessions
    '''
    
    def __init__(self,session):

        self.n_trials = session.n_succeeded
        self.path = session.path[np.invert(session.failure)]
        self.rat_no = session.rat_no[np.invert(session.failure)]
        self.folder = session.folder 
        self.fp_trial = session.fp_trial
        self.pad_left_x = session.pad_left_x[np.invert(session.failure)]
        self.pad_right_x = session.pad_right_x[np.invert(session.failure)]
        self.pad_y = session.pad_y[np.invert(session.failure)]
        self.lever_x = session.lever_x[np.invert(session.failure)]
        self.lever_y = session.lever_y[np.invert(session.failure)]
        self.epochs_x = session.epochs_x[np.invert(session.failure)]
        self.epochs_y = session.epochs_y[np.invert(session.failure)]
        self.likelihood = session.likelihood[np.invert(session.failure)]
        self.velocity_r = session.velocity_r[np.invert(session.failure)]
        self.pad_off_t = session.pad_off_t[np.invert(session.failure)]
        self.got_reward_t = session.got_reward_t[np.invert(session.failure)]
        self.pad_miss_detection = session.pad_miss_detection[np.invert(session.failure)]
        self.starts = session.starts[np.invert(session.failure)]
        self.MT = session.MT[np.invert(session.failure)]
        self.ends = session.ends[np.invert(session.failure)] 
        self.steps = session.steps[np.invert(session.failure)] 
#         print(self.got_reward_t)
#         print(self.pad_off_t)
#         self.max_time = max(self.got_reward_t - self.pad_off_t)
#         self.min_time = min(self.got_reward_t - self.pad_off_t)
#         self.trial_no_max_time = np.argmax(self.got_reward_t - self.pad_off_t)
#         self.trial_no_min_time = np.argmin(self.got_reward_t - self.pad_off_t)
        self.distance = session.distance[np.invert(session.failure)] # the distace travelled during MT.
        self.tortuosity = session.tortuosity[np.invert(session.failure)]  # ratio of the traveled distance to the straight line from pad to lever
        
        if session.n_laser_trials != None : # if this concerns optogenetic manipulation
            self.laser_pulse = session.laser_pulse[np.invert(session.failure)] 
            self.laser_intensity = session.laser_intensity[np.invert(session.failure)] 
            self.laser_protocol = session.laser_protocol[np.invert(session.failure)] 
            self.laser_trial = session.laser_trial[np.invert(session.failure)] 
            self.laser_start = session.laser_start[np.invert(session.failure)] 
            self.laser_duration = session.laser_duration[np.invert(session.failure)] 
            self.distance_during_laser = np.zeros((len(self.laser_trial)))
              
class Laser(Session):
    ''' storing only laser trials from the session'''
    
    def __init__(self,session):
        self.laser_pulse= session.laser_pulse[session.laser_trial]
        self.laser_intensity = session.laser_intensity[session.laser_trial]
        self.laser_protocol = session.laser_protocol[session.laser_trial]
        self.n_trials = session.n_laser_trials
        self.n_laser_trials = session.n_laser_trials 
        self.path = session.path[session.laser_trial]
        self.rat_no = session.rat_no[session.laser_trial]
        self.pad_left_x = session.pad_left_x[session.laser_trial]
        self.pad_right_x = session.pad_right_x[session.laser_trial]
        self.pad_y = session.pad_y[session.laser_trial]
        self.lever_x = session.lever_x[session.laser_trial]
        self.lever_y = session.lever_y[session.laser_trial]
        self.folder = session.folder 
        self.fp_trial = session.fp_trial
        self.laser_pulse = session.laser_pulse[session.laser_trial]
        self.laser_intensity = session.laser_intensity[session.laser_trial]
        self.laser_protocol = session.laser_protocol[session.laser_trial]
        self.epochs_x = session.epochs_x[session.laser_trial]
        self.epochs_y = session.epochs_y[session.laser_trial]
        self.likelihood = session.likelihood[session.laser_trial]
        self.velocity_r = session.velocity_r[session.laser_trial]
        self.pad_off_t = session.pad_off_t[session.laser_trial]
        self.got_reward_t = session.got_reward_t[session.laser_trial]
        self.pad_miss_detection = session.pad_miss_detection[session.laser_trial]
        self.starts = session.starts[session.laser_trial]
        self.ends = session.ends[session.laser_trial]
        self.steps = session.steps[session.laser_trial]
        self.MT = session.MT[session.laser_trial]
        self.failure = session.failure[session.laser_trial]
        self.to_keep = session.to_keep[session.laser_trial]
        self.laser_duration = session.laser_duration[session.laser_trial]
        self.n_failed = sum(self.failure)
        self.n_succeeded = self.n_trials - sum(self.failure)
        self.laser_trial = session.laser_trial[session.laser_trial]
        self.laser_start = session.laser_start[session.laser_trial]
        self.distance = session.distance[session.laser_trial]
        self.tortuosity = session.tortuosity[session.laser_trial]
        self.mean_velocity_during_laser = np.zeros((self.n_laser_trials)) # mean velocity during laser stimulus

        for i in range(len(self.laser_trial)):

            self.mean_velocity_during_laser[i] = np.sum(self.steps[i,int(self.laser_start[i]):int(self.laser_start[i]
                                +self.laser_duration[i])])/self.laser_duration[i]

class Non_Laser(Session):
    ''' storing only non-laser trials from the session'''
    
    def __init__(self,session):
        self.laser_pulse = session.laser_pulse
        self.laser_intensity = session.laser_intensity
        self.laser_protocol = session.laser_protocol
        self.n_laser_trials = session.n_laser_trials
        self.n_trials =session.n_trials - session.n_laser_trials
        self.path = session.path[np.invert(session.laser_trial)]
        self.rat_no = session.rat_no[np.invert(session.laser_trial)]
        self.pad_left_x = session.pad_left_x[np.invert(session.laser_trial)]
        self.pad_right_x = session.pad_right_x[np.invert(session.laser_trial)]
        self.pad_y = session.pad_y[np.invert(session.laser_trial)]
        self.lever_x = session.lever_x[np.invert(session.laser_trial)]
        self.lever_y = session.lever_y[np.invert(session.laser_trial)]
        self.folder = session.folder 
        self.fp_trial = session.fp_trial
        self.laser_pulse = session.laser_pulse[np.invert(session.laser_trial)]
        self.laser_intensity = session.laser_intensity[np.invert(session.laser_trial)]
        self.laser_protocol = session.laser_protocol[np.invert(session.laser_trial)]
        self.epochs_x = session.epochs_x[np.invert(session.laser_trial)]
        self.epochs_y = session.epochs_y[np.invert(session.laser_trial)]
        self.likelihood = session.likelihood[np.invert(session.laser_trial)]
        self.velocity_r = session.velocity_r[np.invert(session.laser_trial)]
        self.pad_off_t = session.pad_off_t[np.invert(session.laser_trial)]
        self.got_reward_t = session.got_reward_t[np.invert(session.laser_trial)]
        self.pad_miss_detection = session.pad_miss_detection[np.invert(session.laser_trial)]
        self.starts = session.starts[np.invert(session.laser_trial)]
        self.ends = session.ends[np.invert(session.laser_trial)]
        self.steps = session.steps[np.invert(session.laser_trial)]
        self.MT = session.MT[np.invert(session.laser_trial)]
        self.failure = session.failure[np.invert(session.laser_trial)]
        self.to_keep = session.to_keep[np.invert(session.laser_trial)]
        self.laser_duration = session.laser_duration[np.invert(session.laser_trial)]
        self.n_failed = sum(self.failure)
        self.laser_trial = session.laser_trial[np.invert(session.laser_trial)]
        self.laser_start = session.laser_start[np.invert(session.laser_trial)]
        self.velocity_r = session.velocity_r[np.invert(session.laser_trial)]
        self.distance = session.distance[np.invert(session.laser_trial)]
        self.tortuosity = session.tortuosity[np.invert(session.laser_trial)]
        self.n_succeeded = self.n_trials - sum(self.failure)
#         self.distance_during_laser = np.zeros((self.n_trials))

#         for i in range(len(self.laser_trial)):
            
#             self.distance_during_laser[i] = np.sum(self.steps[i,start:end],axis = 1)/(end - start)

class All_Session(Session):
    ''' class storing information for each video session'''
    
    def __init__(self,estimated_n_trials,optogenetic_manip,fp_trial):

        self.path = np.array(['nan']*estimated_n_trials,dtype='object')
        self.rat_no = np.array(['nan']*estimated_n_trials,dtype='object')
        self.pad_left_x = np.array(['nan']*estimated_n_trials,dtype='object')
        self.pad_right_x = np.array(['nan']*estimated_n_trials,dtype='object')
        self.pad_y = np.array(['nan']*estimated_n_trials,dtype='object')
        self.lever_x = np.array(['nan']*estimated_n_trials,dtype='object')
        self.lever_y = np.array(['nan']*estimated_n_trials,dtype='object')
        self.folder = None
        self.n_pad_miss_detection = np.empty((estimated_n_trials)) * np.nan
        self.fp_trial = np.empty((estimated_n_trials)) * np.nan
        self.tortuosity = np.empty((estimated_n_trials)) * np.nan
        self.distance = np.empty((estimated_n_trials)) * np.nan
        self.epochs_x = np.empty((estimated_n_trials, fp_trial)) * np.nan
        self.epochs_y = np.empty((estimated_n_trials, fp_trial)) * np.nan
        self.likelihood = np.empty((estimated_n_trials, fp_trial)) * np.nan
        self.velocity_r = np.empty((estimated_n_trials, fp_trial)) * np.nan
        self.steps = np.empty((estimated_n_trials, fp_trial)) * np.nan

        self.pad_off_t = np.empty((estimated_n_trials)) * np.nan # stores times when paw left the pad relative to start of trial (-1 if didn't)
        self.got_reward_t = np.empty((estimated_n_trials)) * np.nan # stores times of lever press in each trial (-1 for failed)
        self.pad_miss_detection = np.array([False]* estimated_n_trials) # set true for mis detected pad trials
        self.starts = np.empty((estimated_n_trials)) * np.nan # start of each trial in the session time line
        self.ends = np.empty((estimated_n_trials)) * np.nan # stop of each trial in the session time line
        self.MT = np.empty((estimated_n_trials)) * np.nan
#         self.distance = np.empty((estimated_n_trials)) * np.nan

        self.failure = np.array([False]* estimated_n_trials) # True for trials where either paw didn't leave the pad or didn't press the lever
        self.to_keep = np.array([True]* estimated_n_trials)
        self.n_trials = None
        self.n_failed = None
        self.n_succeeded = None
        self.n_pad_miss_detection = None
        self.n_laser_trials = None
        if optogenetic_manip == True  :
            self.laser_trial = np.array([False]* estimated_n_trials) # True for trials that contain laser stimulation
            self.laser_ind = np.empty((estimated_n_trials, fp_trial)) * np.nan # stores the indices for laser-on time points 
            self.laser_duration = np.empty((estimated_n_trials)) * np.nan
            self.laser_start = np.empty((estimated_n_trials)) * np.nan
            self.laser_pulse = np.array(['nan']*estimated_n_trials,dtype='object')
            self.laser_intensity = np.array(['nan']*estimated_n_trials,dtype='object') 
            self.laser_protocol = np.array(['nan']*estimated_n_trials,dtype='object')
            

    def add_session(self,n_trials_before, n_trial_to_add, session,optogenetic_manip):
        ''' gets each session and adds it's data to this class  '''
        end = n_trials_before+ n_trial_to_add  

        self.path[n_trials_before : end] = session.path
        self.rat_no[n_trials_before : end] = session.rat_no

        self.pad_left_x[n_trials_before : end] = session.pad_left_x
        self.pad_right_x[n_trials_before : end] = session.pad_right_x
        self.pad_y[n_trials_before : end] = session.pad_y
        self.lever_x[n_trials_before : end] = session.lever_x
        self.lever_y[n_trials_before : end] = session.lever_y
        self.folder = session.folder
        self.fp_trial[n_trials_before : end] = session.fp_trial

        self.epochs_x[n_trials_before : end] = session.epochs_x
        self.epochs_y[n_trials_before : end] = session.epochs_y
        self.likelihood[n_trials_before : end] = session.likelihood
        self.velocity_r[n_trials_before : end] = session.velocity_r
        self.distance[n_trials_before : end] = session.distance
        self.tortuosity[n_trials_before : end] = session.tortuosity
        self.pad_off_t[n_trials_before : end] = session.pad_off_t
        self.got_reward_t[n_trials_before : end] = session.got_reward_t
        self.pad_miss_detection[n_trials_before : end] = session.pad_miss_detection
        self.starts[n_trials_before : end] = session.starts
        self.ends[n_trials_before : end] = session.ends
        self.steps[n_trials_before : end] = session.steps
#         self.distance[n_trials_before : end] = session.distance

        self.MT[n_trials_before : end] = session.MT
        self.failure[n_trials_before : end] = session.failure
        self.to_keep[n_trials_before : end] = session.to_keep
        self.n_pad_miss_detection = sum(self.pad_miss_detection)
        
        if optogenetic_manip == True  : # if this concerns optogenetic manipulation
            self.laser_trial[n_trials_before : end] = session.laser_trial
            self.laser_duration[n_trials_before : end]= session.laser_duration
            self.laser_start[n_trials_before : end]= session.laser_start
            self.laser_pulse[n_trials_before : end] = session.laser_pulse
            self.laser_intensity[n_trials_before : end] = session.laser_intensity
            self.laser_protocol[n_trials_before : end] = session.laser_protocol
            self.n_laser_trials = sum(self.laser_trial)


    def remove_extra_nans(self,optogenetic_manip):
        ''' variables initialisez with estimated size to avoid using append, 
        here the empty leftovers are discarded'''
    
        ind = ~np.isnan(self.pad_off_t) 
        self.path = self.path[ind] 
        self.rat_no = self.rat_no[ind] 
        self.pad_left_x = self.pad_left_x[ind] 
        self.pad_right_x = self.pad_right_x[ind] 
        self.pad_y = self.pad_y[ind] 
        self.lever_x = self.lever_x[ind] 
        self.lever_y = self.lever_y[ind] 
        self.fp_trial = self.fp_trial[ind]
        self.epochs_x = self.epochs_x[ind]
        self.epochs_y = self.epochs_y[ind]
        self.likelihood = self.likelihood[ind]

        self.velocity_r = self.velocity_r[ind]
        self.tortuosity = self.tortuosity[ind]
        self.distance = self.distance[ind]
        self.pad_off_t = self.pad_off_t[ind]
        self.got_reward_t = self.got_reward_t[ind]
        self.pad_miss_detection = self.pad_miss_detection[ind]
        self.starts = self.starts[ind]
        self.ends = self.ends[ind]
        self.MT = self.MT[ind]
        self.steps = self.steps[ind]
#         self.distance = self.distance[ind]
        self.failure = self.failure[ind]
        self.to_keep = self.to_keep[ind]
        self.n_trials = len(self.path)
        self.n_failed = sum(self.failure)
        self.n_succeeded = self.n_trials - self.n_failed
        self.n_pad_miss_detection = sum(self.pad_miss_detection)
        
        if optogenetic_manip == True  : # if this concerns optogenetic manipulation
            self.laser_trial = self.laser_trial[ind]
            self.laser_duration = self.laser_duration[ind]
            self.laser_start = self.laser_start[ind]
            self.laser_pulse = self.laser_pulse[ind]
            self.laser_intensity = self.laser_intensity[ind]
            self.laser_protocol = self.laser_protocol[ind]
            self.n_laser_trials = sum(self.laser_trial)





###################################### Functions ##############################################3



def attempt_to_make_folder(foldername):
    """ Attempts to create a folder with specified name. Does nothing if it already exists. """
    try:
        os.path.isdir(foldername)
    except TypeError:  # https://www.python.org/dev/peps/pep-0519/
        foldername = os.fspath(
            foldername
        )  # https://github.com/AlexEMG/DeepLabCut/issues/105 (cfg['window']s)

    if os.path.isdir(foldername):
        print(foldername, " already exists!")
    else:
        os.mkdir(foldername)
        
def read_pickle(filename):
    """ Read the pickle file """
    with open(filename, "rb") as handle:
        return pickle.load(handle)

def write_pickle(filename, data):
    """ Write the pickle file """
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_DLC_csv(file_name_pos,fp_trial):
    ''' Read DeepLab Cut Data and remove the <incomplete-trial> at the end'''
    
    df = pd.read_csv(file_name_pos, delimiter=",", header=[1,2])#*scale_pix_to_cm # scale to cm
    remainer = len(df.index)%fp_trial
    df = df.drop(np.arange(len(df.index)-remainer,len(df.index)))
    return df

def read_LED_csv(file_name_LED,DLC_end):
    ''' Read LED states from file and cut the <half-trial> at the end according to the DLC file size'''
    df = pd.read_csv(file_name_LED, delimiter=",")
    df = df.drop(np.arange(DLC_end,len(df.index)))
    return df

def drop_incomplete_trials(df,df_LED,fp_trial):
    ''' in case the number of read frames by the two algorithms
    is not the same the frames are dropped according to the dataframe with least number of frames'''
    
    length = min(len(df.index),len(df_LED.index))
    n_tr = len(df.index)/fp_trial
    df = df.drop(np.arange(len(df.index)-remainer,len(df.index)))

    return df,df_LED

def list_all_files(path, extension):
    
    '''get all the files with said extention in the path where you want to search 
        return full path
    '''
    files = [x for x in os.listdir(path) if not x.startswith('.')]
    files.sort()
    files_list = list(filter(lambda x: extension in x, files))

    return list(map(lambda x: os.path.join(path,x), files_list)) # join paths to file names and return

def convert_csv_to_xlsx(path):
    
    '''check if the directory has all the files in this format, if not convert to this format
        and remove the csv to make space
    '''
    files = [x for x in os.listdir(path) if not x.startswith('.')]
    files.sort()
    csv_files = list(filter(lambda x: ".csv" in x, files))
    csv_file_names = [x.replace(".csv","") for x in csv_files] # remove extensions to be able to compare lists
    xlsx_files = list(filter(lambda x: extension in x, files))
    xlsx_file_names = [x.replace(".xlsx","") for x in xlsx_files]
    if not set(csv_file_names) < set(xlsx_file_names): # if most files are in csv convert them to xlsx

        for filepath_in in csv_files:
            name = os.path.join(path, filepath_in)

            try:
                pd.read_csv(name, delimiter=",").to_excel(os.path.join(path,filepath_in.replace(".csv",".xlsx")), header = True,index = False)
            except pd.errors.ParserError: # it must be a laser file
                pd.read_csv(name, delimiter=",",skiprows= 4).to_excel(os.path.join(path,filepath_in.replace(".csv",".xlsx")), startrow=4, header = True, index = False)

            os.remove(name) # remove the csv file.
                 
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def moving_average_array(X, n):
    ''' function to return the moving average with same dimentions'''
    z2= np.cumsum(np.pad(X, (n,0), 'constant', constant_values=0))
    z1 = np.cumsum(np.pad(X, (0,n), 'constant', constant_values=X[-1]))
    return (z1-z2)[(n-1):-1]/n

def equidistant_points_between(p1, p2, n_p_between):
    x = np.linspace(p1[0], p2[0], n_p_between+1, endpoint=False)
    y = np.linspace(p1[1], p2[1], n_p_between+1, endpoint=False)
    return x[1:],y[1:]

def issubset(a, b):
    """Return whether sequence `a` is a subset of sequence `b`"""
    return len(np.setdiff1d(a, b)) == 0

def derivative(x,delta_t,cfg):
    ''' take the derivative with delta_t '''
    derivative_out = (x - shift(x, delta_t, cval= x[0]))/ (delta_t/cfg['fps'])
    return shift(derivative_out,-int(delta_t/2),cval= derivative_out[len(derivative_out)-1])

def derivative_mov_ave(x,delta_t,cfg):
    ''' take the derivative with delta_t and do a moving average over the cfg['window']'''
    derivative_out = (x - shift(x, delta_t, cval= x[0]))/ (delta_t/cfg['fps'])
    dx_dt = shift(derivative_out,-int(delta_t/2),cval= derivative_out[len(derivative_out)-1])
    return moving_average_array(dx_dt, cfg['window_veloc']) # return the moving average
#     return dx_dt # if you don't want to do a moving average 

def include_patterns(*patterns):
    """ Function that can be used as shutil.copytree() ignore parameter that
    determines which files *not* to ignore, the inverse of "normal" usage.

    This is a factory function that creates a function which can be used as a
    callable for copytree()'s ignore argument, *not* ignoring files that match
    any of the glob-style patterns provided.

    ‛patterns’ are a sequence of pattern strings used to identify the files to
    include when copying the directory tree.

    Example usage:

        copytree(src_directory, dst_directory,
                 ignore=include_patterns('*.sldasm', '*.sldprt'))
    """
    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).
        keep = (name for pattern in patterns
                        for name in fnmatch.filter(all_names, pattern))
        # Ignore file names which *didn't* match any of the patterns given that
        # aren't directory names.
        dir_names = (name for name in all_names if os.path.isdir(os.path.join(path, name)))
        return set(all_names) - set(keep) - set(dir_names)

    return _ignore_patterns

def input_plot(df, laser_t):
    ''' get the specifics of the plot as input and call the corresponding plot function '''
    
    print("Select what you want to see: \n")
    print(" 1. Tail \n 2. Nose \n 3. Fore Limb \n 4. Hind Limb")
    where_plot = [int(x)-1 for x in input().split()] # ask what body part to plot
    
    print(" 1. Position \n 2. Velocity \n 3. Acceleration")
    what_plot = int(input())-1 # ask what body part to plot
    
    print(" 1. X \n 2. Y ")
    which_plot = int(input())-1 # ask what body part to plot
    
    print(" 1. Right & Left \n 2. Average")
    Average_sep_plot = int(input()) # ask what body part to plot
    
    if Average_sep_plot == 2:
        print(Average_sep_plot)
        plot_what_which_where(df, laser_t, np.array(which_plot),np.array(what_plot),np.array(where_plot))
    else:
        plot_what_which_where_r_l(df, laser_t, np.array(which_plot),np.array(what_plot),np.array(where_plot))
        
def get_input_cor_body_part():
    ''' ask for the body part and coordinate from user'''
    
    print("Select for which parts you want to see the trial: \n")
    print(" 1. finger 0 \n 2. finger 1 joint \n 3. finger 1 base \n 4. finger 2 joint \n 5. finger 2 base")
    where_plot = [int(x)-1 for x in input().split()] # ask what body part to plot
#     print(" 1. X \n 2. Y ")
#     which_plot = int(input())-1 # ask what body part to plot
    print(" 1. Position \n 2. Velocity \n ")
    what_plot = int(input())-1 # ask what body part to plot
    return where_plot,what_plot

# def run_one_folder(rat_no, folder_list): #### not compatible anymore
    
#     ''' run data over all rats of one group and one intensity and 
#         save data of epochs and individal rats to a npz file
#     '''
    
#     plt.figure(2)
#     fig = plt.figure(figsize=(20,15))
#     nrows=2;ncols=3
#     direct = os.path.join(pre_direct, 'Rat_' + str(rat_no))  # directory to the folder for each mouse
#     count = 0
#     for folder in folder_list: # Run over all the mice
#         print(folder)
#         count +=1
#         start = timeit.default_timer()
        
#         path = os.path.join(direct, folder)
#         files_list_DLC = list_all_files(os.path.join(path, 'DLC'),'.csv')
#         files_list_LED = list_all_files(os.path.join(path, 'LED'),'.csv')

#         body_part,what_plot = [0],0
#         all_sessions = extract_epochs_over_sessions(files_list_DLC, files_list_LED, direct, folder, cfg['fp_trial'])
#         all_sessions.discard_unacceptable_trials()

#         failed = Failed(all_sessions)
#         successful = Successful(all_sessions)
#         print(" succeeded = ", successful.n_trials,"\n failed = ", failed.n_trials, "\n pad miss detections = ", session.n_pad_miss_detection)
#         print(" max trial time = ", successful.max_time/cfg['fps']*1000 ," ms", "\n min trial time = ", successful.min_time/cfg['fps']*1000 ," ms")


#         if len(files_list_DLC)==0 :
#             print("No files for ",folder)
#             continue
#         elif len(files_list_LED)==0 :
#             print("No LED detection")
#             continue
#         else:

#             ax = fig.add_subplot(nrows,ncols,count)
#             plot_mean_trajectory(all_sessions,folder)
#             stop = timeit.default_timer()
#             print('run time = ',stop-start)

#     plt.tight_layout()

#     plt.savefig(os.path.join(pre_direct,'Subplots','All_days_'+'Rat_'+str(rat_no)+'.png'),bbox_inches='tight',orientation='landscape',dpi=200)
#     plt.show()

#     save_npz(mouse_type,exp_par,folder,folder, cfg['window'],cfg['n_timebin'],"",
#              epochs_all_mice, epochs_mean_each_mouse, epochs_spont_all_mice)
# 
def run_one_folder_traj(path, rat_no, folder_list, pad_constraint = False, mask_beg_end = False):
    ''' run data over all mice of one group and one intensity and 
    save data of epochs and individal rats to a npz file'''
    fig = plt.figure(figsize=(30,20))

    nrows=3;ncols= int(len(folder_list)/nrows)+1

    count = 0
    for folder in folder_list: # Run over all the mice
        count +=1
        cfg_sample,files_list_DLC, files_list_LED = find_all_files_same_protocol_non_laser(path, folder)
        where_plot,what_plot = 0,0
        print(folder)
        body_part = cfg_sample['body_part_list'][where_plot]
        all_sessions = extract_epochs_over_sessions( files_list_DLC, files_list_LED, folder,
                                                    body_part,cfg_sample, pad_constraint= pad_constraint,
                                                    mask_beg_end = mask_beg_end)

        failed = Failed(all_sessions)
        successful = Successful(all_sessions)
        print(" succeeded = ", successful.n_trials,"\n failed = ", failed.n_trials, "\n pad miss detections = ", all_sessions.n_pad_miss_detection)
#         print(" max trial time = ", successful.max_time/cfg['fps']*1000 ," ms", "\n min trial time = ", successful.min_time/cfg['fps']*1000 ," ms")


        if len(files_list_DLC)==0 :
            print("No files for ",folder)
            continue
        elif len(files_list_LED)==0 :
            print("No LED detection")
            continue
        else:

            ax = fig.add_subplot(nrows,ncols,count)
            set_ticks(ax)
            plot_mean_trajectory(all_sessions,folder,cfg_sample)
    plt.tight_layout()

def run_one_folder_stat(path , rat_no, folder_list, pad_constraint = False , mask_beg_end = False):
    ''' run data over all mice of one group and one intensity and 
    save data of epochs and individal rats to a npz file'''
    fig = plt.figure(figsize=(30,20))

    nrows=3;ncols= int(len(folder_list)/nrows)+1

    count = 0
    for folder in folder_list: # Run over all the mice
        count +=1
        cfg_sample,files_list_DLC, files_list_LED = find_all_files_same_protocol_non_laser(path, folder)
        where_plot,what_plot = 0,0
        print(folder)
        body_part = cfg_sample['body_part_list'][where_plot]
        all_sessions = extract_epochs_over_sessions( files_list_DLC, files_list_LED, folder,
                                                    body_part,cfg_sample, pad_constraint= pad_constraint,
                                                    mask_beg_end = mask_beg_end)

        failed = Failed(all_sessions)
        successful = Successful(all_sessions)
        print(" succeeded = ", successful.n_trials,"\n failed = ", failed.n_trials, "\n pad miss detections = ", all_sessions.n_pad_miss_detection)
#         print(" max trial time = ", successful.max_time/cfg['fps']*1000 ," ms", "\n min trial time = ", successful.min_time/cfg['fps']*1000 ," ms")


        if len(files_list_DLC)==0 :
            print("No files for ",folder)
            continue
        elif len(files_list_LED)==0 :
            print("No LED detection")
            continue
        else:

            ax = fig.add_subplot(nrows,ncols,count)
            set_ticks(ax)
            plot_successful_failed(all_sessions,folder,cfg_sample)

    plt.tight_layout()

def plot_successful_failed(sessions,folder,cfg):
    x = ["successful", "failed"]
    plt.bar(x, [sessions.n_succeeded/sessions.n_trials*100 , sessions.n_failed/sessions.n_trials*100],
           color = ['g','r'])
    name= folder.split('_')
    title  = name[0]+' '+name[1]
    if len(name) > 6:
        title = title +' ' + name[2] + ' ' + name[3]
    plt.title(title,fontsize = 28)
    plt.ylabel("% trials",fontsize = 20)
    plt.xlabel("",fontsize = 20)
    plt.ylim(0,100)

def plot_trajectory(epochs_x,epochs_y,day):
#     epochs_mean_x = np.average(epochs_x, axis = 0)
#     epochs_mean_y = np.average(epochs_y, axis = 0)
    rand_epochs = np.random.randint(0,epochs_x.shape[0],50)
    for j in rand_epochs:
        ind, = np.where(epochs_x[j,:] != -1) 
        plt.plot(epochs_x[j,ind],epochs_y[j,ind], color = 'lightskyblue', linestyle='-',linewidth=1, alpha=.5)#, marker='o',markersize=1)
    #     plt.plot(np.arange(len(epochs_x[j])),epochs_x[j], color = 'lightskyblue', linestyle='-',linewidth=2, alpha=.8, marker='o',markersize=1)
    #     plt.scatter(np.arange(len(epochs_x[j])),epochs_x[j], color = 'lightskyblue')

#     plt.plot(epochs_mean_x,epochs_mean_y, color = 'k', linestyle='-',linewidth=3, alpha=1,label = "mean trajectory")#, marker='o',markersize=1)
    plt.title("Rat #"+str(rat_no)+"\n"+day).set_fontproperties(font)
    plt.ylabel("Y cm").set_fontproperties(font_label)
    plt.ylabel("X cm").set_fontproperties(font_label)
    plt.xlim(1,6)
    plt.ylim(1.5,5.5)

def correct_labeling_jitter(epochs_x, epochs_y, likelihood, cfg):
    ''' correct the single point jitters in detections exceeding the max speed of the mouse with
        the average of before and after time stamps 
        '''
    n_trials = epochs_x.shape[0]
    cfg['fp_trial'] = epochs_x.shape[1]
    xx = np.concatenate((np.repeat(epochs_x[:,0].reshape(-1,1),cfg['t_s']+1,axis = 1),epochs_x,
                         np.repeat(epochs_x[:,-1].reshape(-1,1),cfg['t_s']+1,axis = 1)),axis = 1) # repeat boundaries to avoid boundary condition
    yy = np.concatenate((np.repeat(epochs_y[:,0].reshape(-1,1),cfg['t_s']+1,axis = 1),epochs_y,
                         np.repeat(epochs_y[:,-1].reshape(-1,1),cfg['t_s']+1,axis = 1)),axis = 1) # repeat boundaries to avoid boundary condition
    likelihood = np.concatenate((np.full((n_trials,cfg['t_s']+1),1),likelihood,
                         np.full((n_trials,cfg['t_s']+1),1)),axis = 1) # repeat boundaries to avoid boundary condition

    for i in range(xx.shape[0]): # go over trials
#     for i in range(m,m+1): # go over single trial
        ind_h_lik, = np.where(likelihood[i] > cfg['p_cutoff'])
        if len(ind_h_lik) < cfg['fp_trial']*cfg['n_good_points_ratio']:
            continue
        x = np.copy(xx[i,ind_h_lik]) 
        y = np.copy(yy[i,ind_h_lik])
        
        shifted_x = shift(x, -1, cval=0) #  shifts in a periodic manner, hence the next line
        shifted_x[-1] = shifted_x[-2] # set the shifted value 
        deltas_after_x = x - shifted_x
        
        shifted_y = shift(y, -1, cval=0) # np.roll shifts in a periodic manner, hence the next line
        shifted_y[-1] = shifted_y[-2] # set the shifted value 
        deltas_after_y =  y - shifted_y 
        
        shifted_x = shift(x, 1, cval=0) #  shifts in a periodic manner, hence the next line
        shifted_x[0] = shifted_x[1] # set the shifted value 
        deltas_before_x = x - shifted_x

        shifted_y = shift(y, 1, cval=0) # np.roll shifts in a periodic manner, hence the next line
        shifted_y[0] = shifted_y[1] # set the shifted value 
        deltas_before_y =  y - shifted_y

        deltas_after = np.sqrt(np.power(deltas_after_x,2)+np.power(deltas_after_y,2))
        deltas_before = np.sqrt(np.power(deltas_before_x,2)+np.power(deltas_before_y,2))
        for j in range(1,len(ind_h_lik)-1):
            jitter_thresh_before = cfg['max_delta_r_per_frame'] * (ind_h_lik[j] - ind_h_lik[j-1])
            jitter_thresh_after = cfg['max_delta_r_per_frame'] * (ind_h_lik[j+1] - ind_h_lik[j])
            
            if deltas_before[j] > jitter_thresh_before and deltas_after[j] > jitter_thresh_after:
#                 plt.plot(x[ind_h_lik[j]],y[ind_h_lik[j]],'ro')
#                 plt.plot(x,y,'-o',c = 'g')
#                 print(ind_h_lik[j] - ind_h_lik[j-1])
#                 print(ind_h_lik[j+1] - ind_h_lik[j])
#                 xx[i,ind_h_lik[j]] = (np.average(x[j-cfg['t_s']-1:j-1])+np.average(x[j+1:j+cfg['t_s']+1]))/2
#                 yy[i,ind_h_lik[j]] = (np.average(y[j-cfg['t_s']-1:j-1])+np.average(y[j+1:j+cfg['t_s']+1]))/2
                xx[i,ind_h_lik[j]] = (x[j-1]+x[j+1])/2
                yy[i,ind_h_lik[j]] = (y[j-1]+y[j+1])/2


#             print("# jitter in DLC  = ", len(ind_row))                                     

    return xx[:, cfg['t_s'] + 1 : -( cfg['t_s'] + 1 ) ] , yy[:, cfg['t_s'] + 1 : -( cfg['t_s'] + 1 ) ]

def correct_lasting_labeling_jitter(epochs_x, epochs_y,likelihood,to_keep, cfg):
    
    ''' correct the detections exceeding the max speed of the mouse (chekcing for r vector) with
        extrapolation with a straight line. Thus enforcing corrections to be dependent on early detections.
        for early jitters t extra points are provided before time zero by 
        <set_points_around_the_pad> function.
    '''
    t = cfg['t_s']
    count = 0
    for i in range(epochs_x.shape[0]): # go over trials
#     for i in range(m,m+1): # go over single trial
        ind_h_lik, = np.where(likelihood[i] > cfg['p_cutoff'])
        x_copy = np.copy(epochs_x[i,ind_h_lik]) 
        y_copy = np.copy(epochs_y[i,ind_h_lik]) 
        if len(ind_h_lik) < cfg['fp_trial']*cfg['n_good_points_ratio']:
            continue
        shifted_x = shift((x_copy), 1, cval=0) #  shifts in a periodic manner, hence the next line
        shifted_x[0] = shifted_x[1] # set the shifted value 
        deltas_before_x = np.copy(shifted_x - x_copy)

        shifted_y = shift((y_copy), 1, cval=0) # np.roll shifts in a periodic manner, hence the next line
        shifted_y[0] = shifted_y[1] # set the shifted value 
        deltas_before_y =  np.copy(shifted_y - y_copy)
#         x,y,deltas_x,deltas_y = x_copy,y_copy,deltas_before_x, deltas_before_y
        x,y,deltas_x,deltas_y = set_points_around_the_pad(x_copy,y_copy,
                                                        deltas_before_x, deltas_before_y,t)

        rho,phi = cart2pol(deltas_x,deltas_y) # we have the displacement vector in polar coordinates
        ind, = np.where(rho > cfg['max_delta_r_per_frame']) # find the jumps
        
#         ind = ind[ind > t] # leave room for having reference before the jump
#         print("exceedings",ind)
#         print("jitters",x[ind[0]-1:ind[-1]+1])
        if len(ind) >= 2: # since we're checking for back and forths we need couples            
            for j in range (0,len(ind)-1):
#                 print("ind",ind[j],ind[j+1])
                if ind[j+1] - ind[j] > cfg['cor_t_prox_thresh']: # if the points are too far apart
                    continue
#                 print("phi", abs(abs(phi[ind[j]] - phi[ind[j+1]])-np.pi),cfg['jitter_proximity_thresh_phi'])
#                 print("rho", abs(rho[ind[j]] - rho[ind[j+1]]),cfg['jitter_proximity_thresh_rho'])
                if abs(rho[ind[j]] - rho[ind[j+1]]) < cfg['jitter_proximity_thresh_rho'] and abs(abs(phi[ind[j]] - phi[ind[j+1]])-np.pi) < cfg['jitter_proximity_thresh_phi']: 
                ## check to see if there is an approximatly same size backward jump to the correct path from this jump
#                     print("in")
                    # referece provided by average of t points before the jump
                    ref_av_x = np.sum(x[ind[j]-t:ind[j]-1])/ len((x[ind[j]-t:ind[j]-1])) 
                    ref_av_y = np.sum(y[ind[j]-t:ind[j]-1])/ len((y[ind[j]-t:ind[j]-1]))
                    n_points_to_correct = len((x[ind[j]:ind[j+1]])) # number of points included in the jump
                    # average of jumped points provided by average of points included in the jump
                    av_x = np.sum(x[ind[j]:ind[j+1]])/ n_points_to_correct
                    av_y = np.sum(y[ind[j]:ind[j+1]])/n_points_to_correct
                    
                    r_ref,theta_ref = cart2pol(ref_av_x,ref_av_y)# get a vector pointing to reference average  
                    r_av,theta_av = cart2pol(av_x,av_y)# get a vector pointing to jump average 
#                     print("Rs",r_av,r_ref)
#                     print("thetas",theta_av,theta_ref)
#                     print((n_points_to_correct*cfg['jitter_proximity_thresh_rho']*cfg['thresh_adjust']))
#                     print((n_points_to_correct*cfg['jitter_proximity_thresh_phi']*cfg['thresh_adjust']/2))

                    # since points are corrected as we go. If this is not a forth and back, the reference and jump vectors would be almost the same
                    if abs(r_av - r_ref) <  (n_points_to_correct
                    *cfg['jitter_proximity_thresh_rho']*cfg['thresh_adjust']) and abs(theta_av - theta_ref) < (n_points_to_correct
                    *cfg['jitter_proximity_thresh_phi']*cfg['thresh_adjust']/2):
                        continue # if these criteria aren't met this not a forth and back of a jump but the back and forth of two consecutive jumps
#                         print("passed")
                    p1 = np.array([x[ind[j]-1],y[ind[j]-1]])
                    p2 = np.array([x[ind[j+1]],y[ind[j+1]]])
                    correction_x,correction_y = equidistant_points_between(p1, p2, n_p_between = ind[j+1]- ind[j])# extrapolate the correction for jumped points using
                     
                    count += 1
                    ind = ind - t
#                     print("before",x_copy[ind[j]:ind[j+1]])
                    x_copy[ind[j]:ind[j+1]] = correction_x
                    y_copy[ind[j]:ind[j+1]] = correction_y
#                     print("after",x_copy[ind[j]:ind[j+1]])
        epochs_x[i,likelihood[i] > cfg['p_cutoff']] = x_copy
        epochs_y[i,likelihood[i] > cfg['p_cutoff']] = y_copy
    print("# second order jitter :", count)
    return epochs_x, epochs_y, to_keep

def set_points_around_the_pad(x, y, delta_x, delta_y, t):
    
    ''' concatenate t elements before pad-off  '''
#     print("beg",x[0])
    x = np.hstack((np.repeat(x[0],t),x)) 
    y = np.hstack((np.repeat(y[0],t),y)) 
    delta_x = np.hstack((np.zeros((t)),delta_x)) 
    delta_y = np.hstack((np.zeros((t)),delta_y)) 
    
    return x,y,delta_x,delta_y

def filter_by_likelihood(df, body_part, cfg):
    
    ''' return the indices of frames with likelihood of more than cfg['p_cutoff']'''
    likelihood = np.copy( df [ (cfg['r_or_l'] + body_part, 'likelihood' ) ].values).reshape(-1,1)
    ind_high_certainty, = np.where( likelihood > cfg['p_cutoff'] )
    return ind_high_certainty

def position(df,body_part,r_or_l,scale_pix_to_cm):
    ''' read selected body part's x,y and likelihood from the data frame '''
    
    x = np.copy(df[(r_or_l+body_part,'x')].values).reshape(-1,1)*scale_pix_to_cm
    y = np.copy(df[(r_or_l+body_part,'y')].values).reshape(-1,1)*scale_pix_to_cm
    likelihood = np.copy(df[(r_or_l+body_part,'likelihood')].values).reshape(-1,1)

    return   x,y,likelihood #moving_average_array(averaged_position, cfg['window'])

# def average_position(df,body_part,cfg['r_or_l'],cfg['fp_trial'],scale_pix_to_cm):
#     ''' average over the selected body parts '''
    
#     cfg['body_part_list'] = cfg['body_part_list'][body_part]
#     averaged_position_x = np.zeros((len(df.index),1))
#     averaged_position_y = np.zeros((len(df.index),1))

#     for param in cfg['body_part_list'] : # average over body parts
#         x = np.copy(df[(cfg['r_or_l']+param,'x')].values).reshape(-1,1)*scale_pix_to_cm

#         y = np.copy(df[(cfg['r_or_l']+param,'y')].values).reshape(-1,1)*scale_pix_to_cm

#         x, y = correct_labeling_jitter(x,y,cfg['max_delta_r_per_frame'],n_iter_jitter, cfg['t_s'],cfg['fp_trial'])
#         averaged_position_x += x ; averaged_position_y += y
#     averaged_position_x = averaged_position_x/len(cfg['body_part_list'])
#     averaged_position_y = averaged_position_y/len(cfg['body_part_list'])
#     return   averaged_position_x,averaged_position_y #moving_average_array(averaged_position, cfg['window'])
#     return 0

def set_pix_scale(df_LED,ref_length_cm):
    
    ref_length_pix = abs(df_LED['x'][df_LED['point'] == 'pad_right'].values.copy()-
                         df_LED['x'][df_LED['point'] == 'pad_left'].values.copy())
    
    scale_pix_to_cm = ref_length_cm /  ref_length_pix
    
    return scale_pix_to_cm

def extract_opto_epochs(df, df_LED, path,folder,body_part,cfg):
    
    ''' extract epochs of one session 
        return the session class containing all the info for the session
    '''

    scale_pix_to_cm = set_pix_scale(df_LED,cfg['pad_length_cm'])
#     variable_x , variable_y = average_position(df,body_part,cfg['r_or_l'],cfg['fp_trial'],scale_pix_to_cm)
    x , y, likelihood = position(df,body_part,cfg['r_or_l'],scale_pix_to_cm)
    print(cfg['frame_height'])
    y = cfg['frame_height']*scale_pix_to_cm - y # image has decreasing y instead of increasing

    session = Session(folder)
    session.set_epochs(x, y,likelihood, cfg) # reshapes to separate trials  
    session.set_properties(path,df_LED,cfg)
    session.set_laser_properties(df_LED,cfg)
#     session.refine_pad_LED_based_on_cfg['laser_protocol'](cfg['laser_delay'], cfg['laser_duration'])
#     session.ave_LED_info_based_on_cfg['laser_protocol'](cfg['laser_delay'], cfg['laser_duration'])

    session.apply_pad_constraint(cfg)
    session.mark_unreasonable_n_acc_likelihood(cfg)
    
    session.epochs_x, session.epochs_y = correct_labeling_jitter(session.epochs_x, session.epochs_y,
                                                session.likelihood,cfg)
    session.epochs_x, session.epochs_y,session.to_keep = correct_lasting_labeling_jitter(session.epochs_x, session.epochs_y,
                                                session.likelihood,session.to_keep,cfg)
    session.correct_small_likelihoods(cfg)
    session.calculate_steps_traveled(cfg) # get the steps before masking 
    session.calculate_velocity(cfg)
    session.discard_unacceptable_trials()
    session.mask_beginnings_and_end_of_trials(cfg, True, True)
    print("n trials = ",session.n_trials)
    return session

def extract_epochs(df, df_LED, path, folder, body_part, cfg, pad_constraint = True, mask_beg_end = True):
    
    ''' extract epochs of one session 
        return the session class containing all the info for the session
    '''

    scale_pix_to_cm = set_pix_scale(df_LED, cfg['pad_length_cm'])
#     variable_x , variable_y = average_position(df,body_part,cfg['r_or_l'],cfg['fp_trial'],scale_pix_to_cm)
    x , y, likelihood = position(df,body_part,cfg['r_or_l'],scale_pix_to_cm)
    y = cfg['frame_height'] * scale_pix_to_cm - y # image has decreasing y instead of increasing

    session = Session(folder)
    session.set_epochs(x, y,likelihood, cfg) # reshapes to separate trials  
    session.set_properties(path,df_LED,cfg)
    if pad_constraint :
    	session.apply_pad_constraint(cfg) # may not be applied in 6OHDA rats due to akinesia
    session.mark_unreasonable_n_acc_likelihood(cfg)
    
    session.epochs_x, session.epochs_y = correct_labeling_jitter(session.epochs_x, session.epochs_y,
                                                session.likelihood,cfg)
    session.epochs_x, session.epochs_y,session.to_keep = correct_lasting_labeling_jitter(session.epochs_x, session.epochs_y,
                                                session.likelihood,session.to_keep,cfg)
    session.correct_small_likelihoods(cfg)
    session.calculate_steps_traveled(cfg) # get the steps before masking 
    session.calculate_velocity(cfg)
    session.discard_unacceptable_trials()
    if mask_beg_end :
    	session.mask_beginnings_and_end_of_trials(cfg, True, True)
    print("n trials = ",session.n_trials)
    return session

def extract_epochs_over_sessions(files_list_DLC, files_list_LED, folder,body_part,cfg_sample, trials_each = 45, 
								pad_constraint = True, mask_beg_end = True):
    
    '''return all the epochs of all trials for one animal '''
    # trials_each = 45 estimated number of trials per session
    all_sessions = All_Session(len(files_list_DLC)* trials_each,cfg_sample['optogenetic_manip'],cfg_sample['fp_trial'])
    trial_count = 0
    for i in range(0,len(files_list_DLC)):
#         print(' number of session = {} out of {}'.format(i+1,len(files_list_DLC)))
#         print(files_list_DLC[i])
        
        configname = os.path.join(up(up(up(files_list_DLC[i]))), 'config_'+folder+'.yaml')
        cfg = read_plainconfig(configname)
        df = read_DLC_csv(files_list_DLC[i],cfg['fp_trial'])
        df_LED = read_LED_csv(files_list_LED[i],len(df.index))
#         r_or_l = r_or_l_list[i]
        path = files_list_LED[i]
        if cfg['optogenetic_manip'] == True:
            session = extract_opto_epochs(df, df_LED, path, folder, body_part, cfg)
        elif cfg['optogenetic_manip'] == False:
            session = extract_epochs(df, df_LED, path, folder, body_part, cfg, pad_constraint = pad_constraint,
            						mask_beg_end = mask_beg_end)
#         print(np.max(session.velocity_r,axis = 1))
        all_sessions.add_session(trial_count, session.n_trials, session,cfg['optogenetic_manip'])
        trial_count += session.n_trials
    all_sessions.remove_extra_nans(cfg['optogenetic_manip'])
    
    return all_sessions

# def find_mean_trajectory(cfg, all_sessions): ####### primitive version
    
#     ''' finds the mean trajectory by averaging within a 
#         each x-grid cfg['window'] and return the mean and stats in class <trials>
#     '''
    
#     epochs_x, epochs_y = all_sessions.epochs_x,all_sessions.epochs_y
#     x_grid = np.linspace(min(epochs_x[epochs_x > 0]), max(epochs_x[epochs_x > 0]), cfg['n_grid'])

#     trials = Trials(cfg['n_grid']-1)

#     for i in range(len(x_grid[:-1])):
#     #     plt.axvline(x=x_grid[i], ls='-', c='y',linewidth = 1)
#         xs = epochs_x[np.logical_and(epochs_x < x_grid[i+1], epochs_x > x_grid[i])]
#         ys = epochs_y[np.logical_and(epochs_x < x_grid[i+1], epochs_x > x_grid[i])]
#         if len(xs) !=0 : # if there are actual observations for this grid cfg['window']
#             trials.x[i] = np.average(xs)
#             trials.y[i] = np.average(ys)
#             temp_x = sms.DescrStatsW(xs).tconfint_mean(alpha=0.05, alternative='two-sided')
#             trials.conf_inter_x [i,:] = temp_x[1] - trials.x[i], trials.x[i]-temp_x[0]
#             temp_y = sms.DescrStatsW(xs).tconfint_mean(alpha=0.05, alternative='two-sided')
#             trials.conf_inter_y [i,:] = temp_y[1] - trials.y[i], trials.y[i]-temp_y[0]

#             trials.err_y[i] = np.std(ys)
#             trials.err_x[i] = np.std(xs)

# #             trials.err_y[i] = stats.sem(ys)
# #             trials.err_x[i] = stats.sem(xs)

#     trials.remove_nans()
#     return trials

def find_mean_trajectory(cfg, all_sessions):  #### taken from the 6OHDA notebook
    ''' finds the mean trajectory by averaging within a 
    each x-grid cfg['window'] and return the mean and stats in class <trials>'''
    epochs_x, epochs_y = all_sessions.epochs_x,all_sessions.epochs_y
    ep_x = epochs_x[epochs_x > 0]
    if len(ep_x) == 0:
        print("no data")
        min_x = 0 # set borders of the exp chamber
        max_x = 7
    else:
        min_x = min(ep_x)
        max_x = max(ep_x)
    x_grid = np.linspace( min_x, max_x , cfg['n_grid'])

    trials = Trials(cfg['n_grid']-1)

    for i in range(len(x_grid[:-1])):
    #     plt.axvline(x=x_grid[i], ls='-', c='y',linewidth = 1)
        xs = epochs_x[np.logical_and(epochs_x < x_grid[i+1], epochs_x > x_grid[i])]
        ys = epochs_y[np.logical_and(epochs_x < x_grid[i+1], epochs_x > x_grid[i])]
        if len(xs) !=0 : # if there are actual observations for this grid cfg['window']
            trials.x[i] = np.average(xs)
            trials.y[i] = np.average(ys)
            temp_x = sms.DescrStatsW(xs).tconfint_mean(alpha=0.05, alternative='two-sided')
            trials.conf_inter_x [i,:] = temp_x[1] - trials.x[i], trials.x[i]-temp_x[0]
            temp_y = sms.DescrStatsW(xs).tconfint_mean(alpha=0.05, alternative='two-sided')
            trials.conf_inter_y [i,:] = temp_y[1] - trials.y[i], trials.y[i]-temp_y[0]

            trials.err_y[i] = np.std(ys)
            trials.err_x[i] = np.std(xs)

#             trials.err_y[i] = stats.sem(ys)
#             trials.err_x[i] = stats.sem(xs)

    trials.remove_nans()
    return trials

    
def plot_mean_trajectory(session,folder,cfg):
    trials = find_mean_trajectory(cfg, session)
    plt.errorbar(trials.x,trials.y,  trials.err_y, trials.err_x,  marker = 'o',
             markersize=5, linewidth=2, capsize=5, capthick=1, color = 'navy', ecolor='powderblue')
    plt.plot([session.lever_x[0] - 0.1, session.lever_x[0] + 0.1],[session.lever_y[0],session.lever_y[0]],lw = 5, c = 'grey')
    plt.plot([session.pad_left_x[0],session.pad_right_x[0]],[session.pad_y[0],session.pad_y[0]],lw = 5, c = 'grey')
    name= folder.split('_')
    title  = name[0]+' '+name[1]
    if len(name) > 6:
        title = title +' ' + name[2] + ' ' + name[3]
    plt.title(title,fontsize = 28)
    plt.ylabel("Y (cm)",fontsize = 20)
    plt.xlabel("X (cm)",fontsize = 20)
    plt.legend(fontsize = 10)
    plt.xlim(0,7.5)
    plt.ylim(0,6)
  

def plot_laser_trajectory(pre_direct, rat_no, n,files_list_DLC,files_list_LED,path,folder,cfg, font, font_label):
    df, df_LED = get_DLC_LED_df(files_list_DLC, files_list_LED, n,cfg)
    i = 0
    j = 0
    where_plot= 0
    body_part = cfg['body_part_list'][where_plot]
    session = extract_opto_epochs(df,df_LED,path,folder,body_part,cfg)
    laser = Laser(session)
    failed = Failed(laser)
    successful = Successful(laser)
    print(" succeeded = ", successful.n_trials,"\n failed = ", failed.n_trials, "\n pad miss detections = ", session.n_pad_miss_detection)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
    ax[0] = plt.subplot(121)
#     for j in range (m,m+1):
    for j in range (0,successful.n_trials):
        overall_ind, = np.where(successful.epochs_x[j,:] != -1 )

#         alphas = np.linspace(0.1, 1, len(overall_ind))
#         rgba_colors = np.zeros((len(overall_ind),4))
#         rgba_colors[:,:-1] = np.random.random((3))
#         rgba_colors[:, 3] = alphas
#         plt.scatter(successful.epochs_x[j,overall_ind],successful.epochs_y[j,overall_ind], color = rgba_colors )
#         plt.plot(successful.epochs_x[j,overall_ind],successful.epochs_y[j,overall_ind],alpha = 0.2, color = rgba_colors[0,:-1] )
        ind_bef = overall_ind[overall_ind <= successful.laser_start[j]]
        ind_aft = overall_ind[overall_ind >= (successful.laser_start[j]+successful.laser_duration[j])]
        plt.plot(successful.epochs_x[j,ind_bef],successful.epochs_y[j,ind_bef],'-o', color = 'k',alpha = 0.6 , markersize = 4)
        plt.plot(successful.epochs_x[j,ind_aft],successful.epochs_y[j,ind_aft],'-o', color = 'k',alpha = 0.6 , markersize = 4)
        # to doo
        ind = np.arange(successful.laser_duration[j],dtype = int)+int(successful.laser_start[j])
        ind = ind[ind >= successful.pad_off_t[j]]
        plt.plot(successful.epochs_x[j,ind],successful.epochs_y[j,ind],'-o', c = 'deepskyblue', alpha = 1, markersize = 4 )

    plt.plot(successful.epochs_x[j,ind],successful.epochs_y[j,ind],'-o', c = 'deepskyblue', alpha = 1, markersize = 4,label= "Laser-ON" )
    plt.plot([session.lever_x[0] - 0.1, session.lever_x[0] + 0.1],[session.lever_y[0],session.lever_y[0]],lw = 5, c = 'grey')
    plt.plot([session.pad_left_x[0],session.pad_right_x[0]],[session.pad_y[0],session.pad_y[0]],lw = 5, c = 'grey')
    plt.legend(fontsize = 20)
    plt.title("Rat #"+str(rat_no)+" "+folder+"\n Laser session : "+str(n+1)+"\n (Successful)").set_fontproperties(font)
    plt.ylabel("Y (cm)").set_fontproperties(font_label)
    plt.xlabel("X (cm)").set_fontproperties(font_label)
    plt.xlim(1,6.5)
    plt.ylim(0,6)
    set_ticks(ax[0])
    ax[1] = plt.subplot(122)

    for i in range (0,failed.n_trials):
#         failed.epochs_x[i,failed.likelihood[i] < cfg['p_cutoff']] = -1
        overall_ind, = np.where(failed.epochs_x[i,:] != -1 )
        ind = np.arange(failed.laser_duration[i],dtype = int)+int(failed.laser_start[i]) 
        
#         alphas = np.linspace(0.1, 1, len(overall_ind))
#         rgba_colors = np.zeros((len(overall_ind),4))
#         rgba_colors[:,:-1] = np.random.random((3))
#         rgba_colors[:, 3] = alphas
#         plt.scatter(failed.epochs_x[i,overall_ind],failed.epochs_y[i,overall_ind], color = rgba_colors )
#         plt.plot(failed.epochs_x[i,ind],failed.epochs_y[i,ind], c = 'navy', alpha = 0.5 )
#         plt.plot(failed.epochs_x[i,overall_ind],failed.epochs_y[i,overall_ind],alpha = 0.2, color = rgba_colors[0,:-1] )
        
        ind_bef = overall_ind[overall_ind <= failed.laser_start[i]]
        ind_aft = overall_ind[overall_ind >= (failed.laser_start[i]+failed.laser_duration[i])]
        plt.plot(failed.epochs_x[i,ind_bef],failed.epochs_y[i,ind_bef],'-o', color = 'k',alpha = 0.6 , markersize = 4)
        plt.plot(failed.epochs_x[i,ind_aft],failed.epochs_y[i,ind_aft],'-o', color = 'k',alpha = 0.6 , markersize = 4)

    if failed.n_trials > 0:
#         plt.plot(failed.epochs_x[i,ind],failed.epochs_y[i,ind], c = 'navy', alpha = 0.1, label= "Laser-ON" )
        plt.plot(failed.epochs_x[i,ind],failed.epochs_y[i,ind],'-o', c = 'deepskyblue', alpha = 1, markersize = 4, label= "Laser-ON" )

    plt.plot([session.lever_x[0] - 0.1, session.lever_x[0] + 0.1],[session.lever_y[0],session.lever_y[0]],lw = 5, c = 'grey')
    plt.plot([session.pad_left_x[0],session.pad_right_x[0]],[session.pad_y[0],session.pad_y[0]],lw = 5, c = 'grey')
    plt.legend(fontsize = 20)
    plt.title("Rat #"+str(rat_no)+" "+folder+"\n Laser session : "+str(n+1)+'\n (Failed)').set_fontproperties(font)
    plt.ylabel("Y (cm)").set_fontproperties(font_label)
    plt.xlabel("X (cm)").set_fontproperties(font_label)
    plt.xlim(1,6.5)
    plt.ylim(0,6)
    set_ticks(ax[1])
    plt.savefig(os.path.join(pre_direct, 'Subplots', 'Rat_'+str(rat_no)+'_'+folder+'_session = '+str(n+1)+
                 '.png'),bbox_inches='tight',orientation='landscape',dpi=200)

def plot_laser_velocity(pre_direct, rat_no, n,files_list_DLC,files_list_LED,path,folder,cfg, font, font_label):


    df, df_LED = get_DLC_LED_df(files_list_DLC, files_list_LED, n,cfg)

    i = 0
    j = 0
    where_plot= 0
    body_part = cfg['body_part_list'][where_plot]
    session = extract_opto_epochs(df,df_LED,path,folder,body_part,cfg)  
    laser = Laser(session)
    failed = Failed(laser)
    successful = Successful(laser)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), sharey=True)
    for j in range (0,successful.n_trials):
        overall_ind, = np.where(successful.velocity_r[j,:] != cfg['velocity_mask'] )
        if successful.laser_start[j] < successful.pad_off_t[j]:
            laser_start = successful.pad_off_t[j]
        else:
            laser_start = successful.laser_start[j]
        if successful.laser_duration[j]+successful.laser_start[j] > successful.got_reward_t[j]:
            laser_end = successful.got_reward_t[j]
        else: 
            laser_end = successful.laser_duration[j] + successful.laser_start[j]
        ind_bef = overall_ind[overall_ind <= laser_start]
        ind_aft = overall_ind[(overall_ind >= laser_end)]
        time_series_bef = np.arange(laser_start-successful.pad_off_t[j] + 1 ) / cfg['fps']
        time_series_aft = ( np.arange(laser_end, overall_ind[-1] + 1 ) / cfg['fps'] - 
                                    successful.pad_off_t[j] / cfg['fps'] )
        try:
            ax.plot(time_series_bef,successful.velocity_r[j,ind_bef],'-o', color = 'k',alpha = 0.6 , 
                     markersize = 4)
            ax.plot(time_series_aft,successful.velocity_r[j,ind_aft],'-o', color = 'k',alpha = 0.6 , markersize = 4)
        
        except ValueError as err:
            print(err, "Velocities could not be computed, cannot be plotted for trial ")
        ind = np.arange(laser_start,laser_end,dtype = int)
        
        if len(ind) != 0: # it could be possible that the hand doesn't leave the pad during laser
            time_series_laser = (np.arange(laser_start,laser_end)-successful.pad_off_t[j])/cfg['fps']
            ax.plot(time_series_laser,successful.velocity_r[j,ind],'-o', c = 'deepskyblue', 
                     alpha = 1 , markersize = 4)

    ax.set_title("Rat #" + str(rat_no)+ " " + folder + "\n Laser session : " 
              + str(n+1) + "\n (Successful)").set_fontproperties(font)
    
    ax.set_ylabel("V (cm/s)").set_fontproperties(font_label)
    ax.set_xlabel("t (s)").set_fontproperties(font_label)
    black_patch = mpatches.Patch(color='k', label='Laser-OFF')
    blue_patch = mpatches.Patch(color='deepskyblue', label='Laser-ON')
    ax.legend(handles=[black_patch, blue_patch], fontsize = 20)
    ax.set_xlim(-0.02,1.)
    ax.set_ylim(-45,80)
    set_ticks(ax)
    fig.savefig(os.path.join(pre_direct, 'Subplots','Velocity_Rat_' + str(rat_no)+ 
                             '_'+folder+'_session = '+str(n+1)+'.png'),
                bbox_inches='tight',orientation='landscape',dpi=200)

def plot_non_laser_trajectory(pre_direct, rat_no, n,files_list_DLC,files_list_LED,path,folder,cfg, font, font_label):
    df, df_LED = get_DLC_LED_df(files_list_DLC, files_list_LED, n,cfg)
    i = 0
    j = 0
    where_plot= 0
    body_part = cfg['body_part_list'][where_plot]
    session = extract_opto_epochs(df,df_LED,path,folder,body_part,cfg)
    non_laser = Non_Laser(session)
    failed = Failed(non_laser)
    successful = Successful(non_laser)
    print(" succeeded = ", successful.n_trials,"\n failed = ", failed.n_trials, "\n pad miss detections = ", session.n_pad_miss_detection)
    plt.figure(1)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
    ax[0] = plt.subplot(121)
    for j in range (0,successful.n_trials):
#     for j in range (m,m+1):
#         successful.epochs_x[j,successful.likelihood[j] < cfg['p_cutoff']] = -1
        overall_ind, = np.where(successful.epochs_x[j,:] != -1 )
        ind = np.arange(successful.laser_duration[j],dtype = int)+int(successful.laser_start[j])

#         alphas = np.linspace(0.1, 1, len(overall_ind))
#         rgba_colors = np.zeros((len(overall_ind),4))
#         rgba_colors[:,:-1] = np.random.random((3))
#         rgba_colors[:, 3] = alphas
#         plt.scatter(successful.epochs_x[j,overall_ind],successful.epochs_y[j,overall_ind], color = rgba_colors )
#         plt.plot(successful.epochs_x[j,overall_ind],successful.epochs_y[j,overall_ind],alpha = 0.2, color = rgba_colors[0,:-1] )

        plt.plot(successful.epochs_x[j,overall_ind],successful.epochs_y[j,overall_ind],'-o', color = 'k',alpha = 0.6 , markersize = 4,label = 'traveled path')

    plt.plot([session.lever_x[0] - 0.1, session.lever_x[0] + 0.1],[session.lever_y[0],session.lever_y[0]],lw = 5, c = 'grey')
    plt.plot([session.pad_left_x[0],session.pad_right_x[0]],[session.pad_y[0],session.pad_y[0]],lw = 5, c = 'grey')
#     plt.plot([successful.epochs_x[j,overall_ind[0]],session.lever_x[0]],[session.pad_y[0],session.lever_y[0]],'--',
#              lw = 3, c = 'lightskyblue',label = 'straight line')

    plt.title("Rat #"+str(rat_no)+" "+folder+"\n non Laser session : "+str(n+1)+"\n (Successful)").set_fontproperties(font)
    plt.ylabel("Y (cm)").set_fontproperties(font_label)
    plt.xlabel("X (cm)").set_fontproperties(font_label)
#     plt.legend(fontsize = 20)
#     plt.xlim(1,6.5)
#     plt.ylim(0,6)
    set_ticks(ax[0])
    ax[1] = plt.subplot(122)

    for i in range (0,failed.n_trials):
#         failed.epochs_x[i,failed.likelihood[i] < cfg['p_cutoff']] = -1
        overall_ind, = np.where(failed.epochs_x[i,:] != -1 )
#         ind = np.arange(failed.cfg['laser_duration'][i],dtype = int)+int(failed.laser_start[i]) 
#         alphas = np.linspace(0.1, 1, len(overall_ind))
#         rgba_colors = np.zeros((len(overall_ind),4))
#         rgba_colors[:,:-1] = np.random.random((3))
#         rgba_colors[:, 3] = alphas
#         plt.scatter(failed.epochs_x[i,overall_ind],failed.epochs_y[i,overall_ind], color = rgba_colors )
#         plt.plot(failed.epochs_x[i,overall_ind],failed.epochs_y[i,overall_ind],alpha = 0.2, color = rgba_colors[0,:-1] )
        plt.plot(failed.epochs_x[i,overall_ind],failed.epochs_y[i,overall_ind],'-o', color = 'k',alpha = 0.6 , markersize = 4)
    
    plt.plot([session.lever_x[0] - 0.1, session.lever_x[0] + 0.1],[session.lever_y[0],session.lever_y[0]],lw = 5, c = 'grey')
    plt.plot([session.pad_left_x[0],session.pad_right_x[0]],[session.pad_y[0],session.pad_y[0]],lw = 5, c = 'grey')

    plt.title("Rat #"+str(rat_no)+" "+folder+"\n non Laser session : "+str(n+1)+'\n (Failed)').set_fontproperties(font)
    plt.ylabel("Y (cm)").set_fontproperties(font_label)
    plt.xlabel("X (cm)").set_fontproperties(font_label)
#     plt.xlim(1,6.5)
#     plt.ylim(0,6)
    set_ticks(ax[1])
    plt.savefig(os.path.join(pre_direct,'Subplots', 'Rat_'+str(rat_no)+'_'+folder+'_non_laser_session = '+str(n+1)+
                 '.png'),bbox_inches='tight',orientation='landscape',dpi=200)

    
    return session


def plot_6OHDA_trajectory_separate_colors(pre_direct, rat_no, n,files_list_DLC, files_list_LED,path,folder
										,cfg, font, font_label,  pad_constraint = False, mask_beg_end = False):
    df, df_LED = get_DLC_LED_df(files_list_DLC, files_list_LED, n,cfg)
    i = 0
    j = 0
    where_plot= 0
    body_part = cfg['body_part_list'][where_plot]
    sessions = extract_epochs_over_sessions( files_list_DLC, files_list_LED, folder,
                                                    body_part,cfg, pad_constraint= pad_constraint,
                                                    mask_beg_end = mask_beg_end)
    failed = Failed(sessions)
    successful = Successful(sessions)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
    ax[0] = plt.subplot(121)
    for j in range (0,successful.n_trials):
#     for j in range (m,m+1):
#         successful.epochs_x[j,successful.likelihood[j] < cfg['p_cutoff']] = -1
        overall_ind, = np.where(successful.epochs_x[j,:] != -1 )
        # ind = np.arange(successful.laser_duration[j],dtype = int)+int(successful.laser_start[j])

        alphas = np.linspace(0.1, 1, len(overall_ind))
        rgba_colors = np.zeros((len(overall_ind),4))
        rgba_colors[:,:-1] = np.random.random((3))
        rgba_colors[:, 3] = alphas
        plt.scatter(successful.epochs_x[j,overall_ind],successful.epochs_y[j,overall_ind], color = rgba_colors )
        plt.plot(successful.epochs_x[j,overall_ind],successful.epochs_y[j,overall_ind],alpha = 0.2, color = rgba_colors[0,:-1] )


    plt.plot([sessions.lever_x[0] - 0.1, sessions.lever_x[0] + 0.1],[sessions.lever_y[0],sessions.lever_y[0]],lw = 5, c = 'grey')
    plt.plot([sessions.pad_left_x[0],sessions.pad_right_x[0]],[sessions.pad_y[0],sessions.pad_y[0]],lw = 5, c = 'grey')
#     plt.plot([successful.epochs_x[j,overall_ind[0]],session.lever_x[0]],[session.pad_y[0],session.lever_y[0]],'--',
#              lw = 3, c = 'lightskyblue',label = 'straight line')

    plt.title("Rat #"+str(rat_no)+" "+folder+"\n non Laser session : "+str(n+1)+"\n (Successful)").set_fontproperties(font)
    plt.ylabel("Y (cm)").set_fontproperties(font_label)
    plt.xlabel("X (cm)").set_fontproperties(font_label)
#     plt.legend(fontsize = 20)
#     plt.xlim(1,6.5)
#     plt.ylim(0,6)
    set_ticks(ax[0])
    ax[1] = plt.subplot(122)

    for i in range (0,failed.n_trials):
#         failed.epochs_x[i,failed.likelihood[i] < cfg['p_cutoff']] = -1
        overall_ind, = np.where(failed.epochs_x[i,:] != -1 )
#         ind = np.arange(failed.cfg['laser_duration'][i],dtype = int)+int(failed.laser_start[i]) 
        alphas = np.linspace(0.1, 1, len(overall_ind))
        rgba_colors = np.zeros((len(overall_ind),4))
        rgba_colors[:,:-1] = np.random.random((3))
        rgba_colors[:, 3] = alphas
        plt.scatter(failed.epochs_x[i,overall_ind],failed.epochs_y[i,overall_ind], color = rgba_colors )
        plt.plot(failed.epochs_x[i,overall_ind],failed.epochs_y[i,overall_ind],alpha = 0.2, color = rgba_colors[0,:-1] )
    
    plt.plot([sessions.lever_x[0] - 0.1, sessions.lever_x[0] + 0.1],[sessions.lever_y[0],sessions.lever_y[0]],lw = 5, c = 'grey')
    plt.plot([sessions.pad_left_x[0],sessions.pad_right_x[0]],[sessions.pad_y[0],sessions.pad_y[0]],lw = 5, c = 'grey')

    plt.title("Rat #"+str(rat_no)+" "+folder+"\n non Laser session : "+str(n+1)+'\n (Failed)').set_fontproperties(font)
    plt.ylabel("Y (cm)").set_fontproperties(font_label)
    plt.xlabel("X (cm)").set_fontproperties(font_label)
#     plt.xlim(1,6.5)
#     plt.ylim(0,6)
    set_ticks(ax[1])
    plt.savefig(os.path.join(pre_direct,'Subplots', 'Rat_'+str(rat_no)+'_'+folder+'_non_laser_session = '+str(n+1)+
                 '.png'),bbox_inches='tight',orientation='landscape',dpi=200)

    

def plot_non_laser_velocity(pre_direct, rat_no, n,files_list_DLC,files_list_LED,path,folder,cfg, font, font_label):
    df, df_LED = get_DLC_LED_df(files_list_DLC, files_list_LED, n,cfg)

    i = 0
    j = 0
    where_plot= 0
    body_part = cfg['body_part_list'][where_plot]
    session = extract_opto_epochs(df,df_LED,path,folder,body_part,cfg)  

    laser = Non_Laser(session)
    failed = Failed(laser)
    successful = Successful(laser)
    plt.figure(1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), sharey=True)
    for j in range (0,successful.n_trials):
        overall_ind, = np.where(successful.velocity_r[j,:] != cfg['velocity_mask'] )
        time_series_overall = np.arange(len(overall_ind))/cfg['fps']
        plt.plot(time_series_overall,successful.velocity_r[j,overall_ind],'-o', color = 'k',alpha = 0.4, markersize = 3)

    plt.title("Rat #"+str(rat_no)+" "+folder+"\n Non-Laser session : "+str(n+1)+"\n (Successful)").set_fontproperties(font)
    plt.ylabel("V (cm/s)").set_fontproperties(font_label)
    plt.xlabel("t (s)").set_fontproperties(font_label)
    plt.legend(fontsize = 10)
    plt.xlim(-0.02,1.)
    plt.ylim(-45,80)
    set_ticks(ax)
    plt.savefig(os.path.join(pre_direct, 'Subplots', 'Velocity_Rat_'+str(rat_no)+'_'+folder+'_Non-Laser_session = '+str(n+1)+
                     '.png'),bbox_inches='tight',orientation='landscape',dpi=200)
    
def find_all_files_same_protocol(direct, protocol_name):
    
    ''' gets folder as input and finds all sessions of a certain protocol among different animals.
        returns an array specifying L or R handedness together with DLC and LED file paths 
    '''
    files_list_DLC = []
    files_list_LED = []
    r_or_l_list = []
    
    for dirpath, dirnames, filenames in os.walk(direct):
        for dirname in dirnames:
            if dirname == protocol_name:
                
                prop = protocol_name.split('_')
                path = os.path.join(dirpath, protocol_name)
                R_L_folder = [ f.name for f in os.scandir(path) if f.is_dir() ][0] # gives the Right or Left
                
                config_sup_info = { 
                                    'optogenetic_manip': True , 
                                    'exp_par': os.path.basename( os.path.normpath( direct ) ) ,
                                    'r_or_l': R_L_folder[0] ,
                                    'fp_trial': int( prop[-1] ) , 
                                    'laser_protocol': prop[3] , 
                                    'laser_pulse': prop[0] ,
                                    'laser_intensity': prop[1]
                                  } 
                
                configname = set_config_file(protocol_name,dirpath)
                edit_config(configname, config_sup_info)
                attempt_to_make_folder( os.path.join(path, R_L_folder, 'Plots') )
                
                files_list_DLC += list_all_files(os.path.join(path, R_L_folder, 'DLC'),'.csv')
                files_list_LED += (list_all_files(os.path.join(path, R_L_folder, 'LED'),'.csv'))
#     print("number of sessionsf for "+ protocol_name + "protocol " , len(files_list_DLC))
    cfg_sample = read_plainconfig(configname)
    return  cfg_sample, files_list_DLC, files_list_LED


def find_all_files_same_protocol_non_laser(direct, protocol_name):
    ''' gets folder as input and finds all sessions of a certain protocol (non-laser) among different animals.
        returns an array specifying L or R handedness together with DLC and LED file paths '''
    files_list_DLC = []
    files_list_LED = []
    r_or_l_list = []
    for dirpath, dirnames, filenames in os.walk(direct):
        for dirname in dirnames:
            if dirname == protocol_name:
                path = os.path.join(dirpath, protocol_name)
                prop = protocol_name.split('_')
                R_L_folder = [ f.name for f in os.scandir(path) if f.is_dir() ][0] # gives the Right or Left
                config_sup_info = { 'optogenetic_manip': False, 
                                    'exp_par': os.path.basename(os.path.normpath(direct)),
                                    'r_or_l': R_L_folder[0],
                                    'fp_trial': int(prop[-1]),
                                    'fps': int(prop[-3]),
                                    'pad_thresh': 0} 
                configname = set_config_file(protocol_name,dirpath)
                edit_config(configname, config_sup_info)
                attempt_to_make_folder(os.path.join(path,R_L_folder, 'Plots'))
                
                files_list_DLC += list_all_files(os.path.join(path, R_L_folder, 'DLC'),'.csv')
                files_list_LED += (list_all_files(os.path.join(path, R_L_folder, 'LED'),'.csv'))
#     print("number of sessionsf for "+ protocol_name + "protocol " , len(files_list_DLC))
    cfg_sample = read_plainconfig(configname)
    return  cfg_sample, files_list_DLC, files_list_LED


def group_and_av_animals(result,y):

    rat_key = np.unique(result['rat_no'])
    grouped_ave_var = np.empty((len(rat_key),2))
    
    for c, rat_no in enumerate(rat_key):
        
        grouped_ave_var[c,0] = np.average(result[(result['rat_no'] == rat_no) & (result['laser'] == 'laser')][y])
        grouped_ave_var[c,1] = np.average(result[(result['rat_no'] == rat_no) & (result['laser'] == 'no laser')][y])
    
    return grouped_ave_var

# def build_sessions_transf_to_df(pre_direct, folder,exp_par):
    
#     ''' build a class object from all sessions with <folder> protocol
#         separate laser/non-laser and successful/failed trials
#         put important measurements into a df and return
#     '''

#     pre_direct_chosen = os.path.join(pre_direct, exp_par)


#     cfg, files_list_DLC, files_list_LED = find_all_files_same_protocol(pre_direct_chosen, folder)
#     where_plot,what_plot = 0,0
#     body_part = cfg['body_part_list'][where_plot]
#     all_sessions = extract_epochs_over_sessions(files_list_DLC, files_list_LED, folder,body_part,cfg)
#     laser = Laser(all_sessions)
#     laser_failed = Failed(laser)
#     laser_successful = Successful(laser)

#     non_laser = Non_Laser(all_sessions)
#     normal_failed = Failed(non_laser)
#     normal_successful = Successful(non_laser)
    
#     n1 = laser_successful.n_trials 
#     n2 = laser_failed.n_trials
#     n3 = normal_successful.n_trials
#     n4 = normal_failed.n_trials 
    
#     print("laser success trials = ",laser_successful.n_trials)
#     print("laser failed trials = ",laser_failed.n_trials)

#     print("normal success trials = ",normal_successful.n_trials)
#     print("normal failed trials = ",normal_failed.n_trials)


#     col_names =  ['rat_no','opto_par','pulse','intensity','protocol','Nfpt','RT(ms)','MT(ms)', 'tortuosity','distance(cm)','v_max(cm/s)', 'trial', 'laser']
    
#     result = pd.DataFrame(columns = col_names)
    
#     df = pd.DataFrame(({'rat_no':laser_successful.rat_no, 
#     					'opto_par':[exp_par] * n1, 
#     					'pulse':[cfg['laser_pulse']] * n1, 
#         				'intensity':[cfg['laser_intensity']] * n1,
#         				'protocol':[cfg['laser_protocol']] * n1,
#         				'Nfpt':[cfg['fp_trial']] * n1,
#         				'RT(ms)':laser_successful.pad_off_t, 
#         				'distance(cm)':laser_successful.distance,
#         				'tortuosity':laser_successful.tortuosity,
#         				'MT(ms)':laser_successful.got_reward_t - laser_successful.pad_off_t,
#         				'v_max(cm/s)':np.max(laser_successful.velocity_r,axis=1),
#         				'trial':['successful'] * n1 , 
#         				'laser':['laser'] * n1 }))
    
#     df1 = pd.DataFrame(({'rat_no':laser_failed.rat_no, 'opto_par':[exp_par]*n2, 'pulse':[cfg['laser_pulse']]*n2, 
#         'intensity':[cfg['laser_intensity']]*n2,'protocol':[cfg['laser_protocol']]*n2,'Nfpt':[cfg['fp_trial']]*n2,
#         'RT(ms)':laser_failed.pad_off_t, 'distance(cm)':laser_failed.distance,'tortuosity':laser_failed.tortuosity,
#         'MT(ms)':laser_failed.fp_trial[0]-laser_failed.pad_off_t,'v_max(cm/s)':np.max(laser_failed.velocity_r,axis=1),
#         'trial':['failed']* n2, 'laser':['laser']* n2}))
    
#     df2 = pd.DataFrame(({'rat_no':normal_successful.rat_no, 'opto_par':[exp_par]*n3, 'pulse':[cfg['laser_pulse']]*n3,
#         'intensity':[cfg['laser_intensity']]*n3,'protocol':[cfg['laser_protocol']]*n3,'Nfpt':[cfg['fp_trial']]*n3,
#         'RT(ms)':normal_successful.pad_off_t,'distance(cm)':normal_successful.distance, 'tortuosity':normal_successful.tortuosity,
#         'MT(ms)':normal_successful.got_reward_t - normal_successful.pad_off_t,'v_max(cm/s)':np.max(normal_successful.velocity_r,axis=1),
#         'trial':['successful']* n3 , 'laser':['no laser']* n3 }))
    
#     df3 = pd.DataFrame(({'rat_no':normal_failed.rat_no, 'opto_par':[exp_par]*n4, 'pulse':[cfg['laser_pulse']]*n4, 
#         'intensity':[cfg['laser_intensity']]*n4,'protocol':[cfg['laser_protocol']]*n4,'Nfpt':[cfg['fp_trial']]*n4,
#         'RT(ms)':normal_failed.pad_off_t, 'distance(cm)':normal_failed.distance,'tortuosity':normal_failed.tortuosity,
#         'MT(ms)':normal_failed.fp_trial[0]-normal_failed.pad_off_t,'v_max(cm/s)':np.max(normal_failed.velocity_r,axis=1),
#         'trial':['failed']* n4 , 'laser':['no laser']* n4 }))

#     result = pd.concat([result, df, df1, df2, df3],ignore_index=True)

#     result['RT(ms)'] = result['RT(ms)']*1000/cfg['fp_trial']
#     result['MT(ms)'] = result['MT(ms)']*1000/cfg['fp_trial']
#     return result

def build_sessions_transf_to_df(pre_direct, folder,exp_par):
    
    ''' build a class object from all sessions with <folder> protocol
        separate laser/non-laser and successful/failed trials
        put important measurements into a df and return
    '''

    pre_direct_chosen = os.path.join(pre_direct, exp_par)


    cfg, files_list_DLC, files_list_LED = find_all_files_same_protocol(pre_direct_chosen, folder)
    where_plot,what_plot = 0,0
    body_part = cfg['body_part_list'][where_plot]
    all_sessions = extract_epochs_over_sessions(files_list_DLC, files_list_LED, folder,body_part,cfg)
    laser = Laser(all_sessions)

    non_laser = Non_Laser(all_sessions)

    
    sessions_dict = { 'laser' : { 'successful' : Successful(laser) , 'failed' : Failed(laser)}, 
    			'non-laser' :  { 'successful' : Successful(non_laser) , 'failed' : Failed(non_laser) }
    			}


    col_names =  ['rat_no','opto_par','pulse','intensity','protocol','Nfpt','RT(ms)','MT(ms)', 'tortuosity','distance(cm)','v_max(cm/s)', 'trial', 'laser']
    
    results = pd.DataFrame(columns = col_names)
    
    for laser, sessions in sessions_dict.items():
    	for result, session in sessions.items():
    		print( laser , result, 'trials =', session.n_trials )

    		df = pd.DataFrame({'rat_no': session.rat_no, 
		    					'opto_par': [exp_par] * session.n_trials, 
		    					'pulse': [ cfg['laser_pulse'] ] * session.n_trials, 
		        				'intensity': [ cfg['laser_intensity']] * session.n_trials,
		        				'protocol': [ cfg['laser_protocol']] * session.n_trials,
		        				'Nfpt': [ cfg['fp_trial']] * session.n_trials,
		        				'RT(ms)': session.pad_off_t, 
		        				'distance(cm)': session.distance,
		        				'tortuosity': session.tortuosity,
		        				'MT(ms)': session.got_reward_t - session.pad_off_t,
		        				# 'v_max(cm/s)': np.max( session.velocity_r, axis=1 ),
		        				'v_max(cm/s)': get_max_velocities_of_all_trials(session, cfg),
		        				'trial': ['successful'] * session.n_trials , 
		        				'laser': ['laser'] * session.n_trials })


    		results = pd.concat([results, df], ignore_index=True)

    results['RT(ms)'] = results['RT(ms)']*1000/cfg['fp_trial']
    results['MT(ms)'] = results['MT(ms)']*1000/cfg['fp_trial']
    return results
def get_max_velocities_of_all_trials(session, cfg):
	''' exclude the masked velocities and return the maximum velocity reached in each trial'''
	v_max = np.zeros(session.n_trials)
	for trial in range(session.n_trials):
		v_trial = session.velocity_r[:,trial]
		v_filtered_trial = v_trial[ v_trial != cfg['velocity_mask'] ]
		if len(v_filtered_trial) > 0 :
			v_max[trial] = np.max( v_filtered_trial)
		else:
			print('no movement velocity could be calculated')
			v_max[trial] = np.nan
	return v_max
def build_metadeta_all_folders(pre_direct, experiment_dict):
              
    result = build_sessions_transf_to_df(pre_direct, experiment_dict['folder'][0],experiment_dict['exp_par'][0])
    for i in range(1,len(experiment_dict['folder'])):
        result = pd.concat([result,build_sessions_transf_to_df( pre_direct, experiment_dict['folder'][i],
                experiment_dict['exp_par'][i])],ignore_index=True)
    return result

def set_conf_and_df(pre_direct, folder, rat_no, exp_par ):
    
    prop = folder.split('_')
    direct = os.path.join(pre_direct, exp_par, 'Rat_' +str(rat_no))  # directory to the folder for each mouse
    R_L_folder = [ f.name for f in os.scandir(os.path.join(direct,folder)) if f.is_dir() ][0]
    path = os.path.join(direct, folder, R_L_folder)
    config_sup_info = { 
                        'optogenetic_manip': True, 
                        'exp_par': Path(path).parts[-4],
                        'r_or_l': R_L_folder[0],
                        'fp_trial': int(prop[5]), 
                        'laser_protocol': prop[3], 
                        'laser_pulse': prop[0],
                        'laser_intensity': prop[1]
                        } 
    configname = set_config_file(folder,direct)
    edit_config(configname, config_sup_info)
    attempt_to_make_folder(os.path.join(path, 'Plots'))
    files_list_DLC = list_all_files(os.path.join(path, 'DLC'),'.csv')
    files_list_LED = list_all_files(os.path.join(path, 'LED'),'.csv')
    cfg = read_plainconfig(configname)
    where_plot,what_plot = 0,0
    body_part = cfg['body_part_list'][where_plot]

    return files_list_DLC, files_list_LED, path, body_part, cfg

def get_DLC_LED_df(files_list_DLC, files_list_LED, n_session,cfg):
    
    df = read_DLC_csv(files_list_DLC[n_session], cfg['fp_trial'] )
    df_LED = read_LED_csv(files_list_LED[n_session],len(df.index))
    
    return df, df_LED

def copy_csv_with_directory_tree(srcDir, dstDir):
    if not os.path.isdir(dstDir):
        shutil.copytree(srcDir, dstDir, 
                        ignore=include_patterns('*.csv'))

def categorize_csv_to_folders(dstDir):
    ''' move the LED csv to  <LED> folder and DLC ones to <DLC> folder '''
    for dirpath, dirnames, filenames in os.walk(dstDir):
        for filename in filenames:

            if filename.endswith('.csv'):

                if not os.path.exists(os.path.join(dirpath, 'LED')):
                    os.makedirs(os.path.join(dirpath, 'LED'))
                if not os.path.exists(os.path.join(dirpath,'DLC')):
                    os.makedirs(os.path.join(dirpath,'DLC'))

                if filename.endswith('LED.csv'):
                    shutil.move(os.path.join(dirpath, filename), 
                                os.path.join(dirpath, 'LED', filename))
                elif filename.endswith('.csv'):
                    shutil.move(os.path.join(dirpath, filename), 
                                os.path.join(dirpath, 'DLC', filename))