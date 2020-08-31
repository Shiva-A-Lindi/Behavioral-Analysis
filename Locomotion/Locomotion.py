
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
import seaborn as sns
from statannot import add_stat_annotation
from scipy.signal import find_peaks
from colour import Color
from pathlib import Path

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
font.set_size('30')
font.set_weight('bold')

font_label = FontProperties()
font_label.set_family('serif')
font_label.set_name('Times New Roman')
font_label.set_style('italic')
font_label.set_size('25')
font_label.set_weight('bold')

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def set_ticks(ax):
    ax.get_xaxis().set_tick_params(direction='out',labelsize = 30,length=10)
    ax.xaxis.set_ticks_position('bottom')
    ax.get_yaxis().set_tick_params(direction='out',labelsize = 30,length=10)
    ax.yaxis.set_ticks_position('left')

def read_DLC(file_name,scale_pix_to_cm):
    '''Read DeepLabCut Data.'''
    df = pd.read_excel(file_name, header=[1,2])*scale_pix_to_cm # scale to cm
    return df

def read_laser(laser_file_name):
    '''Read laser onset offset times for Square pulse stimulation'''
    laser_t = pd.read_excel(laser_file_name, skiprows = 4 )
    return laser_t

def read_laser_beta_stim(laser_file_name):
    '''Read laser onset offset times for Beta pulse stimulation'''
    laser_t = pd.read_excel(laser_file_name, skiprows = 4 )
    onset = np.array(laser_t['ON'].values)
    starts = np.where(shift(onset, -1, cval= 0)-onset > min_t_between_stim) # find the starts of sessions if 
                                     # they are not oscilations of a beta wave
    
    temp = np.zeros((len(onset[starts[0]])+1,))
    temp[0] = onset[0] # save the first detected laser as the dtart of the first session

    temp[1:temp.shape[0],]= onset[starts[0]+1] # the rest are as detected above

    df = pd.DataFrame(({'ON':temp,'OFF':temp+interval}))
    return df

def list_all_files(path,extension):
    '''List all the files with this extention in this directory'''
    files = [x for x in os.listdir(path) if not x.startswith('.')]
    files.sort()
    return list(filter(lambda x: extension in x, files))

def convert_csv_to_xlsx(path):
    '''Check if a .xlsx version of all the .csv files exists, if not convert to this format
        and remove the .csv to save space'''
    files = [x for x in os.listdir(path) if not x.startswith('.')]
    files.sort()
    csv_files = list(filter(lambda x: ".csv" in x, files))
    csv_file_names = [x.replace(".csv","") for x in csv_files] # remove extensions to be able to compare lists
    xlsx_files = list(filter(lambda x: '.xlsx' in x, files))
    xlsx_file_names = [x.replace(".xlsx","") for x in xlsx_files]
    if not set(csv_file_names) < set(xlsx_file_names): # if most files are in csv convert them to xlsx

        for filepath_in in csv_files:
            name = os.path.join(path, filepath_in) 

            try:
                pd.read_csv(name, delimiter=",").to_excel(os.path.join(path, filepath_in.replace(".csv",".xlsx")), header = True,index = False)
            except pd.errors.ParserError: # it must be a laser file
                pd.read_csv(name, delimiter=",",skiprows= 4).to_excel(os.path.join(path, filepath_in.replace(".csv",".xlsx")), startrow=4, header = True, index = False)

            os.remove(name) # remove the csv file.
            

def save_npz(pre_direct,mouse_type,opto_par,folder,pulse_inten,fps, window,n_timebin,file_name_ext,
             epochs_all_mice, epochs_mean_each_mouse, epochs_spont_all_mice,pre_info,
             cor,body_part,plot_param):
    '''Save the trial epochs in one .npz file with the following elements:
        1: all of laser sessions one mouse type
        2: average for each mouse in mouse type
        3: eauivalent number of laser sessions derived from spontaneous'''
    s = '_'
    s = s.join(body_part)

    file_name = (mouse_type+'_'+opto_par+'_'+pulse_inten+file_name_ext+"_mov_aver="+str(int(window/fps*1000))+
        "_n_t="+str(int(n_timebin/fps*1000))+'_'+cor+'_'+plot_param+'_'+s)
    print(epochs_all_mice.shape,pre_info.shape)
    np.savez( os.path.join(pre_direct,'data_npz',folder,opto_par, file_name),
             epochs_all_mice = epochs_all_mice,
             epochs_mean_each_mouse = epochs_mean_each_mouse, 
             epochs_spont_all_mice = epochs_spont_all_mice,
             avg_pre_stim_position = pre_info[:,0],
             avg_pre_stim_velocity = pre_info[:,1],
            avg_pre_stim_acc = pre_info[:,2],
            cor=[cor],
            body_part=body_part,
            plot_param=[plot_param])
#     dat = np.load(pre_direct+'data_npz/'+file_name+'.npz')
#     dat.files
    
def moving_average_array(X, n):
    '''Return the moving average over X with window n without changing dimesions of X'''
    z2= np.cumsum(np.pad(X, (n,0), 'constant', constant_values=0))
    z1 = np.cumsum(np.pad(X, (0,n), 'constant', constant_values=X[-1]))
    return (z1-z2)[(n-1):-1]/n

def align_right_left(right,left):
    '''Correct if for any reason there has been a shift between labelings of right and left side
    
    Parameters
    ----------
    right : 1-D array
        position in time detected from the right camera
    left : 1-D array
        position in time detected from the left camera
    '''
    delta = np.average(right-left)
    if delta > 0: # if the right is ahead
        right -=delta/2 ; left += delta/2
    else: # if the left is ahead
        right +=delta/2 ; left -= delta/2
    return right,left


def derivative(x,delta_t,fps):
    '''Take the derivative with delta_t.'''
    
    derivative_out = (x - shift(x, delta_t, cval= x[0]))/ (delta_t/fps)
    ## got crazy zith keyerror -1 
#     print(derivative_out[-1])
#     return shift(derivative_out,-int(delta_t/2),cval= derivative_out[-1])
    return shift(derivative_out,-int(delta_t/2),cval= derivative_out[len(derivative_out)-1])

def derivative_mov_ave(x,delta_t,window_veloc,fps):
    '''Take the derivative with delta_t and do a moving average.'''
    
    derivative_out = (x - shift(x, delta_t, cval= x[0]))/ (delta_t/fps)
    dx_dt = shift(derivative_out,-int(delta_t/2),cval= derivative_out[len(derivative_out)-1])
    return moving_average_array(dx_dt, window_veloc) # return the moving average
#     return dx_dt # if you don't want to do a moving average 


# def detect_remove_jitter(velocity, max_accel = 95):
#     ''' detect when the DLC labeled wrongly by filtering with acceleration criteria
    
#     to do: set the inconsistent one to the average of before and after'''
#     shifted = shift(velocity, 1, cval= velocity[0])
#     accel = (velocity - shifted)/(1/fps)
#     print(np.sort(accel)[-2])
#     print("max accel = ", max(accel), np.argmax(accel)/fps)
#     ind = np.where(np.absolute(accel) > max_accel)
#     print("# jitters in speed = ",len(ind[0]))
#     return velocity

def input_plot(df, laser_t,mouse_type,mouse_no,trial_no,opto_par,pre_direct,exp_dict,t_window_dict, misdetection_dict,save_as_format = '.pdf'):
    '''Get the specifics of the plot as input and call the corresponding plot function.'''
    study_param_dict = get_input_cor_body_part(**exp_dict)
 
    print(" 1. Right & Left \n 2. Average of both")
    Average_sep_plot = int(input()) # ask what body part to plot
    
    if Average_sep_plot == 2:
        print(Average_sep_plot)
        plot_what_which_where(df,laser_t,mouse_type,mouse_no,trial_no,opto_par,pre_direct, misdetection_dict,
                              **study_param_dict,**t_window_dict,save_as_format='.pdf')
    else:
        plot_what_which_where_r_l(df,laser_t,mouse_type,mouse_no,trial_no,opto_par,pre_direct, misdetection_dict,
                              **study_param_dict,**t_window_dict,save_as_format='.pdf')
        
def get_input_cor_body_part(cor_list,body_part_list,plot_param_list):
    '''Ask for the body part and coordinate from user.'''
    
    print("Select for which parts you want to see the pre/On/post: \n")
    print(" 1. Tail \n 2. Nose \n 3. Fore Limb \n 4. Hind Limb")
    where_plot = [int(x)-1 for x in input().split()] # ask what body part to plot
    print(" 1. X \n 2. Y ")
    which_plot = int(input())-1 # ask what body part to plot
    print(" 1. Position \n 2. Velocity \n ")
    what_plot = int(input())-1 # ask what body part to plot
    study_param_dict = {'cor' : cor_list[which_plot],
                        'body_part' : body_part_list[where_plot],
                       'plot_param' : plot_param_list[what_plot]}
    return study_param_dict

def produce_random_bins_for_spont(max_time,n_sample,pre_interval,interval,post_interval,max_distance,min_distance,n_trials_spont):
    '''Produce a grid for start of bins then perturb to have randomly 
    spaced trials with a minimum and maximum distance between them.
    
    Parameters
    ----------
    
    max_time : int
        number of timebins of the spontaneous session
        
    n_sample : int
        number of trial samples that needs to be extracted from this session
        
    max_distance : int
        maximum number of timebins between extracted trials
    
    min_distance : int
        minimum number of timebins between extracted trials
        
    Returns
    -------
        
    bins : 2D-array (int)
        an array with two columns each row containing (start,end) of each trial
        
    '''
    
    half_max_distance =int(max_distance/2)
    start = pre_interval + half_max_distance # the first timebin eligible for the start of the first timebin
    end = max_time - (post_interval+interval+half_max_distance) # the last timebin eligible for the start of the last trial 
    time_points = np.arange(start, end-((end-start)%(min_distance+half_max_distance)),min_distance+half_max_distance ) # produce the grid
    perturb = np.random.randint(-half_max_distance,half_max_distance, size = (n_sample,len(time_points))) # produce the random perturbations
    start_arr = np.repeat(time_points.reshape(1,len(time_points)),n_sample,axis = 0)+perturb # perturb the starting points
    starts = (np.array(start_arr.ravel())).reshape(len(start_arr.ravel()),1)
    ends = starts+interval
    bins = np.concatenate((starts,ends),axis = 1) #stack start and ends of bins

    return bins

def compare_r_l_correct_misdetect (right_p, left_p, acc_deviance,t_s, 
                                   internal_ctr_dev=0,percent_thresh_align=0,n_iter_jitter=0,jitter_threshold=0):
    '''Compare the right and left sides and correct if the difference between 
        the two is more than an acceptable amount. 
        
    
    Parameters
    ----------
    right_p : 1-D array
        position in time detected from the right camera
    left_p : 1-D array
        position in time detected from the left camera
    acc_deviance : float (passed within misdetect dictionary)
        acceptable deviance of the right and left side detections
    t_s : int
        the number of procedeeing and post time bins to average from and correct the misdetection with
    
    There are other parameters passed as a **kwarg that are not used in this function
    
    Returns 
    -------
    
    right : 1-D array
        corrected position in time detected from the right camera
    left : 1-D array
        corrected position in time detected from the left camera
    '''
    right_x = np.copy(right_p); left_x = np.copy(left_p) ### if not it changes the df that was passed to the function along with the right_x and left_x
#     r_x = np.copy(right_x)
#     l_x = np.copy(left_x)
    delta_x = np.absolute(right_x - left_x) # the difference between detections of right and left cameras
    ind, = np.where(delta_x > internal_ctr_dev)
    
    if len(ind)> percent_thresh_align*len(right_x):# if more than a percentage of detections are not aligned there must be a shift
        print("There's a shift between left and right detections. Don't worry we will fix it!")
        right_x, left_x = align_right_left(right_x,left_x)
        
    delta_x = np.absolute(right_x - left_x) 
    mis_r_l, = np.where(delta_x > acc_deviance) # spot where  
    removed_edge_ind = np.logical_and((mis_r_l > t_s+1), (mis_r_l <(len(right_x)-t_s-1)))
    mis_r_l = mis_r_l[removed_edge_ind]
#     print(mis_r_l)
    # spot which is more deviant from it's neghbors in time
    print("# inconsistent right left = ", len(mis_r_l))
#     print("time stamps = ", mis_r_l[(mis_r_l>34.5*fps) & (mis_r_l<35*fps)]/fps)
    if len(mis_r_l) > 0: # only bother if only you find any mismatches
        compare_within_r = np.zeros((delta_x.shape))
        compare_within_l = np.zeros((delta_x.shape))
        bef_r = np.hstack([ np.absolute(np.average(right_x[j-t_s:j-1])-right_x[j])+
                np.absolute(np.average(right_x[j+1:j+t_s])-right_x[j]) for j in mis_r_l ])/2
        bef_l = np.hstack([ np.absolute(np.average(left_x[j-t_s:j-1])-left_x[j])+
                np.absolute(np.average(left_x[j+1:j+t_s])-left_x[j]) for j in mis_r_l ])/2
        compare_within_r[mis_r_l] = bef_r
        compare_within_l[mis_r_l]= bef_l

        ind_right_larger, =np.where(compare_within_l < compare_within_r) 
        ind_left_larger, =np.where(compare_within_l > compare_within_r) 

        temp_r = np.in1d(mis_r_l,ind_right_larger)
        ind_r_corr = mis_r_l[temp_r] # where there's mismatch and it's annonated to the right one
        temp_l = np.in1d(mis_r_l,ind_left_larger)
        ind_l_corr = mis_r_l[temp_l] # where there's mismatch and it's annonated to the left one
        # correct based on the findings
        # set to the average of the other side rather than the same track because there's a better chance the
        # mistake happens again in the proximity of the same side
        if len(ind_l_corr) > 0 :
            left_x[ind_l_corr] = np.hstack([np.average( right_x[j-t_s:j-1] ) +
                                       np.average( right_x[j+1:j+t_s] ) for j in ind_l_corr])/2
        if len(ind_r_corr) > 0 :
            right_x[ind_r_corr] = np.hstack([np.average( left_x[j-t_s:j-1] ) +
                                       np.average( left_x[j+1:j+t_s] ) for j in ind_r_corr])/2

    return np.array(right_x).reshape((-1, 1)), np.array(left_x).reshape((-1, 1))


def correct_labeling_jitter(x,jitter_threshold,n_iter_jitter,t_s,acc_deviance=0,internal_ctr_dev=0,
                                   percent_thresh_align=0):
    '''Correct the detections exceeding the max speed of the mouse.
    
    The detected jitters are replaced by the average of before and after time stamps.
    
    Parameters
    ----------
    x : 1-D array
        The time series that needs correction (usually the average of right and left cameras).
        
    jitter_threshold : float
        The allowed maximum movement in one timebin with the max allowed velocity.
    
    n_iter_jitter : int
        This process can be done several times (n_iter) to make sure if jitters appear with averaging they are resolved.
    
    t_s : int
        The number of timebins to average from when correcting the jitter point.
        
    Returns
    -------
    x : 1-D array
        The corrected time series.
        
    '''
    for i in range(n_iter_jitter):
        deltas = x - shift(x, 1, cval= x[0])
        ind, = np.where(np.absolute(deltas) > jitter_threshold)
        ind = ind[(ind > t_s+1) & (ind <(len(x)-t_s-1))]
        if len(ind) > 0: # if jumped in detection set to the mean of <t_s> previous and next detections
            x[ind] = np.hstack([np.average(x[j-t_s:j-1])+np.average(x[j+1:j+t_s]) for j in ind])/2
            print("# jitter in mean(righ,left)  = ", len(ind))                                     
#     print("nan found? How many? ",sum(np.isnan(x)))
#     print(x[ind])
    return x


def average_position_r_l(df,window,misdetection_dict,cor,body_part,plot_param):
    '''Return three-level correction of the average between right and left side for multiple selected body-parts.
    
    Extract the right and left trace for each selected body part, correct right and left traces with 
    "compare_r_l_correct_misdetect", then average over right and left and remove jitters by calling
    "correct_labeling_jitter". Then return the moving average of all selected body-parts.
    
    Parameters
    ----------
    
    df : dataframe
        Dataframe derived from DLC for one session of one animal
        
    which_plot : int
        Which coordinate to work with. 0-->x or 1-->y
    
    where_plot : list (int)
        Which body parts to average over. list of int that will be translated to list of strings with a dict.
    
    window : int
        Moving average window for smoothing the position time series
        
    misdetection_dict : dictionary
        Dictionary containing constants for misdetection corrections
        
    Returns
    -------
    corrected_averaged : 1-D array 
        Moving average of all selected body-parts.
        
    '''

    averaged_position = np.zeros((df[('r'+body_part[0],cor)].values.shape))
    
    for param in body_part : # average over body parts
        print("Looking at ---",param,'---')
        right_x = np.copy(df[('r'+param,cor)].values)
        left_x = np.copy(df[('l'+param,cor)].values)
        right, left = compare_r_l_correct_misdetect(right_x,left_x,**misdetection_dict) #first compare r and l
        left_right_corrected_averaged = np.average(np.concatenate((right,left),axis = 1),axis=1)
        # (this corrects when the misdetection happened for both left and right because
        # left and right have already been aligned together)
        averaged_position += correct_labeling_jitter(left_right_corrected_averaged,**misdetection_dict)
#         averaged_position += correct_labeling_jitter(left_x,jitter_threshold,n_iter_jitter, t_s)

    averaged_position = averaged_position/len(body_part)
    corrected_averaged = moving_average_array(averaged_position, window)
    return  corrected_averaged
   
def position_r_l(df,window,misdetection_dict,cor,body_part,plot_param):
    '''Remove jitters with "correct_labeling_jitter" on right and left on the 
    selected body parts and return right and left separately.
    
    Parameters
    ----------
    
    df : dataframe
        Dataframe derived from DLC for one session of one animal
        
    which_plot : int
        Which coordinate to work with. 0-->x or 1-->y
    
    where_plot : list (int)
        Which body parts to average over. list of int that will be translated to list of strings with a dict.
        
    window : int
        Moving average window for smoothing the position time series
        
    misdetection_dict : dictionary 
        Dictionary containing constants for misdetection corrections
        
    Returns
    -------
    right_corrected : 1-D array 
        Moving average of right side for all selected body-parts
        
    left_corrected : 1-D array 
        Moving average of left side for all selected body-parts
        '''

    
    averaged_position_r = np.zeros((df[('r'+body_part[0],cor)].values.shape))
    averaged_position_l = np.zeros((df[('r'+body_part[0],cor)].values.shape))

    for param in body_part : # average over body parts
        right_x = np.copy(df[('r'+param,cor)].values)
        left_x = np.copy(df[('l'+param,cor)].values)
        averaged_position_l += correct_labeling_jitter(left_x,**misdetection_dict)
        averaged_position_r += correct_labeling_jitter(right_x,**misdetection_dict)

    averaged_position_r = averaged_position_r/len(body_part)
    averaged_position_l = averaged_position_l/len(body_part)
    right_corrected = moving_average_array(averaged_position_r, window)
    left_corrected = moving_average_array(averaged_position_l, window)
    return   right_corrected, left_corrected
   





    

def min_and_mean_on_off(epochs,measure,pre_interval,interval,post_interval,pre_stim_inter):
    '''Report the min or mean (specified by measure) velocity in the off-on periods'''
    
    if measure =='Mean':
        try:
            pre = np.average(epochs[:,0:pre_interval],axis = 1).reshape(epochs.shape[0],1)
            ON = np.average(epochs[:,pre_interval:pre_interval+interval],axis = 1).reshape(epochs.shape[0],1)
            post = np.average(epochs[:,pre_interval+interval:pre_interval+interval+post_interval],axis = 1).reshape(epochs.shape[0],1)

        except ZeroDivisionError:
            pre = np.zeros((epochs.shape[0],1))
            ON = np.zeros((epochs.shape[0],1))
            post = np.zeros((epochs.shape[0],1))
    elif measure =='Min':
        try:
            pre = np.average(epochs[:,0:pre_interval],axis = 1).reshape(epochs.shape[0],1)
            ON = np.min(epochs[:,pre_interval:pre_interval+interval],axis = 1).reshape(epochs.shape[0],1)
            post = np.average(epochs[:,pre_interval+interval:pre_interval+interval+post_interval],axis = 1).reshape(epochs.shape[0],1)
        except ZeroDivisionError:
            pre = np.zeros((epochs.shape[0],1))
            ON = np.zeros((epochs.shape[0],1))
            post = np.zeros((epochs.shape[0],1))
    average_of_on_off_on = np.concatenate((pre, ON, post), axis = 1)
    return average_of_on_off_on

def plot_what_which_where_r_l(df,laser_t,mouse_type,mouse_no,trial_no,opto_par, pre_direct,misdetection_dict, 
                        cor,body_part,plot_param,
                         fps,n_timebin,window_pos,window_veloc,save_as_format='.pdf'):
    '''Choose which body part/
                what measure/
                for x or y
        to see the left vs. right traces
    '''
    label_1 = "Right" 
    label_2 = "left "
    time_series  = df.index / fps ## time axis in seconds for stimulation trial
    trial_time = max(time_series)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)

    if plot_param == 'velocity':
        
        velocity_r = derivative_mov_ave(df[('r'+body_part[0],cor)], n_timebin, window_veloc,fps)
        velocity_l = derivative_mov_ave(df[('l'+body_part[0],cor)], n_timebin, window_veloc,fps)
        plt.plot(time_series+n_timebin/fps, velocity_r, 'navy',label = label_1, linewidth = 0.8)
        plt.plot(time_series+n_timebin/fps, velocity_l, 'orangered',label = label_2, linewidth = 0.8)
        plt.xlim(n_timebin/fps, trial_time/2)
        plt.ylabel(cor+ " Velocity (cm/s)" ,fontproperties=font_label)
        min_end = min(min(velocity_r),min(velocity_l))
        max_end = max(max(velocity_r),max(velocity_l))
        plt.axhline(y = 0, color = 'r',linestyle= '--', linewidth = 0.9) # plot zero velocity threshold
        
    elif plot_param == "acceleration":
        
        accel_r = derivative_mov_ave(derivative_mov_ave(df[('r'+body_part[0],cor)],n_timebin,window_veloc,fps),
                                     n_timebin,window_veloc,fps)
        accel_l = derivative_mov_ave(derivative_mov_ave(df[('l'+body_part[0],cor)],n_timebin,window_veloc,fps),
                                     n_timebin,window_veloc,fps)
        t_shift = n_timebin/fps/2
        plt.plot(time_series+t_shift+t_shift, accel_r, 'navy',label = label_1, linewidth = 0.8)
        plt.plot(time_series+t_shift+t_shift, accel_l, 'orangered',label = label_2, linewidth = 0.8)
        plt.xlim(t_shift+t_shift, trial_time/2)
        plt.ylabel(cor+ " Acceleration (cm/s**2)" ,fontproperties=font_label)
        min_end = min(min(accel_r),min(accel_l))
        max_end = max(max(accel_r),max(accel_l))
        plt.axhline(y = 0, color = 'r',linestyle= '--', linewidth = 0.9) # plot zero velocity threshold
        
    else:
        r,l = compare_r_l_correct_misdetect (df[('r'+body_part[0],cor)].values, df[('l'+body_part[0],cor)].values, **misdetection_dict)
        plt.plot(time_series, r, 'navy',  label = label_1, linewidth = 0.8)
        plt.plot(time_series, l, 'orange', label = label_2, linewidth = 0.8)

        #plt.plot(time_series, df[('r'+body_part[0],cor)], 'navy',  marker = 'o', label = label_1,markersize=1)
        #plt.plot(time_series, df[('l'+body_part[0],cor)], 'orangered', label = label_2, marker = 'o', linewidth = 0.8, markersize=1)
        #plt.xlim(min_x,max_x)
        #plt.ylim(min_y,max_y)

        plt.ylabel(cor+ " (cm)" ,fontproperties=font_label)
        min_end = min(min(df[('r'+body_part[0],cor)]),min(df[('l'+body_part[0],cor)]))
        max_end = max(max(df[('r'+body_part[0],cor)]),max(df[('l'+body_part[0],cor)]))
         
    set_ticks(ax)   
    plt.ylim(min_end,max_end)
    plt.xlabel("Time(s)" ,fontproperties=font_label)
    plt.title(mouse_type+' '+ opto_par+' #'+str(mouse_no),fontproperties=font)
    plt.legend(fontsize = 20)
#     plt.ylim(min_end,max_end)
    for i in range(len(laser_t['ON'].values)):
        plt.axvspan(laser_t['ON'].values[i]/fps, laser_t['OFF'].values[i]/fps, alpha=0.4, color='lightskyblue')
#     plt.vlines(laser_t['ON']/fps,min_end,max_end, color = 'orange', linewidth = 0.4) # plot stimulus onsets
#     plt.vlines(laser_t['OFF']/fps,min_end,max_end, color = 'orange', linewidth = 0.4) # plot stimulus offsets

    plt.savefig(os.path.join(pre_direct,"One_session",'Mouse_trial'+str(trial_no)+'_mouse_' +str(mouse_no)+'_'+
                body_part[0] + '_' + plot_param + '_' +cor+ save_as_format),bbox_inches='tight',orientation='landscape',dpi=300)



def plot_what_which_where(df,laser_t,mouse_type,mouse_no,trial_no,opto_par, pre_direct,misdetection_dict, 
                        cor,body_part,plot_param,
                         fps,n_timebin,window_pos,window_veloc,save_as_format='.pdf'):
    '''Choose to see averaged velocity, position or acceleration for a chosen combination of body_parts
        for either x or y coordiante.
    '''
    s = '_'
    s = s.join(body_part)
    time_series  = df.index / fps ## time axis in seconds for stimulation trial
    trial_time = max(time_series)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)

    averaged_pos = average_position_r_l(df,window_pos,misdetection_dict,cor,body_part,plot_param)

    time_series  = df.index / fps
    trial_time = max(time_series)
    
    if plot_param == "velocity":
        
        velocity = derivative_mov_ave(averaged_pos,n_timebin,window_veloc,fps)
        plt.plot(time_series+n_timebin/fps/2, velocity, 'k', linewidth = 2, label = s) # plot all body
        plt.ylabel(cor+ " Velocity (cm/s)" ,fontproperties=font_label)
        min_end = min(velocity)
        max_end = max(velocity)
        plt.axhline(y = 0, color = 'r',linestyle= '--', linewidth = 0.9) # plot zero velocity threshold
    
    elif plot_param == "acceleration":
        
        accel = derivative_mov_ave(derivative_mov_ave(averaged_pos,n_timebin,window_veloc,fps),n_timebin,window_veloc,fps)
        t_shift = n_timebin/fps/2
        plt.plot(time_series+t_shift+t_shift, accel, 'k', linewidth = 2, label = s) # plot all body
        plt.xlim(t_shift+t_shift, trial_time/2)
        plt.ylabel(cor+ " Acceleration (cm/(s2))" ,fontproperties=font_label)
        min_end = min(accel)
        max_end = max(accel)
        plt.axhline(y = 0, color = 'r',linestyle= '--', linewidth = 0.9) # plot zero velocity threshold
        
    else:
        plt.plot(time_series, averaged_pos, '-k', linewidth = 1, label = s,markersize = 1) # plot all body        
        plt.ylabel(cor+ " (cm)" ,fontproperties=font_label)
        min_end = min(averaged_pos)
        max_end = max(averaged_pos)
        
    set_ticks(ax)
    
    plt.xlabel("Time(s)" ,fontproperties=font_label)
    plt.title("Average "+mouse_type+' '+ opto_par+' #'+str(mouse_no),fontproperties=font)
    plt.legend(fontsize = 20)
    plt.ylim(min_end,max_end)
    for i in range(len(laser_t['ON'].values)):
        plt.axvspan(laser_t['ON'].values[i]/fps, laser_t['OFF'].values[i]/fps, alpha=0.4, color='lightskyblue')
#     plt.vlines(laser_t['ON']/fps,min_end,max_end, color = 'darkskyblue', linewidth = 0.4) # plot stimulus onsets
#     plt.vlines(laser_t['OFF']/fps,min_end,max_end, color = 'darkskyblue', linewidth = 0.4) # plot stimulus offsets
    plt.savefig(os.path.join(pre_direct,"One_session",'Mouse_trial'+str(trial_no)+'_mouse_' +str(mouse_no)+'_' +cor+ '_' +
                plot_param+ '_averaged_'+s+save_as_format),bbox_inches='tight',orientation='landscape',dpi=300)


 ###################  looking at the pre/on/post stimulus behavior #########################
def plot_pre_on_post(ax,pre_direct,mouse_type,opto_par,folder,epochs,epochs_spont,treadmil_velocity,ylim,
                        fps,n_timebin, window_pos,window_veloc, 
                        cor,body_part,plot_param,
                     pre_interval,interval,post_interval,pre_stim_inter,
                    average = 'Averg_trials_all_mice',c_laser = 'deepskyblue',c_spont = 'k',save_as_format = '.pdf'):
    '''Plot (pre Laser/ON-Laser/post Laser) velocity/position/acceleration comparison between 
    laser trials and spontaneous'''
    
#     epochs_mean = np.average(epochs, axis = 0) # average over different stimuli
#     epochs_mean_spont = np.average(epochs_spont, axis = 0) # average over different stimuli
    
#     epochs_sem = stats.sem(epochs, axis=0) # SEM 
#     epochs_sem_spont = stats.sem(epochs, axis=0) # SEM 

    confidence_inter = np.empty((0,2), int)
    
    if len(epochs.shape) > 1 :
        epochs_mean = np.average(epochs, axis = 0)
        epochs_mean_spont = np.average(epochs_spont, axis = 0)
        
        for i in range (epochs.shape[1]): #calculate the two sided confidence interval for every timestep
            m = [sms.DescrStatsW(epochs[:,i]).tconfint_mean(alpha=0.05, alternative='two-sided')]
            confidence_inter = np.append(confidence_inter,[[m[0][0],m[0][1]]],axis=0)
        confidence_inter_spont = np.empty((0,2), int)
        
        for i in range (epochs_spont.shape[1]):
            m = [sms.DescrStatsW(epochs_spont[:,i]).tconfint_mean(alpha=0.05, alternative='two-sided')]
            confidence_inter_spont = np.append(confidence_inter_spont,[[m[0][0],m[0][1]]],axis=0)
    else:
        epochs_mean = epochs
        epochs_mean_spont = epochs_spont
        confidence_inter = np.zeros(((epochs_mean.shape[0]),2))
        confidence_inter_spont = np.zeros((epochs_mean.shape[0],2))
    time_series = np.arange(-pre_interval,interval+post_interval+1)/fps
    
    plt1,=plt.plot(time_series, epochs_mean, color = c_laser, label = body_part, linestyle='-',linewidth=2)#, marker='o',markersize=1)
    plt.fill_between(time_series, confidence_inter[:,0],  confidence_inter[:,1], color=c_laser, alpha=0.2)
    
    plt2,=plt.plot(time_series, epochs_mean_spont, color = c_spont, label = "Spontaeous")
    plt.fill_between(time_series, confidence_inter_spont[:,0],  confidence_inter_spont[:,1], color=c_spont, alpha=0.2)
    plt.axvspan(0, interval/fps, alpha=0.2, color='lightskyblue')
    
#     plt.fill_between(time_series, epochs_mean - epochs_sem,  epochs_mean+ epochs_sem,
#                     color='gray', alpha=0.2)
#     plt.fill_between(time_series, epochs_mean_spont - epochs_sem_spont,  epochs_mean_spont+ epochs_sem_spont,
#                    color='b', alpha=0.2)

    if plot_param == 'velocity':
        plt.ylabel(" Velocity (cm/s)").set_fontproperties(font_label)
    elif plot_param == 'position':
        plt.ylabel(" Position (cm)").set_fontproperties(font_label)
    else:
        plt.ylabel(" Acceleration (cm/s**2)").set_fontproperties(font_label)
    plt.axhline( y = treadmil_velocity, ls='--', c='red')
    
    plt.xlabel("Time(s)").set_fontproperties(font_label)
    plt.ylim(ylim[0],ylim[1]) #set limits
    plt.legend(fontsize = 10)
#     plt.xlim(-.5,2) #set limits
    s = '_'
    s = s.join(body_part)
    if average == 'n':
        ax.get_xaxis().set_tick_params(direction='out',labelsize = 20 ,length=6)
        ax.xaxis.set_ticks_position('bottom')
        ax.get_yaxis().set_tick_params(direction='out',labelsize = 20, length=6)
        ax.yaxis.set_ticks_position('left')
        plt.title(plot_param+"("+mouse_type+") #"+str(mouse_no) +" "+opto_par+"\n").set_fontproperties(font)
        plt.savefig(os.path.join(pre_direct,'Compare','Mouse'+str(mouse_no)+'_'+folder+"_"+opto_par+'_'+mouse_type+'_'+cor+'_'+plot_param+
            '_' +s+'_timebin_'+str(int(n_timebin*1000/fps))+'_window_pos_'+str(int(window_pos*1000/fps))+
                    '_window_veloc_'+str(int(window_veloc*1000/fps))+'_pre_post_stim'+save_as_format),bbox_inches='tight',orientation='landscape',dpi=300)
    elif average == 'Averg_trials': # one one mouse
        plt.title("("+mouse_type+") #"+str(mouse_no) +" "+opto_par+"\n"+folder).set_fontproperties(font)
#         plt.title(" #"+str(mouse_no) +" "+mouse_type , fontproperties=font)
#         plt.savefig(pre_direct+'Compare'+'/'+average+'_Mouse'+str(mouse_no)+'_'+folder+"_"+opto_par+'_'+mouse_type+'_'+ cor+ '_Velociy_' +str(n_timebin)+ 'spont_sampl='+str(no_sample)+'_pre_post_stim.png',bbox_inches='tight',orientation='landscape',dpi=400)
    else:
        ax.get_xaxis().set_tick_params(direction='out',labelsize = 20 ,length=6)
        ax.xaxis.set_ticks_position('bottom')
        ax.get_yaxis().set_tick_params(direction='out',labelsize = 20, length=6)
        ax.yaxis.set_ticks_position('left')
        plt.legend(handles=[plt1,plt2],labels=['Laser Stimulation','Spontaneous'],loc = 'lower right',fontsize =20)
        plt.title( mouse_type+"\n"+" Average of all "+opto_par).set_fontproperties(font)

        plt.savefig(os.path.join(pre_direct,'Average'+'_'+folder+'_'+opto_par+'_'+mouse_type+'_'+ cor+ '_'+plot_param+'_' +s
                    +'_timebin_'+str(int(n_timebin*1000/fps))+'_window_pos_'+str(int(window_pos*1000/fps))+
                    '_window_veloc_'+str(int(window_veloc*1000/fps))+'_pre_post_stim'+save_as_format),bbox_inches='tight',
                    orientation='landscape',dpi=300)

def epochs_single_file(file_name_pos,file_name_spont,file_name_laser):
    '''Read a single file for a mouse and return the epochs.'''
    
    df = read_DLC(file_name_pos,scale_pix_to_cm)
    df_spont = read_DLC(file_name_spont,scale_pix_to_cm)
    laser_t = read_laser(file_name_laser)


    time_series  = df.index / fps ## time axis in seconds for stimulation trial
    trial_time = max(time_series)

    time_series_spont  = df_spont.index / fps ## time axis in seconds for spontaneous activity
    trial_time_spont = max(time_series_spont)

    velocity = derivative(average_position_r_l(df,window_pos,misdetection_dict,**study_param_dict).values,n_timebin)  # velocity
    bins  = np.array(laser_t.values).astype(int)
    epochs = extract_epochs(bins,velocity,*accep_interval_range,**intervals_dict)
    velocity_spont = derivative(average_position_r_l(df_spont,window_pos,misdetection_dict, **study_param_dict).values,n_timebin)  # velocity 
    bins_spont  = np.array(laser_t.values).astype(int) # for now
    epochs_spont = extract_epochs(bins,velocity_spont,*accep_interval_range,**intervals_dict)
    return epochs,epochs_spont


def extract_pre_laser_x_epochs_over_trials(files_list,files_list_laser,
                                           direct,folder,scale_pix_to_cm,window_pos,misdetection_dict,
                                           smallest_accep_inter,largest_accep_inter,pre_stim_inter):
    '''Return all the x positions in epochs preceding to a ON-laser of all trials for one mouse.'''

    i = 0
    epochs = np.empty((0,pre_stim_inter))

    for i in range(0,len(files_list)):
        
        print('session {} out of {}'.format(i+1,len(files_list)))
        file_path_DLC = os.path.join(direct,folder,'DLC',files_list[i])
        df = read_DLC(file_path_DLC,scale_pix_to_cm)
        study_param_dict = {'cor':'x', 'body_part' : ['Tail','Nose'], 'plot_param' : 'position'}
        x_average = average_position_r_l(df,window_pos,misdetection_dict, **study_param_dict) # x positions
        file_path_Laser = os.path.join(direct,folder,'Laser',files_list_laser[i])
        laser_t = read_laser(file_path_Laser)
        bins  = np.copy(laser_t.values).astype(int)
        duration = bins[:,1]-bins[:,0]
        acceptable = np.logical_and(duration>smallest_accep_inter,duration<largest_accep_inter)
        print(len(acceptable)-sum(acceptable),' trials discarded')
        bins = bins[acceptable,:]
        
        take = np.hstack([np.arange(i[0]-pre_stim_inter,i[0]) for i in bins[:-1]]) # make an array with indices of laser ON 
        pre_x_series = x_average[take]
        # calculate the x position relative to the front edge of the treadmill
        epochs_trial = np.absolute(pre_x_series-max(pre_x_series)).reshape(len(bins)-1,pre_stim_inter)        
        epochs = np.append(epochs,epochs_trial,axis = 0)

    return epochs


    
def plot_every_mouse_mean(epochs_mean_each_mouse):
    '''plot mean of each mouse for epochs OFF-ON-OFF'''
    
    for i in range (epochs_mean_each_mouse.shape[0]):
        plt.plot(median,epochs_mean_each_mouse[i,:],'-',color = 'gray',marker = '.')

def violin_plot_summary(data_init,names,measure):
    ''' plot violin plot for different columns in data'''

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Time Epoch')

    # transform the data into list as to be fed to the violin plot function
    data = list([data_init[:,i] for i in range(data_init.shape[1])])
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(20, 10), sharey=True)

    ax2.set_title(mouse_type)
    parts = ax2.violinplot(data, widths=0.3, showmeans=False, showmedians=False,showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('teal')
        pc.set_edgecolor('None')
        pc.set_alpha(.4)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)

    inds = np.arange(1, len(medians) + 1)
    confidence_inter = np.empty((0,2), int)

    for i in range (data_init.shape[1]): #calculate the two sided confidence interval for every timestep
        m = [sms.DescrStatsW(data_init[:,i]).tconfint_mean(alpha=0.05, alternative='two-sided')]
        confidence_inter = np.append(confidence_inter,[[m[0][0],m[0][1]]],axis=0)

    ax2.scatter(inds,np.average(data_init,axis=0), marker='o', color='white', s=20, zorder=3, label = 'Means') # measns
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=2, label = 'Quartiles') # quartiles

    ax2.errorbar(inds, np.average(data_init,axis=0),yerr=confidence_inter.T,fmt='none', elinewidth=2, capsize=7,marker='o', color='r',label = '95% Confidence interval') # confidence interval

#     ax2.boxplot(inds, data)
#     ax2.vlines(inds, np.min(data_init,axis=0), np.max(data_init,axis=0), color='r', linestyle='-', lw=1,label = 'Whiskers')
# 
    # set style for the axes
    set_axis_style(ax2, names)
    
    # plot each mouse data individually on top of the summary of all
    if measure == 'Mean':
        for i in range (epochs_mean_each_mouse.shape[0]):
            plt.plot(inds,epochs_mean_each_mouse[i,:],'-',color = 'gray',marker = 'o',fillstyle='none',markersize=10,linewidth=2,alpha=0.6)
    elif measure == 'Min':
        for i in range (epochs_min_each_mouse.shape[0]):
            plt.plot(inds,epochs_min_each_mouse[i,:],'-',color = 'gray',marker = 'o',markersize=10,linewidth=2,alpha=0.6)
            
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.ylabel(" Velocity (cm/s)", fontproperties=font_label)
    plt.xlabel("Epoch", fontproperties=font_label)
    plt.ylim(-16,30) #set limits
    plt.title(cor+ " "+measure+" Velocity"+"\n"+mouse_type+' '+opto_par +"\n"+folder, fontproperties=font)
    plt.legend(fontsize = 20)
    plt.savefig(os.path.join(pre_direct,'Summary','Aver_of_pre_stim_post'+opto_par+'_'+folder+'_'+mouse_type+
                             '_'+ cor+'_'+measure+ '_Velociy_' +str(n_timebin)+'_violin_plot'+'.png'),bbox_inches='tight',orientation='landscape',dpi=400)


def plot_two_protocols_with_mouse_distinction(mouse_type, mouse_list,folder1,folder2,opto_par):
    '''Extract data over all mice of one group in two folders intensity and plot with subplots of individual animals'''
    epochs_spont_all_mice = np.empty((0,pre_interval+interval+post_interval+1))
    epochs_all_mice = np.empty((0,pre_interval+interval+post_interval+1))
#     epochs_mean_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice
    epochs_spont_all_mice_1 = np.empty((0,pre_interval+interval+post_interval+1))
    epochs_all_mice_1 = np.empty((0,pre_interval+interval+post_interval+1))
#     epochs_mean_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice

#     epochs_min_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice
    plt.figure(2)
    fig = plt.figure(figsize=(20,15))
    nrows=3;ncols=4
#     fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharey=True)

    count = 0
    if opto_par == "Control":
        loop_list = mouse_list[1]
    else:
        loop_list = mouse_list[0]

    for n in loop_list: # Run over all the mice
        count +=1
        start = timeit.default_timer()
        global mouse_no
        mouse_no = n
        print(mouse_type+" # ",n)
        direct = os.path.join(pre_direct, mouse_type, opto_par, 'Mouse_' +str(mouse_no))  # directory to the folder for each mouse
#         print(direct+folder)
#         convert_csv_to_xlsx(direct+folder+'/DLC')
        files_list_DLC1 = list_all_files(os.path.join(direct,folder,'DLC'),'.xlsx')
#         convert_csv_to_xlsx(direct+folder+'/Laser') # files might be given in csv, this is to unify 
        files_list_Laser1 = list_all_files(os.path.join(direct,folder,'Laser'),'.xlsx')
        files_list_spont1 = list_all_files(os.path.join(direct,'Spontaneous','DLC'),'.xlsx')
        files_list_DLC2 = list_all_files(os.path.join(direct,folder,'DLC'),'.xlsx')
#         convert_csv_to_xlsx(direct+folder+'/Laser') # files might be given in csv, this is to unify 
        files_list_Laser2 = list_all_files(os.path.join(direct,folder,'Laser'),'.xlsx')
        files_list_spont2 = list_all_files(os.path.join(direct,'Spontaneous','DLC'),'.xlsx')
#         files_list_spont = []
        if len(files_list_DLC1)==0 :
            print("No files for mouse # ",n)
            continue
        elif len(files_list_Laser1)==0 :
            print("No Laser detection for mouse # ",n)
            continue
        else:

            epochs, = extract_epochs_over_trials(files_list_DLC1,files_list_Laser1,direct,folder1,scale_pix_to_cm,'n',accep_interval_range,
                                                spont_trial_dict,misdetection_dict,study_param_dict,**intervals_dict,**t_window_dict) 
            print('total of {} trials'.format(epochs.shape[0]))
            epochs_all_mice = np.append(epochs_all_mice, epochs, axis = 0)# construct an array of all the trial epochs of all mice
            no_epochs = epochs.shape[0] # number of epochs extracted for the mouse

            n_spont_sessions = len(files_list_spont1) # number of spont sessions
            global no_sample
            no_sample = int(no_epochs/(n_spont_sessions*n_trials_spont))+1 # number of repeats 
                                        # over a spont file to get the same number of epochs as laser session
            epochs_spont, = extract_epochs_over_trials(files_list_spont1,files_list_Laser1,direct,
                                                       'Spontaneous',scale_pix_to_cm,'y',accep_interval_range,
                                                spont_trial_dict,misdetection_dict,study_param_dict,**intervals_dict,**t_window_dict)
            epochs_spont_all_mice = np.append(epochs_spont_all_mice, epochs_spont, axis = 0) # construct an array of all the spont epochs

            ax = fig.add_subplot(3,4,count)
            plot_pre_on_post(ax,pre_direct,mouse_type,opto_par,folder,epochs,epochs_spont,treadmil_velocity,ylim,**t_window_dict,**study_param_dict,
                             **intervals_dict,average = 'Averg_trials',c_laser = 'deepskyblue',c_spont = 'k',save_as_format = '.pdf')
            
            epochs, = extract_epochs_over_trials(files_list_DLC2,files_list_Laser2,direct,folder2,scale_pix_to_cm,'n',accep_interval_range,
                                                spont_trial_dict,misdetection_dict,study_param_dict,**intervals_dict,**t_window_dict) 
            print('total of {} trials'.format(epochs.shape[0]))
            epochs_all_mice = np.append(epochs_all_mice, epochs, axis = 0)# construct an array of all the trial epochs of all mice
            no_epochs = epochs.shape[0] # number of epochs extracted for the mouse

            n_spont_sessions = len(files_list_spont2) # number of spont sessions

            no_sample = int(no_epochs/(n_spont_sessions*n_trials_spont))+1 # number of repeats 
                                        # over a spont file to get the same number of epochs as laser session
            epochs_spont, = extract_epochs_over_trials(files_list_spont2,files_list_Laser2,direct,
                                                       'Spontaneous',scale_pix_to_cm,'y',accep_interval_range,
                                                spont_trial_dict,misdetection_dict,study_param_dict,**intervals_dict,**t_window_dict)
            epochs_spont_all_mice = np.append(epochs_spont_all_mice, epochs_spont, axis = 0) # construct an array of all the spont epochs

            plot_pre_on_post(ax,pre_direct,mouse_type,opto_par,folder,epochs,epochs_spont,treadmil_velocity,ylim,**t_window_dict,**study_param_dict,
                             **intervals_dict,average = 'Averg_trials',c_laser = 'mediumseagreen',c_spont = 'hotpink',save_as_format = '.pdf')

    plt.tight_layout()
    
    fig.suptitle(cor+ "-Velocity ("+mouse_type+") "+opto_par+"\n"+" interval = "+ str(n_timebin) , fontproperties=font)
    plt.savefig(os.path.join(pre_direct,'Subplots','All_together_'+folder+"_"+opto_par+'_'+mouse_type+'_'+ cor+ '_'+body_part+'_timebin=' +str(n_timebin)+ "_moving_aver_win="
        +str(window)+ '_spont_sampl='+str(no_sample)+'_pre_post_stim.pdf'),bbox_inches='tight',orientation='landscape',dpi=200)
    plt.show()

    
def MWW_test(result,result_Ctr,mouse_type):
    '''Return two-sided Mann-Whitney test for ON-laser period velocity between Ctr and ChR2.
    
    Parameters
    ----------
    
    result : df
        a data frame containing mean velocity for ChR2 mice.

    result_Ctr : df
        a data frame containing mean velocity for Control mice.
        
    mouse_type : str
        mouse type as a string e.g. FoxP2, D2, Vglut2 ..
    
    Returns
    -------
    
    stat : obj
        MWW stat results
    '''
    x = result_Ctr[(result_Ctr['epoch']=='ON') & (result_Ctr['mouse_type'] == mouse_type)]['mean_velocity'].values
    y = result[(result['epoch']==('ON'+mouse_type)) & (result['mouse_type'] == mouse_type)]['mean_velocity'].values
    stat = stats.mannwhitneyu(x,y)
    print("MWW ChR2 vs. Ctr "+mouse_type+" = ",stat)
    return statt


# def average_pre_x_v_and_save(mouse_type, folder, opto_par,pre_interval,interval,post_interval,pre_stim_inter,back_front_boundary,v_threshold,pre_stim_inter):
#     '''Save data of epochs together with x and v prior to the laser epoch and individal mice to a npz file
#     by running over all mice of one group and one intensity . 
    
#     Parameters
#     ----------
    
#     mouse_type : str
#         mouse type such as 'FoxP2' or ..
    
#     mouse_list : list (int)
#         list of mouse identification numbers
        
#     folder : str
#         the folder with the corresponding experiment protocol e.g. "Square_1_mW".
    
#     opto_par: str
#         "ChR2" --> for ChR2 injected animals
#         "Control" --> for Control group
    
#     '''

#     where_plot,which_plot, what_plot = get_input_cor_body_part() # decide to average over what and coordinates
#     cor = cor_list[which_plot] # gets either x or y by user

#     if pre_interval > pre_stim_inter:
#         pre_era = pre_interval
#     else:
#         pre_era = pre_stim_inter
        
#     epochs_all_mice = np.empty((0,pre_interval+interval+post_interval+1))
#     x_all_mice = np.empty((0,1))
#     mean_pre_vel_all_mice = np.empty((0,1))

#     if opto_par == "Control":
#         loop_list = mouse_no_dict[mouse_type][1]
#     else:
#         loop_list = mouse_no_dict[mouse_type][0]
        
#     count = 0
#     for n in loop_list: # Run over all the mice
#         count +=1
#         start = timeit.default_timer()
#         mouse_no = n
#         print("# mouse = ",mouse_no)
#         direct = os.path.join(pre_direct, mouse_type, opto_par, 'Mouse_' + str(mouse_no)) # directory to the folder for each mouse
#         files_list_DLC = list_all_files(os.path.join(direct,folder,'DLC',".xlsx"))
#         files_list_Laser = list_all_files(os.path.join(direct,folder,'Laser',".xlsx"))
#         if len(files_list_DLC)==0 :
#             print("No files for mouse # ",n)
#             continue
#         elif len(files_list_Laser)==0 :
#             print("No Laser detection for mouse # ",n)
#             continue
#         else:
#             epochs = extract_epochs_over_trials(files_list_DLC,files_list_Laser,direct,folder,'n',accep_interval_range,
#                                                 spont_trial_dict,intervals_dict,**t_window_dict,**exp_dict) 
#             if pre_era > pre_interval:
#                 epochs_all_mice = np.append(epochs_all_mice, epochs[:,pre_stim_inter-pre_interval:], axis = 0)# construct an array of all the trial epochs of all mice
#                 pre_vel = np.average(epochs[:,:pre_stim_inter],axis = 1)
#             else:
#                 epochs_all_mice = np.append(epochs_all_mice, epochs, axis = 0)# construct an array of all the trial epochs of all mice
#                 pre_vel = np.average(epochs[:,pre_interval - pre_stim_inter:pre_era],axis = 1)
#             what_plot = 0 # for x
#             pre_x_position_epochs = extract_pre_laser_x_epochs_over_trials(files_list_DLC,files_list_Laser,direct,folder,
#                                                                            *accep_interval_range, **pre_x_v_dict)
#             pre_x_position = np.average(pre_x_position_epochs[:,:pre_stim_inter],axis = 1)
#             x_all_mice = np.append(x_all_mice,pre_x_position)
#             mean_pre_vel_all_mice = np.append(mean_pre_vel_all_mice,pre_vel)

#             stop = timeit.default_timer()
#             print('run time = ',stop-start)

#     save_npz(mouse_type,opto_par,"Distinction",folder, window,n_timebin,"_pre_x_pre_v_pre_stim_inter="+str(pre_stim_inter),epochs_all_mice, x_all_mice, mean_pre_vel_all_mice,where_plot,which_plot,what_plot,where_plot,**exp_dict)


# def read_npz_return_data_frame(path,pre_interval,post_interval,interval):
#     ''' reads the saved npz and produces a data frame with the following columns'''
    
#     Summary_files_list = list_all_files(path,".npz")
#     col_names =  ['mean_velocity', 'mouse_type', 'optogenetic expression', 'pulse_type','intensity_mW','epoch','velocity']
#     result = pd.DataFrame(columns = col_names)
#     for file in Summary_files_list:
#         print(file)
#         dat = np.load(path+file)
#         properties=file.split("_")
#         epochs = dat[dat.files[0]]
#         n_epochs = epochs.shape[0]
#         ### set the variables of epoch/optogen/pulse type/mouse type/velocity and x of pre stim
#         pre = epochs[:,:pre_interval] ; post = epochs[:,pre_interval+1:pre_interval+interval+1]
#         mouse_type_ = [properties[0]] * n_epochs*2
#         opto_par_ = [properties[1]] * n_epochs*2
#         pulse_ = [properties[2]] * n_epochs*2
#         inten_ = [properties[3]] * n_epochs*2
#         Velocity = dat[dat.files[2]]
#         x = dat[dat.files[1]]
#         x_ = np.concatenate((x,x),axis=0) 
#         Velocity_ = np.concatenate((Velocity,Velocity),axis=0) 
#         try:
#             off_vel = np.average(pre,axis = 1)
#             on_vel = np.average(post,axis = 1)
#             all_ = np.concatenate((off_vel,on_vel),axis = 0)
#         except ZeroDivisionError:
#             all_ = np.empty((0))
#         epoch_off = ['OFF'] * n_epochs
#         epoch_on = ['ON'] * n_epochs
#         epoch_ = epoch_off+epoch_on
#         # append the data of each mouse to a unit dataframe
#         df = pd.DataFrame(({'mean_velocity':all_, 
#                             'mouse_type':mouse_type_, 'optogenetic expression':opto_par_, 'pulse_type':pulse_,
#                             'intensity_mW':inten_,'epoch':epoch_,'velocity':Velocity_,'x':x_}))
#         frames = [result, df]
#         result = pd.concat(frames,ignore_index=True)
#     return result

def read_npz_return_data_frame(file_path_list,pre_interval,interval,post_interval,pre_stim_inter):
    '''Read the saved .npz file and produce a data frame with the following columns.'''
    
    
    col_names =  ['mean_velocity', 'min_velocity', 'mouse_type', 'optogenetic expression', 
                'pulse_type','intensity_mW','epoch','pre_velocity_pos_neg','pre_x','pre_x_front_back','pre_accel','pre_accel_pos_neg']
    result = pd.DataFrame(columns = col_names)
    for file in file_path_list:
        print(file)
        dat = np.load(file)
        properties=file.split("_")
        epochs = dat['epochs_all_mice']
        n_epochs = epochs.shape[0]
        ### set the variables of epoch/optogen/pulse type/mouse type/velocity and x of pre stim
        pre = epochs[:,:pre_interval] ; on = epochs[:,pre_interval+1:pre_interval+interval+1]
        mouse_type_ = [properties[0]] * n_epochs*2
        opto_par_ = [properties[1]] * n_epochs*2
        pulse_ = [properties[2]] * n_epochs*2
        inten_ = [properties[3]] * n_epochs*2

        x_ = np.concatenate((dat['avg_pre_stim_position'],dat['avg_pre_stim_position']),axis=0) 
        Velocity_ = np.concatenate((dat['avg_pre_stim_velocity'],dat['avg_pre_stim_velocity']),axis=0) 
        accel_ = np.concatenate((dat['avg_pre_stim_acc'],dat['avg_pre_stim_acc']),axis=0) 
        print(dat['avg_pre_stim_position'].shape,epochs.shape)
        try:
            off_vel = np.average(pre,axis = 1)
            on_vel = np.average(on,axis = 1)
            all_mean = np.concatenate((off_vel,on_vel),axis = 0)
        except ZeroDivisionError:
            all_mean = np.empty((0))

        all_min = np.concatenate((np.min(pre,axis = 1),np.min(on,axis = 1)),axis = 0)
        epoch_off = ['OFF'] * n_epochs
        epoch_on = ['ON'] * n_epochs
        epoch_ = epoch_off+epoch_on
        # append the data of each mouse to a unit dataframe
        df = pd.DataFrame(({'mean_velocity':all_mean, 'min_velocity':all_min,
                            'mouse_type':mouse_type_, 'optogenetic expression':opto_par_, 'pulse_type':pulse_,
                            'intensity_mW':inten_,'epoch':epoch_,'pre_velocity_pos_neg':Velocity_,'pre_x':x_, 'pre_x_front_back':x_,
                            'pre_accel' : accel_,'pre_accel_pos_neg':accel_}))
        frames = [result, df]
        result = pd.concat(frames,ignore_index=True)
    return result

def categorize_pre_x_and_v(result,back_front_boundary,v_threshold,pre_stim_inter):
    '''Set threshold to velocity and x position averaged over pre_stim_inter.'''
    
    ind_0 = result['pre_velocity_pos_neg'] < v_threshold
    ind_1 = result['pre_velocity_pos_neg'] > v_threshold
    ind_2 = result['pre_x'] < back_front_boundary
    ind_3 = result['pre_x'] > back_front_boundary
    ind_4 = result['pre_accel'] < v_threshold
    ind_5 = result['pre_accel'] > v_threshold
    result.loc[ind_0 ,'pre_velocity_pos_neg'] = 'neg'
    result.loc[ind_1 ,'pre_velocity_pos_neg'] = 'pos'
    result.loc[ind_4 ,'pre_accel_pos_neg'] = 'neg'
    result.loc[ind_5 ,'pre_accel_pos_neg'] = 'pos'
    result.loc[ind_2 ,'pre_x_front_back'] = 'front'
    result.loc[ind_3 ,'pre_x_front_back'] = 'back'
    return result

def Plot_ON_OFF_X_V_mean(result,path,mouse_type,folder,fps,back_front_boundary,v_threshold,pre_stim_inter,ylim=[-20,15],save_as_format='.pdf'):
    '''Plot the mean laser-ON laser-OFF vellociry with distinction of velocity and x prior to laser stim.'''
    
    result_pos = result[result['pre_velocity_pos_neg'] == 'pos']
    result_neg = result[result['pre_velocity_pos_neg'] == 'neg']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)

    y = [np.average(result_pos[(result_pos['pre_x_front_back']=='front') & (result_pos['epoch']=='OFF')]['mean_velocity']),\
         np.average(result_pos[(result_pos['pre_x_front_back']=='front') & (result_pos['epoch']=='ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0],2))

    plt.plot(['OFF','ON'],y,'-',\
             color = 'g',marker = 'o',markersize=10,linewidth=4,alpha=1, label = 'V>0 - front-N='+\
             str(len(result_pos[(result_pos['pre_x_front_back']=='front') & (result_pos['epoch']=='ON')]['mean_velocity']))+\
             ' $\delta$='+delta)

    y = [np.average(result_pos[(result_pos['pre_x_front_back']=='back') & (result_pos['epoch']=='OFF')]['mean_velocity']),\
             np.average(result_pos[(result_pos['pre_x_front_back']=='back') & (result_pos['epoch']=='ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0],2))
    plt.plot(['OFF','ON'],y,'-',
        color = 'r',marker = 'o',markersize=10,linewidth=4,alpha=1, label = 'V>0 - back-N='+\
        str(len(result_pos[(result_pos['pre_x_front_back']=='back') & (result_pos['epoch']=='ON')]['mean_velocity']))+\
        ' $\delta$='+delta)

    y = [np.average(result_neg[(result_neg['pre_x_front_back']=='front') & (result_neg['epoch']=='OFF')]['mean_velocity']),\
             np.average(result_neg[(result_neg['pre_x_front_back']=='front') & (result_neg['epoch']=='ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0],2))
    plt.plot(['OFF','ON'],y,'-',
        color = 'b',marker = 'o',markersize=10,linewidth=4,alpha=1, label = 'V<0 - front-N='+\
        str(len(result_neg[(result_neg['pre_x_front_back']=='front') & (result_neg['epoch']=='ON')]['mean_velocity']))+\
        ' $\delta$='+delta)

    y = [np.average(result_neg[(result_neg['pre_x_front_back']=='back') & (result_neg['epoch']=='OFF')]['mean_velocity']),\
             np.average(result_neg[(result_neg['pre_x_front_back']=='back') & (result_neg['epoch']=='ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0],2))
    plt.plot(['OFF','ON'],y,'-',
        color = 'k',marker = 'o',markersize=10,linewidth=4,alpha=1, label = 'V<0 - back-N='+\
        str(len(result_neg[(result_neg['pre_x_front_back']=='back') & (result_neg['epoch']=='ON')]['mean_velocity']))+\
        ' $\delta$='+delta)
    inten = result['intensity_mW'][0]
    legend = plt.legend(loc='upper right',fontsize= 12)
    plt.xlabel("Laser",fontsize=15)#.set_fontproperties(font_label), 
    plt.ylabel("Average velocity (cm/s)",fontsize=15)#.set_fontproperties(font_label)
    plt.suptitle(mouse_type+'  '+folder+'(I='+inten+')'+'\n'+'pre-stim-inetrval = '+str(int(pre_stim_inter*1000/fps))+' ms',fontsize= 20,y = 1)
    #set_ticks(ax)
    plt.ylim(ylim[0],ylim[1])
    ax.set_facecolor((0.8, 1.0, 1.0))
    plt.savefig(os.path.join(path,'X_V_distinction_mean_'+folder+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps))+
                             '_inten='+inten+save_as_format),bbox_inches='tight',orientation='landscape',dpi=350)
    
def Plot_ON_OFF_V_mean(result,path,mouse_type,folder,fps,back_front_boundary,v_threshold,pre_stim_inter,ylim=[-20,15],save_as_format='.pdf'):
    '''Plot the mean laser-ON laser-OFF vellociry witfps,h distinction of velocity prior to laser stim'''
    
    result_pos = result[result['pre_velocity_pos_neg'] == 'pos']
    result_neg = result[result['pre_velocity_pos_neg'] == 'neg']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)

    y = [np.average(result_pos[(result_pos['epoch']=='OFF')]['mean_velocity']),
             np.average(result_pos[(result_pos['epoch']=='ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0],2))
    yerr = [np.std(result_pos[(result_pos['epoch']=='OFF')]['mean_velocity']),
             np.std(result_pos[(result_pos['epoch']=='ON')]['mean_velocity'])]
    plt.errorbar(['OFF','ON'],y,yerr,marker = 'o',markersize=10,linewidth=2,capsize=10,capthick=3,
             color = 'r', label = 'V>0 -N='+
             str(len(result_pos[(result_pos['epoch']=='ON')]['mean_velocity']))+
             ' $\delta$='+delta)

    
    y = [np.average(result_neg[ (result_neg['epoch']=='OFF')]['mean_velocity']),
             np.average(result_neg[(result_neg['epoch']=='ON')]['mean_velocity'])]
    yerr = [np.std(result_neg[(result_neg['epoch']=='OFF')]['mean_velocity']),
             np.std(result_neg[(result_neg['epoch']=='ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0],2))
    plt.errorbar(['OFF','ON'],y,yerr,marker = 'o',markersize=10,linewidth=2,capsize=10,capthick=3,
        color = 'k', label = 'V<0 -N='+
        str(len(result_neg[ (result_neg['epoch']=='ON')]['mean_velocity']))+
        ' $\delta$='+delta)

    
    inten = result['intensity_mW'][0]
    legend = plt.legend(loc='upper right',fontsize= 12)
    plt.ylim(ylim[0],ylim[1])
    plt.xlabel("Laser",fontsize=15)#.set_fontproperties(font_label), 
    plt.ylabel("Average velocity (cm/s)",fontsize=15)#.set_fontproperties(font_label)
    plt.suptitle(mouse_type+'  '+folder+'(I='+inten+')'+'\n'+'pre-stim-inetrval = '+str(int(pre_stim_inter*1000/fps))+' ms',fontsize= 20,y = 1)
    #set_ticks(ax)
    ax.set_facecolor((1, 1.0, .8))
    plt.savefig(os.path.join(path,'X_V_distinction_mean_'+folder+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps))+
                             '_inten='+inten+save_as_format),bbox_inches='tight',orientation='landscape',dpi=350)
    

def violin_plot_X_V_distiction(result, path,mouse_type,folder,fps,back_front_boundary,v_threshold,pre_stim_inter,ylim=[-30,30],save_as_format = '.pdf'):
    
    g = sns.catplot(x="epoch", y="mean_velocity",
                    hue="pre_x_front_back", col="pre_velocity_pos_neg",
                    data=result, kind="violin", split=True, palette = sns.color_palette("Set2", n_colors=2, desat=.5),
                     scale_hue=False, linewidth = 2, inner="quartile", scale = 'area',
                    hue_order=['front','back'],col_order=['pos','neg'],legend = False)

    ax1, ax2 = g.axes[0]

    sns.set(font_scale = 2)
    sns.set_style("white")
    plt.ylim(ylim[0],ylim[1])

    ax1.axhline( y=0, ls='-', c='y',linewidth = 3)
    ax2.axhline( y=0, ls='-', c='y',linewidth = 3)

    ax1.axhline( y=-15, ls='--', c='r',linewidth = 3)
    ax2.axhline( y=-15, ls='--', c='r',linewidth = 3)

    ax1.set_title('V > 0', y=0.95, fontsize = 25)
    ax2.set_title('V < 0', y=0.95, fontsize = 25)

    ax1.set_xlabel('Laser', fontsize = 25)
    ax2.set_xlabel('Laser', fontsize = 25)

    ax1.set_ylabel('Average velocity (cm/s)', fontsize = 25)

    ax1.get_xaxis().set_tick_params(direction='out',labelsize = 20,length=10)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.get_yaxis().set_tick_params(direction='out',labelsize = 20,length=10)
    ax1.yaxis.set_ticks_position('left')

    ax2.get_xaxis().set_tick_params(direction='out',labelsize = 20,length=10)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.get_yaxis().set_tick_params(direction='out',labelsize = 20,length=10)
    ax2.yaxis.set_ticks_position('left')

    inten = result['intensity_mW'][0]
    #g.set_axis_labels("Laser", "Average velocity (cm/s)")
    plt.suptitle(mouse_type+'  '+folder+' ('+inten+'mW)'+' pre-stim-interval = '+str(int(pre_stim_inter*1000/fps))+' ms',fontsize= 30,y = 1)
    g.fig.set_figwidth(20.0)
    g.fig.set_figheight(12)
    legend = plt.legend(loc='upper right', title='position on treadmill',fontsize= 20)
    
    plt.savefig(os.path.join(path, mouse_type+'_X_V_violin_plot_'+folder+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps))
                             +'_inten='+inten+save_as_format),bbox_inches='tight',orientation='landscape',dpi=350)


def plot_phase_space_V(result,path,mouse_type,folder,fps,back_front_boundary,v_threshold,pre_stim_inter,xlim=[-25,32],ylim=[-30,40],save_as_format='.pdf'):
    '''Plot the phase space of laser-ON vs. laser-OFF velocity for all trials of all mice.'''
    
    result_pos = result[result['pre_velocity_pos_neg'] == 'pos']
    result_neg = result[result['pre_velocity_pos_neg'] == 'neg']
#     result_zero = result[(result['velocity'] != 'neg') & (result['velocity'] != 'pos')]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
    OFF_vel= result_pos[(result_pos['epoch']=='OFF')]['mean_velocity']
    plt.scatter(result_pos[(result_pos['epoch']=='OFF')]['mean_velocity'],
        result_pos[(result_pos['epoch']=='ON')]['mean_velocity'], c='navy',label = r'$V > 0$')
    plt.scatter(result_neg[(result_neg['epoch']=='OFF')]['mean_velocity'],
        result_neg[(result_neg['epoch']=='ON')]['mean_velocity'], c='purple',label = r'$V < 0$')

    inten = result['intensity_mW'][0]
    plt.plot([-40,40],[-40,40], '--', c='k',label = r'$ V_{ON}=V_{OFF}$')
    legend = plt.legend(loc='upper right',fontsize= 20)
    plt.xlabel("Velocity OFF (cm/s)").set_fontproperties(font_label), 
    plt.ylabel("Velocity ON (cm/s)").set_fontproperties(font_label)
    plt.suptitle(mouse_type+'  '+folder+'(I='+inten+')'+'\n'+'pre-stim-interval = '+str(int(pre_stim_inter*1000/fps))+' ms',fontsize= 30,y = 1)
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    set_ticks(ax)
#     ax.set_facecolor((0.8, 1.0, 1.0))
    plt.savefig(os.path.join(path, mouse_type+'_V_phase_space_'+folder+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps))+
                             '_inten='+inten+save_as_format),bbox_inches='tight',orientation='landscape',dpi=350)

def min_velocity_diff_inten_box_plot(file_path_list,path_to_save,intervals_dict,pre_x_v_dict,ylim=[-40,15],save_as_format = '.pdf'):
    n_subplots = len(file_path_list)


    fig, axes = plt.subplots(nrows=1, ncols=n_subplots, figsize=(4*n_subplots, 8))
    for count in range(1,n_subplots+1):
        path =file_path_list[count-1]
        print(Path(path).name)
        result_val = read_npz_return_data_frame([path],**intervals_dict)
        result = categorize_pre_x_and_v(result_val,**pre_x_v_dict)
        result = result[result['epoch']=='ON']
        mouse_type = Path(path).name.split("_")[0]
        inten = Path(path).name.split("_")[2:4]

        s = ' '
        inten = s.join(inten)
        ax = axes[count-1] 
        ax = plt.subplot(100+n_subplots*10+count)
        set_ticks(ax)
        
        sns.stripplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"],dodge=True, data=result,jitter=True, 
                   marker='o',edgecolor = 'k',linewidth = 1, size = 3,
                   alpha=0.5)
        g = sns.boxplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"],dodge=False, width = 0.4 ,
            data=result, fliersize = 0)

        add_stat_annotation(g, data=result,x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], box_pairs=[("pos", "neg")],
                    test='Mann-Whitney', text_format='star', loc='outside', verbose=2,fontsize = 20)
        plt.plot([0.3,0.7],[np.average(result[result['pre_velocity_pos_neg']=='pos']['min_velocity']),
                                          np.average(result[result['pre_velocity_pos_neg']=='neg']['min_velocity'])],
                                           '-',lw = 3, c= 'r',alpha = 0.5,markersize = 12)
        for patch in g.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.ylim(ylim[0],ylim[1])
        # get legend information from the plot object
        # handles, labels = ax.get_legend_handles_labels()
        # specify just one legend
        # plt.legend(handles[0:2], labels[0:2], fontsize = 20)
        plt.ylabel('').set_fontproperties(font_label)
        plt.xlabel(r'$min(V_{laser-On})$').set_fontproperties(font_label)
        plt.title(inten,fontsize = 18,pad=75)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.suptitle(mouse_type,y=1.15, fontproperties=font)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Velocity (cm/s)',fontsize = 25,labelpad=45).set_fontproperties(font_label)
    plt.savefig(os.path.join(path.replace(Path(path).name,''),'Min_V_diff_inten_'+mouse_type+save_as_format),bbox_inches='tight',orientation='landscape',dpi=200)

def min_velocity_mouse_type_box_plot(file_path_list,path_to_save,intervals_dict,pre_x_v_dict,ylim=[-40,15],save_as_format = '.pdf'):
    n_subplots = len(file_path_list)


    fig, axes = plt.subplots(nrows=1, ncols=n_subplots, figsize=(4*n_subplots, 8))
    for count in range(1,n_subplots+1):
        path =file_path_list[count-1]
        print(path)

        result_val = read_npz_return_data_frame([path],**intervals_dict)
        result = categorize_pre_x_and_v(result_val,**pre_x_v_dict)
        result = result[result['epoch']=='ON']
        mouse_type = Path(path).name.split("_")[0]
        ax = axes[count-1] 
        ax = plt.subplot(100+n_subplots*10+count)
        set_ticks(ax)
        
        sns.stripplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"],dodge=True, data=result,jitter=True, 
                   marker='o',edgecolor = 'k',linewidth = 1, size = 3,
                   alpha=0.5)
        g = sns.boxplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"],dodge=False, width = 0.4 ,
            data=result, fliersize = 0)

        add_stat_annotation(g, data=result,x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], box_pairs=[("pos", "neg")],
                    test='Mann-Whitney', text_format='star', loc='outside', verbose=2,fontsize = 20)
        plt.plot([0.3,0.7],[np.average(result[result['pre_velocity_pos_neg']=='pos']['min_velocity']),
                                          np.average(result[result['pre_velocity_pos_neg']=='neg']['min_velocity'])],
                                           '-',lw = 3, c= 'r',alpha = 0.5,markersize = 12)
        for patch in g.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.ylim(ylim[0],ylim[1])
        # get legend information from the plot object
        # handles, labels = ax.get_legend_handles_labels()
        # specify just one legend
        # plt.legend(handles[0:2], labels[0:2], fontsize = 20)
        plt.ylabel('').set_fontproperties(font_label)
        plt.xlabel(r'$min(V_{laser-On})$').set_fontproperties(font_label)
        plt.title(mouse_type,fontsize = 18,pad=75)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.suptitle(folder,y=1.15, fontproperties=font)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Velocity (cm/s)',fontsize = 25,labelpad=45).set_fontproperties(font_label)
    plt.savefig(os.path.join(path,'Min_V_diff_mouse_type_'+folder+save_as_format),bbox_inches='tight',orientation='landscape',dpi=200)


def extract_epochs(bins,x,smallest_accep_inter,largest_accep_inter,pre_interval,interval,post_interval):
    '''Extract the (pre | Laser ON | post) epochs.
    
    Check reported (start,end) of trials from bins and discard trial with unacceptable duration. 
    Return the stacked the corresponding frames of all trials from the measurment array x.
    
    Parameters
    ----------
    
    bins : 2D-array (int)
        array of start and end times for trials
        
    pre_interval : int
        number of timebins for the pre-laser epoch
    
    post_interval : int
        number of timebins for the post-laser epoch
        
    interval : int
        number of timebins for the laser epoch
    '''
    bins_in = np.copy(bins)
    ### remove the unacceptable epochs
    duration = bins_in[:,1]-bins_in[:,0]
    acceptable = np.logical_and(duration>smallest_accep_inter,duration<largest_accep_inter)

    print(len(acceptable)-sum(acceptable),' trials discarded')
    bins_in = bins[acceptable,:]

    ### find the epochs != interval 
    larger_intervals = (duration[acceptable])>interval # find the exterior intervals to the standard interval
    smaller_intervals = (duration[acceptable])<interval # find the inferior intervals to the min interval
    
    # remove or add the extra frames  to make it uniform along the different stimuli
    bins_in[larger_intervals,1] = bins_in[larger_intervals,1] - (bins_in[larger_intervals,1]-bins_in[larger_intervals,0]-interval)
    bins_in[smaller_intervals,1] = bins_in[smaller_intervals,1] + (interval-(bins_in[smaller_intervals,1]-bins_in[smaller_intervals,0]))
    bins_in[:,1] = bins_in[:,0] + interval + post_interval; bins_in[:,0] = bins_in[:,0] - pre_interval  # extend the interval to pre and post
    n_trials = len(bins_in)-1
    take = np.hstack([np.arange(i[0],i[1]+1) for i in bins_in[:-1]]) # make an array with indices of laser ON timebins
    epochs = x[take].reshape(n_trials,pre_interval+post_interval+interval+1)
    
    return epochs,n_trials,take

def extract_epochs_over_trials(files_list,files_list_laser,direct,folder,scale_pix_to_cm,spont,accep_interval_range,
                               spont_trial_dict,misdetection_dict,study_param_dict,pre_interval,interval,post_interval,pre_stim_inter,
                               fps,n_timebin,window_veloc,window_pos,no_sample=25):

    '''Return all the epochs of all similar trials for one mouse.
    
    Parameters
    ----------
    
    files_list : list (str)
        List of paths to all DLC .csv files for sessions
        
    files_list_laser : list (str)
        List of paths to all .csv files that have (start,end) laser times.
        
    direct : str
        Path to the parent directory containing all folders
        
    spont : str
        Either 'n' --> for trials containing laser stimulation
        or 'y' --> for spontaneous trials with no laser
    
    no_sample : int (optional)
        Number of samples to extract from one spontaneous session. Only necessary for spont = 'y' cases.
        
    accep_interval_range: tuple (int)
        Smallest and largest acceptable trial sizes in timebins
        
    
    interval_dict : dictionary
        Contains {'pre_interval',interval,'post_interval'}
        Number of timebins for the pre-laser, laser and post-laser epochs
        
    n_timebin : int
        Number of timebins for the x derivative to yiels velocity
    
    window_veloc : int
        Velocity moving average window in timebins
    
    window_pos : int
        Position moving average window in timebins
        
    Returns
    -------
    
    epochs : 2D-array
        (pre | Laser ON | post) epochs with shape (n_trials, pre_interval+post_interval+interval+1)
    
    '''
    if pre_interval > pre_stim_inter:
        pre_era = pre_interval
    else:
        pre_era = pre_stim_inter
        
    plot_param = study_param_dict['plot_param']
#     print( plot_param)
    i = 0

    epochs = np.empty((0,pre_interval+interval+post_interval+1))
    epochs_pos = np.empty((0,pre_stim_inter+interval+post_interval+1))
    epochs_veloc = np.empty((0,pre_stim_inter+interval+post_interval+1))
    epochs_acc = np.empty((0,pre_stim_inter+interval+post_interval+1))
    pre_info = np.empty((0,3))
    for i in range(0,len(files_list)):
        
        print('session {} out of {}'.format(i+1,len(files_list)))
        file_path_DLC = os.path.join(direct,folder,'DLC',files_list[i])
        df = read_DLC(file_path_DLC,scale_pix_to_cm)
        
        position = average_position_r_l(df,window_pos,misdetection_dict,**study_param_dict)
        velocity = derivative_mov_ave(position,n_timebin,window_veloc,fps)
        acceleration = derivative(velocity,n_timebin,fps)
        # if only onse side is needed
        #variable,left_side = position_r_l(df, which_plot,where_plot)
        if plot_param == 'position':
            variable = position
            
        elif plot_param == 'velocity':
            variable = velocity   # velocity
            
        elif plot_param == 'acceleration':
            variable = acceleration   # velocity

        if spont == 'y': #if it's a spontaneous reading extract epochs randomly 
            bins = produce_random_bins_for_spont(len(variable),no_sample,pre_interval,interval,post_interval,
                                                 **spont_trial_dict)
            
        else: # if a normal trial read bins from laser times
            file_path_Laser = os.path.join(direct,folder,'Laser',files_list_laser[i])
            laser_t = read_laser(file_path_Laser)
            bins  = np.copy(laser_t.values).astype(int)
        epochs_trial,n_trials,take = extract_epochs(bins,variable,*accep_interval_range,pre_era,interval,post_interval)
        epochs_pos = position[take].reshape(n_trials,pre_era+post_interval+interval+1)
        epochs_veloc = velocity[take].reshape(n_trials,pre_era+post_interval+interval+1)
        epochs_acc = acceleration[take].reshape(n_trials,pre_era+post_interval+interval+1)
        if pre_era > pre_interval:

            pre_info = np.append(pre_info,
                np.concatenate((np.average(epochs_pos[:,:pre_stim_inter],axis = 1).reshape(-1,1),
                                np.average(epochs_veloc[:,:pre_stim_inter],axis = 1).reshape(-1,1),
                                np.average(epochs_acc[:,:pre_stim_inter],axis = 1).reshape(-1,1)),  axis=1),    axis=0)
            epochs = np.append(epochs,epochs_trial[:,pre_stim_inter-pre_interval:],axis = 0)
        else:
            
            pre_info = np.append(pre_info,
                np.concatenate((np.average(epochs_pos[:,pre_interval - pre_stim_inter:pre_era],axis = 1).reshape(-1,1),
                                np.average(epochs_veloc[:,pre_interval - pre_stim_inter:pre_era],axis = 1).reshape(-1,1),
                                np.average(epochs_acc[:,pre_interval - pre_stim_inter:pre_era],axis = 1).reshape(-1,1)),    axis=1),    axis=0)
            epochs = np.append(epochs,epochs_trial,axis = 0)

    if plot_param == 'position':
        return epochs- np.repeat(epochs[:,pre_interval].reshape(epochs.shape[0],1),epochs.shape[1],axis = 1)
    else:
        return epochs,pre_info
    

    
def run_one_intensity_save_data(pre_direct,scale_pix_to_cm, mouse_type, mouse_list,folder,opto_par,treadmil_velocity,
                                ylim,spont_trial_dict,misdetection_dict,intervals_dict,t_window_dict,accep_interval_range,study_param_dict,
                                max_distance,min_distance,n_trials_spont):
    '''Save data of epochs and individal mice to a npz file
    by running over all mice of one group and one intensity. 
    
    Parameters
    ----------
    pre_direct : str
        path to rooot directory
        
    mouse_type : str
        mouse type such as 'FoxP2' or ..
    
    mouse_list : list (int)
        list of mouse identification numbers
        
    folder : str
        the folder with the corresponding experiment protocol e.g. "Square_1_mW".
    
    opto_par: str
        "ChR2" --> for ChR2 injected animals
        "Control" --> for Control group

    
    '''
    epochs_spont_all_mice = np.empty((0,intervals_dict['pre_interval']+intervals_dict['interval']+
                                      intervals_dict['post_interval']+1))
    epochs_all_mice = np.empty_like(epochs_spont_all_mice)
    epochs_mean_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice
#     epochs_min_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice
    all_pre_info = np.empty((0,3))
    epochs_mean_pre_v = np.empty((0,1))
    plt.figure(2)
#     fig = plt.figure(figsize=(20,15))
    fig = plt.figure(figsize=(30,20))

    nrows=3;ncols=4
#     fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharey=True)

    count = 0
    if opto_par == "Control":
        loop_list = mouse_list[1]
    else:
        loop_list = mouse_list[0]

    for n in loop_list: # Run over all the mice
        count +=1
        start = timeit.default_timer()
        global mouse_no
        mouse_no = n
        print("Type : ",mouse_type+" # ",n)
        direct = os.path.join(pre_direct, mouse_type, opto_par, 'Mouse_' +str(mouse_no)) # directory to the folder for each mouse

        files_list_DLC = list_all_files(os.path.join(direct,folder,'DLC'),'.xlsx')
#         convert_csv_to_xlsx(direct+folder+'/Laser') # files might be given in csv, this is to unify 
        files_list_Laser = list_all_files(os.path.join(direct,folder,'Laser'),'.xlsx')
        files_list_spont = list_all_files(os.path.join(direct,'Spontaneous','DLC'),'.xlsx')
#         files_list_spont = []
        if len(files_list_DLC)==0 :
            print("No files for mouse # ",n)
            continue
        elif len(files_list_Laser)==0 :
            print("No Laser detection for mouse # ",n)
            continue
        else:

            epochs,pre_info = extract_epochs_over_trials(files_list_DLC,files_list_Laser,direct,folder,scale_pix_to_cm,'n',accep_interval_range,
                                                spont_trial_dict,misdetection_dict,study_param_dict,**intervals_dict,**t_window_dict) 
            print(epochs.shape[0], 'laser trials')
            epochs_all_mice = np.append(epochs_all_mice, epochs, axis = 0)# construct an array of all the trial epochs of all mice
            all_pre_info = np.append(all_pre_info, pre_info, axis = 0)
            no_epochs = epochs.shape[0] # number of epochs extracted for the mouse
            if len(files_list_spont)==0: # if no spont trials recorded set it to zero
                epochs_spont = np.zeros(np.shape(epochs))
                epochs_spont_all_mice = np.append(epochs_spont_all_mice, epochs_spont, axis = 0) # construct an array of all the spont epochs
            else:
                
                n_spont_sessions = len(files_list_spont) # number of spont sessions
                global no_sample
                no_sample = int(no_epochs/(n_spont_sessions*n_trials_spont))+1 # number of repeats 
                                        # over a spont file to get the same number of epochs as laser session
                epochs_spont,blah = extract_epochs_over_trials(files_list_spont,files_list_Laser,direct,'Spontaneous',scale_pix_to_cm,'y',accep_interval_range,
                                                          spont_trial_dict,misdetection_dict,study_param_dict,**intervals_dict,**t_window_dict, no_sample = no_sample)
                print('Spontaneous session available. {} trials extracted'.format(epochs_spont.shape[0]))

            epochs_spont_all_mice = np.append(epochs_spont_all_mice, epochs_spont, axis = 0) # construct an array of all the spont epochs
            temp = min_and_mean_on_off(epochs,'Mean',**intervals_dict)# get the mean value of velocity for pre-on-post intervals
            average_of_on_off_on = np.average(temp,axis = 0).reshape(1,3)# average over trialas
            epochs_mean_each_mouse = np.append(epochs_mean_each_mouse,average_of_on_off_on,axis = 0) # construct an array with these 3values for all the mice

            ax = fig.add_subplot(3,4,count)
#             ax = axes[int(count/ncols),count%nrows]
            plot_pre_on_post(ax,pre_direct,mouse_type,opto_par,folder,epochs,epochs_spont,treadmil_velocity,ylim,**t_window_dict,**study_param_dict,
                             **intervals_dict,average = 'Averg_trials',
                             c_laser = 'deepskyblue',c_spont = 'k',save_as_format = '.pdf')
            stop = timeit.default_timer()
            print('runtime = ',int(stop-start)," s")
            plt.axhline( y = 17, ls='--', c='g')

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()

#     fig.suptitle(cor+ "-Velocity ("+mouse_type+") "+opto_par+"\n"+" interval = "+ str(n_timebin) , fontproperties=font)
    plt.savefig(os.path.join(pre_direct,'Subplots','All_together_'+folder+"_"+opto_par+'_'+mouse_type+'_'+ study_param_dict['cor']
                             +'_'+study_param_dict['body_part'][0]+'_timebin='+str(t_window_dict['n_timebin'])+ "_moving_aver_win="+str(t_window_dict['window_pos'])+ '_spont_sampl='+str(no_sample)+'_pre_post_stim.pdf'),bbox_inches='tight',orientation='landscape',dpi=200)
    plt.show()

    save_npz(pre_direct,mouse_type,opto_par,folder,folder,t_window_dict['fps'],t_window_dict['window_pos'],t_window_dict['n_timebin'],"",
             epochs_all_mice, epochs_mean_each_mouse, epochs_spont_all_mice,all_pre_info,**study_param_dict)
 

def save_npz_limb_and_tail(pre_direct,scale_pix_to_cm, mouse_type, mouse_list,folder,opto_par,
                                misdetection_dict,pre_interval,interval,post_interval,t_window_dict,accep_interval_range,cor_list,body_part_list,plot_param_list):
    '''Save data of epochs and individal mice to a npz file
    by running over all mice of one group and one intensity. 
    
    Parameters
    ----------
    pre_direct : str
        path to rooot directory
        
    mouse_type : str
        mouse type such as 'FoxP2' or ..
    
    mouse_list : list (int)
        list of mouse identification numbers
        
    folder : str
        the folder with the corresponding experiment protocol e.g. "Square_1_mW".
    
    opto_par: str
        "ChR2" --> for ChR2 injected animals
        "Control" --> for Control group

    
    '''
    print(" 1. X \n 2. Y ")
    which_plot = int(input())-1 # ask what body part to plot
    print(" 1. Position \n 2. Velocity \n ")
    what_plot = int(input())-1 # ask what body part to plot
    cor = cor_list[which_plot]
    plot_param = plot_param_list[what_plot]
    
    epochs_limb = np.empty((0,pre_interval+interval+post_interval+1))
    epochs_tail = np.empty_like(epochs_limb)
    count = 0
    if opto_par == "Control":
        loop_list = mouse_list[1]
    else:
        loop_list = mouse_list[0]

    for n in loop_list: # Run over all the mice
        count +=1

        #global mouse_no
        mouse_no = n
        print("Type : ",mouse_type+" # ",n)
        direct = os.path.join(pre_direct, mouse_type, opto_par, 'Mouse_' +str(mouse_no)) # directory to the folder for each mouse

        files_list_DLC = list_all_files(os.path.join(direct,folder,'DLC'),'.xlsx')
#         convert_csv_to_xlsx(direct+folder+'/Laser') # files might be given in csv, this is to unify 
        files_list_Laser = list_all_files(os.path.join(direct,folder,'Laser'),'.xlsx')
        if len(files_list_DLC)==0 :
            print("No files for mouse # ",n)
            continue
        elif len(files_list_Laser)==0 :
            print("No Laser detection for mouse # ",n)
            continue
        else:
            study_param_dict = {'cor': cor,'body_part':['HL'], 'plot_param':plot_param}
            epochs = extract_epochs_over_trials_one_side_one_body_part(files_list_DLC,files_list_Laser,direct,folder,scale_pix_to_cm,accep_interval_range,
                               misdetection_dict,pre_interval,interval,post_interval,
                               **t_window_dict,**study_param_dict)
            epochs_limb = np.append(epochs_limb, epochs, axis = 0)# construct an array of all the trial epochs of all mice

            study_param_dict = {'cor': cor,'body_part':['Tail'], 'plot_param':plot_param}
            epochs = extract_epochs_over_trials_one_side_one_body_part(files_list_DLC,files_list_Laser,direct,folder,scale_pix_to_cm,accep_interval_range,
                               misdetection_dict,pre_interval,interval,post_interval,
                               **t_window_dict,**study_param_dict)
            epochs_tail = np.append(epochs_tail, epochs, axis = 0)# construct an array of all the trial epochs of all mice


    file_name = (mouse_type+'_'+opto_par+'_'+folder+"_mov_aver="+str(int(t_window_dict['window_pos']/t_window_dict['fps']*1000))+
        "_n_t="+str(int(t_window_dict['n_timebin']/t_window_dict['fps']*1000))+'_'+cor+'_'+plot_param+'_pre_inter_'+str(int(pre_interval/t_window_dict['fps']*1000))+
        '_post_inter_'+str(int(post_interval/t_window_dict['fps']*1000)))

    path = os.path.join(pre_direct,'data_npz',folder,opto_par,'Limb_Tail')

    if not os.path.exists(path):
        os.makedirs(path)

    np.savez( os.path.join(path, file_name),
             epochs_all_mice_limb = epochs_limb,
             epochs_all_mice_tail = epochs_tail,
            cor=[cor],
            body_part=['HL','Tail'],
            plot_param=[plot_param])



def extract_epochs_over_trials_one_side_one_body_part(files_list_DLC,files_list_Laser,direct,folder,scale_pix_to_cm,
                                accep_interval_range,misdetection_dict,pre_interval,interval,post_interval,
                               fps,n_timebin,window_veloc,window_pos,cor,body_part,plot_param,left_or_right = 'right'):

    '''Return all the epochs of all similar trials for one mouse.
    
    Parameters
    ----------
    
    files_list : list (str)
        List of paths to all DLC .csv files for sessions
        
    files_list_laser : list (str)
        List of paths to all .csv files that have (start,end) laser times.
        
    direct : str
        Path to the parent directory containing all folders
        
    spont : str
        Either 'n' --> for trials containing laser stimulation
        or 'y' --> for spontaneous trials with no laser
    
    no_sample : int (optional)
        Number of samples to extract from one spontaneous session. Only necessary for spont = 'y' cases.
        
    accep_interval_range: tuple (int)
        Smallest and largest acceptable trial sizes in timebins
        
    
    interval_dict : dictionary
        Contains {'pre_interval',interval,'post_interval'}
        Number of timebins for the pre-laser, laser and post-laser epochs
        
    n_timebin : int
        Number of timebins for the x derivative to yiels velocity
    
    window_veloc : int
        Velocity moving average window in timebins
    
    window_pos : int
        Position moving average window in timebins
        
    Returns
    -------
    
    epochs : 2D-array
        (pre | Laser ON | post) epochs with shape (n_trials, pre_interval+post_interval+interval+1)
    
    '''


    epochs = np.empty((0,pre_interval+interval+post_interval+1))


    for i in range(0,len(files_list_DLC)):
        
        print('session {} out of {}'.format(i+1,len(files_list_DLC)))
        file_path_DLC = os.path.join(direct,folder,'DLC',files_list_DLC[i])
        df = read_DLC(file_path_DLC,scale_pix_to_cm)

        right,left = position_r_l(df, window_pos,misdetection_dict,cor,body_part,plot_param)

        if left_or_right == 'left':
            variable = left
        else:
            variable = right
        file_path_Laser = os.path.join(direct,folder,'Laser',files_list_Laser[i])
        laser_t = read_laser(file_path_Laser)
        bins  = np.copy(laser_t.values).astype(int)
        epochs_trial,n_trials,take = extract_epochs(bins,variable,*accep_interval_range,pre_interval,interval,post_interval)

        epochs = np.append(epochs,epochs_trial,axis = 0)


    return epochs
    