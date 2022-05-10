
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
#from colour import Color
from pathlib import Path
import csv
from File_hierarchy import *
import xlrd

font = FontProperties()
font.set_family('sans-serif')
font.set_name('Times New Roman')
# font.set_style('italic')
font.set_size('28')
# font.set_weight('bold')

font_label = FontProperties()
font_label.set_family('serif')
font_label.set_name('Times New Roman')
# font_label.set_style('italic')
font_label.set_size('20')
# font_label.set_weight('bold')


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

    ax.get_xaxis().set_tick_params(direction='out', labelsize=30, length=10)
    ax.xaxis.set_ticks_position('bottom')
    ax.get_yaxis().set_tick_params(direction='out', labelsize=30, length=10)
    ax.yaxis.set_ticks_position('left')

def find_header_line_laser_file_csv(path):
    
    """ Find which line includes the headers of 'ON' and 'OFF in the csv file"""
    
    # open file in read mode
    with open(path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv using reader object
        for i, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if row == ['ON', 'OFF']:
                return i
                break
    return None

def find_header_line_laser_file_excel(path):
    
    """ Find which line includes the headers of 'ON' and 'OFF in the excel file"""
    
    df = pd.read_excel(path, header = None)

    for i in df.index:
        
        if tuple( df.loc[i]) == ('ON', 'OFF') :
            
            return i
            break
            
    return None

def get_extension(path):

    ''' return file extension from path'''
    
    return os.path.splitext(path) [1]

def get_name(path):

    ''' Return filename from path'''
    name = os.path.basename(path)

    return os.path.splitext( name )[0]

def get_laser_files(path):

    ''' Find laser files'''

    files_list = list_all_files(path, extensions = ['.xlsx', '.csv'])
    laser_filtered_files = [ fi for fi in files_list 
                                if 'Laser'.lower() in get_name(fi).split('_')[-1].lower()]
    laser_filtered_files.sort()
    return laser_filtered_files

def get_DLC_files(path):

    ''' Find DLC files'''

    files_list = list_all_files(path, extensions = ['.xlsx', '.csv'])
    DLC_filtered_files = [ fi for fi in files_list 
                                if 'modif'.lower() not in get_name(fi).split('_')[-1].lower()]
    DLC_filtered_files.sort()

    return DLC_filtered_files

def check_DLC_corresponds_to_laser(filepath_laser, filepath_DLC):

    """ Return true if the filename except for the last elemet is the same 
        between the DLC and the laser files.
    """

    filename_DLC = os.path.splitext(os.path.basename(filepath_DLC))[0]
    filename_laser = os.path.splitext(os.path.basename(filepath_laser))[0]
    print('DLC:', filename_DLC)
    print('Laser:', filename_laser)
    
    if not '_'.join( filename_DLC.split('_')[:-1]) == '_'.join(filename_DLC.split('_')[:-1]):

        raise ValueError (" Laser and DLC files dont' match!")

def read_DLC(file_name, scale_pix_to_cm):

    '''Read DeepLabCut Data.'''

    if get_extension(file_name) == '.csv':

         df = pd.read_csv(file_name, header=[1, 2]) * scale_pix_to_cm  # scale to cm

    elif get_extension(file_name) in ['.xlsx']:
    
        df = pd.read_excel(file_name, header=[1, 2]) * scale_pix_to_cm  # scale to cm
    
    else: 

        raise ValueError ( "DLC data format not right! Should be '.xlsx' or 'csv' ")

    return df


def read_laser(laser_file_name, DLC_file_name):

    '''Read laser onset offset times for Square pulse stimulation'''

    check_DLC_corresponds_to_laser(laser_file_name, DLC_file_name)
    extension = get_extension(laser_file_name)     

    if extension == '.csv':

        n_rows_to_skip = find_header_line_laser_file_csv(laser_file_name)
        laser_t = pd.read_csv(laser_file_name, skiprows = n_rows_to_skip)

    elif extension in ['.xlsx' ]:
        
        
        n_rows_to_skip = find_header_line_laser_file_excel(laser_file_name)
        print(n_rows_to_skip)
        laser_t = pd.read_excel(laser_file_name, skiprows = n_rows_to_skip)
    
    else: 

        raise ValueError ( "laser data format not right! Should be '.xlsx' or 'csv' ")
    
    return laser_t


def read_laser_beta_stim(laser_file_name):

    '''Read laser onset offset times for Beta pulse stimulation'''

    laser_t = pd.read_excel(laser_file_name, skiprows=4)
    onset = np.array(laser_t['ON'].values)
    # find the starts of sessions if
    starts = np.where(shift(onset, -1, cval=0)-onset > min_t_between_stim)
    # they are not oscilations of a beta wave

    temp = np.zeros((len(onset[starts[0]])+1,))
    # save the first detected laser as the dtart of the first session
    temp[0] = onset[0]

    # the rest are as detected above
    temp[1:temp.shape[0], ] = onset[starts[0]+1]

    df = pd.DataFrame(({'ON': temp, 'OFF': temp+interval}))
    
    return df


def list_all_files(path, extensions = ['.xlsx', '.csv']):
    
    '''List all the files with this extention in this directory'''
    
    if os.path.exists(path):
        
        files = [x for x in os.listdir(path) if not x.startswith('.')]
        files.sort()
    
        return [ fi for fi in files 
                    if (fi.endswith( tuple(extensions) ))]
    else:
        
        return []

def convert_csv_to_xlsx(path):

    ''' Check if a .xlsx version of all the .csv files exists, 
    if not convert to this format
    and remove the .csv to save space
    '''

    files = [x for x in os.listdir(path) if not x.startswith('.')]
    files.sort()
    csv_files = list(filter(lambda x: ".csv" in x, files))
    # remove extensions to be able to compare lists
    csv_file_names = [x.replace(".csv", "") for x in csv_files]
    xlsx_files = list(filter(lambda x: '.xlsx' in x, files))
    xlsx_file_names = [x.replace(".xlsx", "") for x in xlsx_files]
    # if most files are in csv convert them to xlsx
    if not set(csv_file_names) < set(xlsx_file_names):

        for filepath_in in csv_files:
            name = os.path.join(path, filepath_in)

            try:
                pd.read_csv(name, delimiter=",").to_excel(os.path.join(
                    path, filepath_in.replace(".csv", ".xlsx")), header=True, index=False)
            except pd.errors.ParserError:  # it must be a laser file
                pd.read_csv(name, delimiter=",", skiprows=4).to_excel(os.path.join(
                    path, filepath_in.replace(".csv", ".xlsx")), startrow=4, header=True, index=False)

            os.remove(name)  # remove the csv file.


def csv_from_excel(filepath):
    
    ''' Convert xlsx file to csv'''
    
    pd.read_excel(filepath).to_csv(os.path.join(path, filepath.replace(".xlsx", ".csv")), 
                                               header=True, index=False)

    
# def build_filePath_list(path):
    
#     '''move all non csv xlsx files out of the Laser and DLC subdirectory'''
    
    
#     for (dirpath, dirnames, filenames) in os.walk(path):
#         for dirname in dirnames:
#         dirname == 'Laser' or dirname == 'DLC'
#         for f in filenames:
#             dirname = os.path.dirname(f)
#             print(f)
#             if ((dirname == 'Laser' or dirname == 'DLC') and 
#                 os.path.splitext(f) [1] not in ['.csv', 'xlsx']):
                
#                 print('moving', f)
#                 os.rename(os.path.join(dirpath, dirname, f),
#                           os.path.join(dirpath, f))
    
# path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2/ChR2/Mouse_116/STN/Spontaneous'
build_filePath_list(path)

def save_npz(pre_direct, mouse_type, opto_par, stim_loc, stim_type, pulse_inten, fps, window, n_timebin, file_name_ext,
             epochs_all_mice, epochs_mean_each_mouse, epochs_spont_all_mice, pre_info,
             cor, body_part, plot_param):
    '''Save the trial epochs in one .npz file.

    Parameters
    ----------
    pre_direct : str
            path to rooot directory
    mouse_type : str
            mouse type such as 'FoxP2' or ..
    opto_par: str
            "ChR2" --> for ChR2 injected animals
            "Control" --> for Control group
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   
    stim_type : str
            the folder with the corresponding experiment protocol e.g. "Square_1_mW".
    pulse_inten : int
            Pulse intensity in mW
    fps : int
            Frame per second
    window : int
            Moving average window for position
    n_timebin : int
            Number of frames to take derivative over
    file_name_ext : str
            File name extension
    epochs_all_mice : 2D-array(float)
            (pre | Laser ON | post) epochs with shape (n_trials, pre_interval+post_interval+interval+1)
    epochs_mean_each_mouse : 2D-array(float)
            average of (pre | Laser ON | post) epochs with shape (n_animals, 3) 
            averaged for individual animals
    epochs_spont_all_mice : 2D-array(float)
            (pre | Laser ON | post) spontaneous epochs with shape (n_trials, pre_interval+post_interval+interval+1)
    pre_info : 2D array(float)
            Three columns with averages of : pre_x , pre_v, pre_acceleration over pre_stim_interval duration.
            with shape (n_trials, 3)
    cor_list : list(str)
            list of available coordinates e.g. 'x', 'y'
    body_part_list : list(str)
            list of DLC labeled body parts 
    plot_param_list : list(str)
            List of available parameters for analysis e.g position or velocity.

    Returns
    -------

    '''

    s = '_'
    s = s.join(body_part)

    file_name = (mouse_type
                 + '_' + opto_par 
                 + '_' + pulse_inten 
                 + file_name_ext 
                 + "_mov_aver=" + str(int(window/fps*1000)) 
                 + "_n_t=" + str(int(n_timebin/fps*1000))
                 + '_' + cor
                 + '_' + plot_param
                 + '_' + s)
    Directory.create_dir_if_not_exist(os.path.join(pre_direct, 'data_npz', stim_loc, stim_type, opto_par))
    np.savez(os.path.join(pre_direct, 'data_npz', stim_loc, stim_type, opto_par, file_name),
             epochs_all_mice=epochs_all_mice,
             epochs_mean_each_mouse=epochs_mean_each_mouse,
             epochs_spont_all_mice=epochs_spont_all_mice,
             avg_pre_stim_position=pre_info[:, 0],
             avg_pre_stim_velocity=pre_info[:, 1],
             avg_pre_stim_acc=pre_info[:, 2],
             cor=[cor],
             body_part=body_part,
             plot_param=[plot_param])


def moving_average_array(X, n):
    '''Return the moving average over X with window n without changing dimesions of X'''

    z2 = np.cumsum(np.pad(X, (n, 0), 'constant', constant_values=0))
    z1 = np.cumsum(np.pad(X, (0, n), 'constant', constant_values=X[-1]))
    return (z1-z2)[(n-1):-1]/n


def align_right_left(right, left):

    '''Correct if for any reason there has been a shift between labelings of right and left side

    Parameters
    ----------
    right : 1-D array
            position in time detected from the right camera
    left : 1-D array
            position in time detected from the left camera
    '''
    delta = np.average(right-left)

    if delta > 0:  # if the right is ahead
        right -= delta/2
        left += delta/2

    else:  # if the left is ahead
        right += delta/2
        left -= delta/2

    return right, left


def derivative(x, delta_t, fps):
    '''Take the derivative over delta_t.'''

    derivative_out = (x - shift(x, delta_t, cval=x[0])) / (delta_t/fps)
    # got crazy zith keyerror -1
#     print(derivative_out[-1])
#     return shift(derivative_out,-int(delta_t/2),cval= derivative_out[-1])
    return shift(derivative_out, -int(delta_t/2), cval=derivative_out[len(derivative_out)-1])


def derivative_mov_ave(x, delta_t, window_veloc, fps):
    '''Take the derivative with delta_t and do a moving average.'''

    derivative_out = (x - shift(x, delta_t, cval=x[0])) / (delta_t/fps)
    dx_dt = shift(derivative_out, -int(delta_t/2),
                  cval=derivative_out[len(derivative_out)-1])
    # return the moving average
    return moving_average_array(dx_dt, window_veloc)
#     return dx_dt # if you don't want to do a moving average


def input_plot(df, laser_t, mouse_type, mouse_no, trial_no, opto_par, pre_direct, exp_dict, t_window_dict, misdetection_dict, save_as_format='.pdf'):
    """Get the specifics of the plot as input and call the corresponding plot function to plot the session.

    Parameters
    ----------
    df : Dataframe
            Dataframe derived from DLC for one session of one animal
    laser_t : Dataframe
            Dataframe with laser onset and offset times as columns
    mouse_type : str
            Mouse type. For example `FoxP2`
    mouse_no : int
            Mouse number
    trial_no : int
            Trial number specified in file name
    opto_par : str
            Optogenetic specification. For example `ChR2` or `Control`
    pre_direct : str
            Path to the root directory of the project
    exp_dict : dict-like
            Experiment dictionary for cor_list, body_part_list and plot_param_list
    t_window_dict : dict-like
            Dictionary of time constants
    misdetection_dict : dict-like 
            Dictionary for misdetection algorithm constants
    save_as_format : str, optional
            The extension of the saved figure 

    Returns
    -------
    None

    """
    study_param_dict = get_input_cor_body_part(**exp_dict)

    print(" 1. Right & Left \n 2. Average of both")
    Average_sep_plot = int(input())  # ask what body part to plot

    if Average_sep_plot == 2:
        print(Average_sep_plot)
        plot_what_which_where(df, laser_t, mouse_type, mouse_no, trial_no, opto_par, pre_direct, misdetection_dict,
                              **study_param_dict, **t_window_dict, save_as_format='.pdf')
    else:
        plot_what_which_where_r_l(df, laser_t, mouse_type, mouse_no, trial_no, opto_par, pre_direct, misdetection_dict,
                                  **study_param_dict, **t_window_dict, save_as_format='.pdf')


def get_input_cor_body_part(cor_list, body_part_list, plot_param_list):
    '''Ask for the body part and coordinate from user.

    Parameters
    ----------

    cor_list : list(str)
            list of available coordinates e.g. 'x', 'y'

    body_part_list : list(str)
            list of DLC labeled body parts 

    plot_param_list : list(str)
            List of available parameters for analysis e.g position or velocity.

    Returns
    -------

    study_param_dict : dict-like
            selected items from the options in the inputs


    '''

    print("Select for which parts you want to see the pre/On/post: \n")
    print(" 1. Tail \n 2. Nose \n 3. Fore Limb \n 4. Hind Limb")
    # ask what body part to plot
    where_plot = [int(x)-1 for x in input().split()]
    print(" 1. X \n 2. Y ")
    which_plot = int(input())-1  # ask what body part to plot
    print(" 1. Position \n 2. Velocity \n ")
    what_plot = int(input())-1  # ask what body part to plot
    study_param_dict = {'cor': cor_list[which_plot],
                        'body_part': body_part_list[where_plot],
                        'plot_param': plot_param_list[what_plot]}
    return study_param_dict


def produce_random_bins_for_spont(max_time, n_sample, pre_interval, interval, post_interval, max_distance, min_distance, n_trials_spont):
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

    bins : 2D-array(int)
            an array with two columns each row containing (start,end) of each trial

    '''

    half_max_distance = int(max_distance/2)
    # the first timebin eligible for the start of the first timebin
    start = pre_interval + half_max_distance
    # the last timebin eligible for the start of the last trial
    end = max_time - (post_interval+interval+half_max_distance)
    time_points = np.arange(start, end-((end-start) % (min_distance+half_max_distance)),
                            min_distance+half_max_distance)  # produce the grid
    perturb = np.random.randint(-half_max_distance, half_max_distance, size=(
        n_sample, len(time_points)))  # produce the random perturbations
    start_arr = np.repeat(time_points.reshape(
        1, len(time_points)), n_sample, axis=0)+perturb  # perturb the starting points
    starts = (np.array(start_arr.ravel())).reshape(len(start_arr.ravel()), 1)
    ends = starts+interval
    # stack start and ends of bins
    bins = np.concatenate((starts, ends), axis=1)

    return bins


def compare_r_l_correct_misdetect(right_p, left_p, acc_deviance, t_s,
                                  internal_ctr_dev=0.5, percent_thresh_align=0.8, n_iter_jitter=1, jitter_threshold=0.5):
    '''Compare the right and left sides and correct if the difference between 
            the two is more than an acceptable amount. 

    Parameters
    ----------
    right_p : 1-D array
            Position in time detected from the right camera
    left_p : 1-D array
            Position in time detected from the left camera
    acc_deviance : float 
            Passed within misdetect dict-like. Acceptable deviance of the right and left side detections
    t_s : int
            The number of procedeeing and post time bins to average from and correct the misdetection with
    internal_ctr_dev : float
            The threshold (cm) by which it is allowed for left and right detections to differ. 
    percent_thresh_align : int
            The percentage above which it's decided that there is a systematic shift between the right and left side detections.

    Returns 
    -------
    right : 1-D array
            Corrected position in time detected from the right camera
    left : 1-D array
            Corrected position in time detected from the left camera

    '''
    right_x = np.copy(right_p)
    left_x = np.copy(left_p)

    
    delta_x = np.absolute(right_x - left_x) # the difference between detections of right and left cameras
    ind, = np.where(delta_x > internal_ctr_dev)

    # if more than a percentage of detections are not aligned there must be a shift
    if len(ind) > percent_thresh_align * len(right_x):

        print(
            "There's a shift between left and right detections. Don't worry we will fix it!")

        right_x, left_x = align_right_left(right_x, left_x)

    delta_x = np.absolute(right_x - left_x)
    mis_r_l, = np.where(delta_x > acc_deviance)  # spot where
    removed_edge_ind = np.logical_and( 
                            (mis_r_l > (t_s + 1)), 
                            (mis_r_l < (len(right_x)- (t_s - 1)))
                                     )
    mis_r_l = mis_r_l[removed_edge_ind]

    # spot which is more deviant from it's neghbors in time
    print("# inconsistent right left = ", len(mis_r_l))

    if len(mis_r_l) > 0:  # only bother if only you find any mismatches

        compare_within_r = np.zeros((delta_x.shape))
        compare_within_l = np.zeros((delta_x.shape))

        bef_r = np.hstack([np.absolute(np.average(right_x[j-t_s:j-1])-right_x[j]) +
                           np.absolute(np.average(right_x[j+1:j+t_s])-right_x[j]) for j in mis_r_l])/2
        bef_l = np.hstack([np.absolute(np.average(left_x[j-t_s:j-1])-left_x[j]) +
                           np.absolute(np.average(left_x[j+1:j+t_s])-left_x[j]) for j in mis_r_l])/2

        compare_within_r[mis_r_l] = bef_r
        compare_within_l[mis_r_l] = bef_l

        ind_right_larger, = np.where(compare_within_l < compare_within_r)
        ind_left_larger, = np.where(compare_within_l > compare_within_r)

        temp_r = np.in1d(mis_r_l, ind_right_larger)
        # where there's mismatch and it's annonated to the right one
        ind_r_corr = mis_r_l[temp_r]

        temp_l = np.in1d(mis_r_l, ind_left_larger)
        # where there's mismatch and it's annonated to the left one
        ind_l_corr = mis_r_l[temp_l]

        # correct based on the findings
        # set to the average of the other side rather than the same track because there's a better chance the
        # mistake happens again in the proximity of the same side
        
        if len(ind_l_corr) > 0:
            left_x[ind_l_corr] = np.hstack([np.average(right_x[j-t_s:j-1]) +
                                            np.average(right_x[j+1:j+t_s]) for j in ind_l_corr])/2
        if len(ind_r_corr) > 0:
            right_x[ind_r_corr] = np.hstack([np.average(left_x[j-t_s:j-1]) +
                                             np.average(left_x[j+1:j+t_s]) for j in ind_r_corr])/2

    return np.array(right_x).reshape((-1, 1)), np.array(left_x).reshape((-1, 1))


def correct_labeling_jitter(x, jitter_threshold, n_iter_jitter, t_s, acc_deviance=0,
                            internal_ctr_dev=0, percent_thresh_align=0):
    '''Correct the detections exceeding the max speed of the mouse.

    The detected jitters are replaced by the average of before and after time stamps.

    Parameters
    ----------
    x : 1-D array
            The time series that needs correction (usually the average of right and left cameras).

    jitter_threshold : float
            The allowed maximum movement in one timebin with the max allowed velocity.

    n_iter_jitter : int 
            Number of times to repeat the correction algorithm.

    t_s : int
            The number of timebins to average from when correcting the jitter point.

    Returns
    -------
    x : 1-D array
            The corrected time series.

    '''
    for i in range(n_iter_jitter):

        deltas = x - shift(x, 1, cval=x[0])

        ind, = np.where(np.absolute(deltas) > jitter_threshold)
        ind = ind[(ind > t_s + 1) & (ind < (len(x) - t_s - 1))]

        if len(ind) > 0:  # if jumped in detection set to the mean of <t_s> previous and next detections

            x[ind] = np.hstack([np.average(x[j - t_s: j - 1]) +
                                np.average(x[j + 1: j + t_s]) for j in ind]) / 2

            print("# jitter in mean(righ,left)  = ", len(ind))

#     print("nan found? How many? ",sum(np.isnan(x)))
#     print(x[ind])

    return x


def average_position_r_l(df, window, misdetection_dict, cor, body_part, plot_param):
    '''Return three-level correction of the average between right and left side for multiple selected body-parts.

    Extract the right and left trace for each selected body part, correct right and left traces with 
    "compare_r_l_correct_misdetect", then average over right and left and remove jitters by calling
    "correct_labeling_jitter". Then return the moving average of all selected body-parts.

    Parameters
    ----------

    df : dataframe
            Dataframe derived from DLC for one session of one animal

    window : int
            Moving average window for smoothing the position time series

    misdetection_dict : dict-like
            dict-like containing constants for misdetection corrections

    cor : str
            Which coordinate to work with. e.g. 'x' or 'y'

    body_part : list(str)
            Which body parts to average over. 

    plot_param : str
            Parameter for analysis e.g. position or velocity.

    Returns
    -------
    corrected_averaged : 1-D array 
            Moving average of all selected body-parts.

    '''

    averaged_position = np.zeros(( df[('r' + body_part[0], cor)].values.shape))

    for param in body_part:  # average over body parts

        print("Looking at ---", param, '---')
        
        right_x = np.copy(df[('r' + param, cor)].values)
        left_x = np.copy(df[('l' + param, cor)].values)
        right, left = compare_r_l_correct_misdetect(
            right_x, left_x, **misdetection_dict)  # first compare r and l
        left_right_corrected_averaged = np.average(
            np.concatenate((right, left), axis=1), axis=1)
        # (this corrects when the misdetection happened for both left and right because
        # left and right have already been aligned together)
        averaged_position += correct_labeling_jitter(
            left_right_corrected_averaged, **misdetection_dict)
#         averaged_position += correct_labeling_jitter(left_x,jitter_threshold,n_iter_jitter, t_s)

    averaged_position = averaged_position/len(body_part)
    corrected_averaged = moving_average_array(averaged_position, window)
    return corrected_averaged


def position_r_l(df, window, misdetection_dict, cor, body_part, plot_param):
    '''Remove jitters with "correct_labeling_jitter" on right and left on the 
    selected body parts and return right and left separately.

    Parameters
    ----------

    df : dataframe
            Dataframe derived from DLC for one session of one animal

    window : int
            Moving average window for smoothing the position time series

    misdetection_dict : dict-like 
            dict-like containing constants for misdetection corrections

    cor : str
            Which coordinate to work with. e.g. 'x' or 'y'

    body_part : list(str)
            Which body parts to average over. 

    plot_param : str
            Parameter for analysis e.g. position or velocity.

    Returns
    -------
    right_corrected : 1-D array 
            Moving average of right side for all selected body-parts

    left_corrected : 1-D array 
            Moving average of left side for all selected body-parts
            '''

    averaged_position_r = np.zeros((df[('r'+body_part[0], cor)].values.shape))
    averaged_position_l = np.zeros((df[('r'+body_part[0], cor)].values.shape))

    for param in body_part:  # average over body parts
        right_x = np.copy(df[('r'+param, cor)].values)
        left_x = np.copy(df[('l'+param, cor)].values)
        averaged_position_l += correct_labeling_jitter(
            left_x, **misdetection_dict)
        averaged_position_r += correct_labeling_jitter(
            right_x, **misdetection_dict)

    averaged_position_r = averaged_position_r/len(body_part)
    averaged_position_l = averaged_position_l/len(body_part)
    right_corrected = moving_average_array(averaged_position_r, window)
    left_corrected = moving_average_array(averaged_position_l, window)
    return right_corrected, left_corrected


def min_and_mean_on_off(epochs, measure, pre_interval, interval, post_interval, pre_stim_inter):
    '''Report the min or mean (specified by measure) velocity in the off-on periods.

    Parameters
    ----------

    epochs : 2D array(float)
            Trial variables in rows. n_col = pre_interval + interval + post_interval 

    measure : str 
            'mean' or 'min'

    pre_interval : int 
            Number of time bins to be taken into account before laser onset 

    interval : int 
            Laser duration in timebins.

    post_interval : int
            Number of time bins to be taken into account after laser onset 

    pre_stim_inter : int
            Unpacked in dict-like. Not used here.

    Returns
    -------

    average_of_off_on_off : 2D array
            array of 3 columns. first column is the aversge of pre-intevsl,
            second the 'measure' during laser interval
            third again the average of post

    '''

    if measure == 'Mean':
        try:
            pre = np.average(epochs[:, 0:pre_interval],
                             axis=1).reshape(epochs.shape[0], 1)
            ON = np.average(
                epochs[:, pre_interval:pre_interval+interval], axis=1).reshape(epochs.shape[0], 1)
            post = np.average(epochs[:, pre_interval+interval:pre_interval +
                              interval+post_interval], axis=1).reshape(epochs.shape[0], 1)

        except ZeroDivisionError:
            pre = np.zeros((epochs.shape[0], 1))
            ON = np.zeros((epochs.shape[0], 1))
            post = np.zeros((epochs.shape[0], 1))
    elif measure == 'Min':
        try:
            pre = np.average(epochs[:, 0:pre_interval],
                             axis=1).reshape(epochs.shape[0], 1)
            ON = np.min(epochs[:, pre_interval:pre_interval +
                        interval], axis=1).reshape(epochs.shape[0], 1)
            post = np.average(epochs[:, pre_interval+interval:pre_interval +
                              interval+post_interval], axis=1).reshape(epochs.shape[0], 1)
        except ZeroDivisionError:
            pre = np.zeros((epochs.shape[0], 1))
            ON = np.zeros((epochs.shape[0], 1))
            post = np.zeros((epochs.shape[0], 1))
    average_of_off_on_off = np.concatenate((pre, ON, post), axis=1)
    return average_of_off_on_off


def plot_what_which_where_r_l(df, laser_t, mouse_type, mouse_no, trial_no, opto_par, pre_direct, misdetection_dict,
                              cor, body_part, plot_param,
                              fps, n_timebin, window_pos, window_veloc, save_as_format='.pdf'):
    '''Choose which body part/what measure/ x or y to see the left vs. right traces.'''

    label_1 = "Right"
    label_2 = "left "
    time_series = df.index / fps  # time axis in seconds for stimulation trial
    trial_time = max(time_series)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)

    if plot_param == 'velocity':

        velocity_r = derivative_mov_ave(
            df[('r'+body_part[0], cor)], n_timebin, window_veloc, fps)
        velocity_l = derivative_mov_ave(
            df[('l'+body_part[0], cor)], n_timebin, window_veloc, fps)
        plt.plot(time_series+n_timebin/fps, velocity_r,
                 'navy', label=label_1, linewidth=0.8)
        plt.plot(time_series+n_timebin/fps, velocity_l,
                 'orangered', label=label_2, linewidth=0.8)
        plt.xlim(n_timebin/fps, trial_time/2)
        plt.ylabel(cor + " Velocity (cm/s)", fontproperties=font_label)
        min_end = min(min(velocity_r), min(velocity_l))
        max_end = max(max(velocity_r), max(velocity_l))
        # plot zero velocity threshold
        plt.axhline(y=0, color='r', linestyle='--', linewidth=0.9)

    elif plot_param == "acceleration":

        accel_r = derivative_mov_ave(derivative_mov_ave(df[('r'+body_part[0], cor)], n_timebin, window_veloc, fps),
                                     n_timebin, window_veloc, fps)
        accel_l = derivative_mov_ave(derivative_mov_ave(df[('l'+body_part[0], cor)], n_timebin, window_veloc, fps),
                                     n_timebin, window_veloc, fps)
        t_shift = n_timebin/fps/2
        plt.plot(time_series+t_shift+t_shift, accel_r,
                 'navy', label=label_1, linewidth=0.8)
        plt.plot(time_series+t_shift+t_shift, accel_l,
                 'orangered', label=label_2, linewidth=0.8)
        plt.xlim(t_shift+t_shift, trial_time/2)
        plt.ylabel(cor + " Acceleration (cm/s**2)", fontproperties=font_label)
        min_end = min(min(accel_r), min(accel_l))
        max_end = max(max(accel_r), max(accel_l))
        # plot zero velocity threshold
        plt.axhline(y=0, color='r', linestyle='--', linewidth=0.9)

    else:
        r, l = compare_r_l_correct_misdetect(
            df[('r'+body_part[0], cor)].values, df[('l'+body_part[0], cor)].values, **misdetection_dict)
        plt.plot(time_series, r, 'navy',  label=label_1, linewidth=0.8)
        plt.plot(time_series, l, 'orange', label=label_2, linewidth=0.8)

        plt.ylabel(cor + " (cm)", fontproperties=font_label)
        min_end = min(min(df[('r'+body_part[0], cor)]),
                      min(df[('l'+body_part[0], cor)]))
        max_end = max(max(df[('r'+body_part[0], cor)]),
                      max(df[('l'+body_part[0], cor)]))

    set_ticks(ax)
    plt.ylim(min_end, max_end)
    plt.xlabel("Time(s)", fontproperties=font_label)
    plt.title(mouse_type+' ' + opto_par+' #' +
              str(mouse_no), fontproperties=font)
    plt.legend(fontsize=20)
    for i in range(len(laser_t['ON'].values)):
        plt.axvspan(laser_t['ON'].values[i]/fps,
                    laser_t['OFF'].values[i]/fps, alpha=0.4, color='lightskyblue')

    plt.savefig(os.path.join(pre_direct, "One_session", 'Mouse_trial'+str(trial_no)+'_mouse_' + str(mouse_no)+'_' +
                             body_part[0] + '_' + plot_param + '_' + cor + save_as_format), bbox_inches='tight', orientation='landscape', dpi=300)


def plot_what_which_where(df, laser_t, mouse_type, mouse_no, trial_no, opto_par, pre_direct, misdetection_dict,
                          cor, body_part, plot_param,
                          fps, n_timebin, window_pos, window_veloc, save_as_format='.pdf'):
    '''Choose to see averaged velocity, position or acceleration for a chosen combination of body_parts
            for either x or y coordiante.
    '''
    s = '_'
    s = s.join(body_part)
    time_series = df.index / fps  # time axis in seconds for stimulation trial
    trial_time = max(time_series)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)

    averaged_pos = average_position_r_l(
        df, window_pos, misdetection_dict, cor, body_part, plot_param)

    time_series = df.index / fps

#   ind = np.logical_and(time_series>23,time_series<55)
#   pd.DataFrame(averaged_pos[ind]).to_csv("/home/shiva/Desktop/ss.csv")

    trial_time = max(time_series)

    if plot_param == "velocity":

        velocity = derivative_mov_ave(
            averaged_pos, n_timebin, window_veloc, fps)
        plt.plot(time_series+n_timebin/fps/2, velocity,
                 'k', linewidth=2, label=s)  # plot all body
        plt.ylabel(cor + " Velocity (cm/s)", fontproperties=font_label)
        min_end = min(velocity)
        max_end = max(velocity)
        # plot zero velocity threshold
        plt.axhline(y=0, color='r', linestyle='--', linewidth=0.9)

    elif plot_param == "acceleration":

        accel = derivative_mov_ave(derivative_mov_ave(
            averaged_pos, n_timebin, window_veloc, fps), n_timebin, window_veloc, fps)
        t_shift = n_timebin/fps/2
        plt.plot(time_series+t_shift+t_shift, accel, 'k',
                 linewidth=2, label=s)  # plot all body
        plt.xlim(t_shift+t_shift, trial_time/2)
        plt.ylabel(cor + " Acceleration (cm/(s2))", fontproperties=font_label)
        min_end = min(accel)
        max_end = max(accel)
        # plot zero velocity threshold
        plt.axhline(y=0, color='r', linestyle='--', linewidth=0.9)

    else:
        plt.plot(time_series, averaged_pos, '-k', linewidth=1,
                 label=s, markersize=1)  # plot all body
        plt.ylabel(cor + " (cm)", fontproperties=font_label)
        min_end = min(averaged_pos)
        max_end = max(averaged_pos)

    set_ticks(ax)

    plt.xlabel("Time(s)", fontproperties=font_label)
    plt.title("Average "+mouse_type+' ' + opto_par +
              ' #'+str(mouse_no), fontproperties=font)
    plt.legend(fontsize=20)
    plt.ylim(min_end, max_end)

    for i in range(len(laser_t['ON'].values)):
        plt.axvspan(laser_t['ON'].values[i]/fps,
                    laser_t['OFF'].values[i]/fps, alpha=0.4, color='lightskyblue')
#     plt.vlines(laser_t['ON']/fps,min_end,max_end, color = 'darkskyblue', linewidth = 0.4) # plot stimulus onsets
#     plt.vlines(laser_t['OFF']/fps,min_end,max_end, color = 'darkskyblue', linewidth = 0.4) # plot stimulus offsets
    plt.savefig(os.path.join(pre_direct, "One_session", 'Mouse_trial'+str(trial_no)+'_mouse_' + str(mouse_no)+'_' + cor + '_' +
                             plot_param + '_averaged_'+s+save_as_format), bbox_inches='tight', orientation='landscape', dpi=300)


def plot_pre_on_post(ax, pre_direct, mouse_type, opto_par, stim_loc, stim_type, epochs, 
                     epochs_spont, treadmil_velocity, ylim,
                     fps, n_timebin, window_pos, window_veloc,
                     cor, body_part, plot_param,
                     pre_interval, interval, post_interval, pre_stim_inter,
                     average='Averg_trials_all_mice', c_laser='deepskyblue',
                     c_spont='k', save_as_format='.pdf', title = True,
                     plot_spont = True, label = None, annotate_n = False, axvspan = True):
    
    """Plot (pre Laser | Laser | post Laser) velocity/position/acceleration comparison between laser and spontaneous trials.

    Parameters 
    ----------
    ax : ax
    pre_direct : str
            Path to project directory
    mouse_type : str
            Type of the animal that makes it different in the experiment e.g. for transgenic mice 'FoxP@' or 'Vglut2'.
    opto_par : str
            Optogentic parameter distinguishing injections. e.g. 'Control' or 'ChR2'.
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   
    stim_type : str 
            folder containing this specific protocol's sessions e.g. 'Square_1_mW'
    epochs : 2D array(float)
            Trials containtning laser stimulation aligned in rows. n_col is equal to pre_interval + interval + post_interval
    epochs_spont : 2D array(float)
            Spontaneous trials aligned in rows. n_col is equal to pre_interval + interval + post_interval
    treadmill_velocity : float
            Treadmill velocity in cm/s
    ylim : list(float)
            Y axis limits for the figure given as a list : [ymin, ymax]
    fps : int
            frame per second of the video
    n_timebin : int
            number of timebins for taking derivatives
    window_pos : int
            Moving average window for position
    window_veloc : int
            Moving average window for velocity
    cor : str
            Which coordinate to work with. e.g. 'x' or 'y'
    body_part : list(str)
            Which body parts to average over. 
    plot_param : str
            Parameter for analysis e.g. position or velocity.
    pre_interval : int 
            Number of time bins to be taken into account before laser onset 
    interval : int 
            Laser duration in timebins.
    post_interval : int
            Number of time bins to be taken into account after laser onset 
    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info
    average : str, optional
            Parameter specifying the output figure. Here default set to 'Averg_trials_all_mice'.
    c_laser : str, optional
            color for plotting trials with laser stimulation. Default value is set to 'deepskyblue'.
    c_spont : str, optional
            color for plotting trials with laser stimulation. Default value is set to 'k' which is black.
    save_as_format : str, optional
            Format to save the figure with. Default set to  '.pdf'

    Returns
    -------
    None

    """


#     epochs_mean = np.average(epochs, axis = 0) # average over different stimuli
#     epochs_mean_spont = np.average(epochs_spont, axis = 0) # average over different stimuli

#     epochs_sem = stats.sem(epochs, axis=0) # SEM
#     epochs_sem_spont = stats.sem(epochs, axis=0) # SEM

    (epochs_mean, epochs_mean_spont, 
    confidence_inter, confidence_inter_spont) = epochs_stats(epochs, epochs_spont)    
    
    time_series = np.arange(-pre_interval, 
                            interval + post_interval + 1)/fps
    
    ax, label = handle_legend_label(ax, label, body_part, epochs, annotate_n)
        
    line_1, = ax.plot(time_series, epochs_mean, color = c_laser, label = label,
                     linestyle='-', linewidth=2)  # , marker='o',markersize=1)
    
    ax.fill_between(time_series, confidence_inter[:, 0],  confidence_inter[:, 1], 
                    color=c_laser, alpha=0.2)
    
    if plot_spont:
        
        line_2, = ax.plot(time_series, epochs_mean_spont,
                         color=c_spont, label="Spontaneous")
        ax.fill_between(time_series, confidence_inter_spont[:, 0],  confidence_inter_spont[:, 1], 
                        color=c_spont, alpha=0.2)
    if axvspan:
        ax.axvspan(0, interval/fps, alpha=0.2, color='lightskyblue')

#     ax.fill_between(time_series, epochs_mean - epochs_sem,  epochs_mean+ epochs_sem,
#                     color='gray', alpha=0.2)
#     ax.fill_between(time_series, epochs_mean_spont - epochs_sem_spont,  epochs_mean_spont+ epochs_sem_spont,
#                    color='b', alpha=0.2)


        
    if plot_param == 'velocity':
        
        ax.set_ylabel(" Velocity (cm/s)").set_fontproperties(font_label)
        
    elif plot_param == 'position':
        
        ax.set_ylabel(" Position (cm)").set_fontproperties(font_label)
        
    else:
        
        ax.set_ylabel(" Acceleration (cm/s**2)").set_fontproperties(font_label)
        
    ax.axhline(y=treadmil_velocity, ls='--', c='red')
    ax.set_xlabel("Time(s)").set_fontproperties(font_label)
    ax.set_ylim(ylim[0], ylim[1])  # set limits
    leg = ax.legend(frameon = False, loc = 'upper left', bbox_to_anchor=(0.1, 1), fontsize = 17)
    p = [l.set_linewidth(3) for l in leg.legendHandles ]
    remove_frame(ax)
    ax = set_ticks(ax)

    if not average:
        
        if title:
            ax.set_title(plot_param
                      + opto_par 
                      + " " + str(mouse_no) 
                      + "(" + mouse_type 
                      + ") #" 
                      + "\n").set_fontproperties(font)
        
        fig.savefig(os.path.join(pre_direct, 
                                 'Compare', 
                                 'Mouse' + str(mouse_no)
                                 + '_' + stim_type
                                 + "_" + opto_par
                                 + '_' + mouse_type
                                 + '_' + cor
                                 + '_' + plot_param 
                                 + '_' + '_'.join(body_part)
                                 + '_timebin_' + str(int(n_timebin*1000/fps))
                                 + '_window_pos_' + str(int(window_pos*1000/fps)) 
                                 + '_window_veloc_' + str(int(window_veloc*1000/fps)) 
                                 + '_pre_post_stim' + save_as_format), 
                    bbox_inches='tight', orientation='landscape', dpi=300)
        
    elif average == 'Averg_trials' and title:  # one mouse
        
        ax.set_title("# " + str(mouse_no) 
                  + "(" + mouse_type 
                  + " " + opto_par + ")" 
                  + "\n" + stim_loc 
                  + "\n" + stim_type.replace('_', ' ').replace('-','.')).set_fontproperties(font)
        
    else:
        
        # if plot_spont :
        #     ax.legend(handles=[line_1, line_2], labels=[label, 'Spontaneous'], 
        #               loc='lower right', fontsize = 12, frameon = False, bbox_to_anchor=(1, -0.02))
        if title:
            ax.set_title(mouse_type 
                         + "\n" 
                         + " Average of all " 
                         +opto_par).set_fontproperties(font)

        ax.get_figure().savefig(os.path.join(pre_direct, 
                                 'Average'
                                 + '_' + stim_type 
                                 + '_' + opto_par
                                 + '_' + mouse_type
                                 + '_' + cor 
                                 + '_' + plot_param
                                 + '_' + '_'.join(body_part)
                                 + '_timebin_'+str(int(n_timebin*1000/fps))
                                 + '_window_pos_' + str(int(window_pos*1000/fps)) 
                                 + '_window_veloc_' +str(int(window_veloc*1000/fps))
                                 + '_pre_post_stim' +save_as_format),
                    bbox_inches='tight', orientation='landscape', dpi=300)
        

        
        return ax
    
def handle_legend_label(ax, label, body_part, epochs, annotate_n):
    
    ''' if annotate is True the number of trials is written as text in the plot
    otherwise it will be reported in the legend'''
    
    label = label or ', '.join(body_part)
    
    if annotate_n:
        
        ax.annotate( 'n = {}'.format(epochs.shape[0]),
                    xy=(0.26,0.85),xycoords='axes fraction', fontsize = 17)
        
    else:
        
        label += ' n=' + str(epochs.shape[0])
        
    return ax, label

def rm_ax_unnecessary_labels_in_subplots(count, n_iter, ax, axis = 'both'):
    
    if axis == 'both':
    
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    elif axis == 'x':
        
        ax.set_xlabel("")
        
    elif axis == 'y':
        
        ax.set_ylabel("")
        
    if count+1 < n_iter:
        # remove the x tick labels except for the bottom plot
        ax.axes.xaxis.set_ticklabels([])

def set_ticks(ax):
    
    """ set tick length and label fontsize"""
    
    ax.get_xaxis().set_tick_params(direction='out', labelsize=20, length=6)
    ax.xaxis.set_ticks_position('bottom')
    ax.get_yaxis().set_tick_params(direction='out', labelsize=20, length=6)
    ax.yaxis.set_ticks_position('left')
    return ax

def epochs_stats(epochs, epochs_spont):
    
    """ Return mean and confidence intervals of epochs"""
    
    confidence_inter = np.empty((0, 2), int)

    if len(epochs.shape) > 1:
        
        epochs_mean = np.average(epochs, axis=0)
        epochs_mean_spont = np.average(epochs_spont, axis=0)

        # calculate the two sided confidence interval for every timestep
        for i in range(epochs.shape[1]):
            
            m = [sms.DescrStatsW(epochs[:, i]).tconfint_mean(
                alpha=0.05, alternative='two-sided')]
            confidence_inter = np.append(
                confidence_inter, [[m[0][0], m[0][1]]], axis=0)
        confidence_inter_spont = np.empty((0, 2), int)

        for i in range(epochs_spont.shape[1]):
            m = [sms.DescrStatsW(epochs_spont[:, i]).tconfint_mean(
                alpha=0.05, alternative='two-sided')]
            confidence_inter_spont = np.append(
                confidence_inter_spont, [[m[0][0], m[0][1]]], axis=0)
    else:
        
        epochs_mean = epochs
        epochs_mean_spont = epochs_spont
        confidence_inter = np.zeros(((epochs_mean.shape[0]), 2))
        confidence_inter_spont = np.zeros((epochs_mean.shape[0], 2))
        
    return epochs_mean, epochs_mean_spont, confidence_inter, confidence_inter_spont
    
# def epochs_single_file(file_name_pos, file_name_spont, file_name_laser):
#     '''Read a single file for a mouse and return the epochs.'''

#     df = read_DLC(file_name_pos, scale_pix_to_cm)
#     df_spont = read_DLC(file_name_spont, scale_pix_to_cm)
#     laser_t = read_laser(file_name_laser, file_name_pos)

#     time_series = df.index / fps  # time axis in seconds for stimulation trial
#     trial_time = max(time_series)

#     # time axis in seconds for spontaneous activity
#     time_series_spont = df_spont.index / fps
#     trial_time_spont = max(time_series_spont)

#     velocity = derivative(average_position_r_l(
#         df, window_pos, misdetection_dict, **study_param_dict).values, n_timebin)  # velocity
#     bins = np.array(laser_t.values).astype(int)
#     epochs = extract_epochs(
#         bins, velocity, *accep_interval_range, **intervals_dict, trial_time )
#     velocity_spont = derivative(average_position_r_l(
#         df_spont, window_pos, misdetection_dict, **study_param_dict).values, n_timebin)  # velocity
#     bins_spont = np.array(laser_t.values).astype(int)  # for now
#     epochs_spont = extract_epochs(
#         bins, velocity_spont, *accep_interval_range, **intervals_dict, trial_time )
#     return epochs, epochs_spont

def remove_frame(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def extract_pre_laser_x_epochs_over_trials(files_list, files_list_laser,
                                           pre_direct, stim_loc, stim_type, scale_pix_to_cm, window_pos, misdetection_dict,
                                           smallest_accep_inter, largest_accep_inter, pre_stim_inter):
    '''Return all the x positions in epochs preceding to a ON-laser of all trials for one mouse.

    Parameters
    ----------

    files_list : list(str)
            List of the file names of the sessions

    files_list_laser : 
            List of file names for laser detections

    pre_direct : str
            Path to the root directory of the project
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   
    stim_type : str
            Folder containing this specific protocol.

    scale_pix_to_cm : float
            Scaling coeficient for pixels to cm.

    window_pos : int
            Moving average window for position

    misdetection_dict : dict-like
            dict-like containing constants for misdetection corrections

    smallest_accep_inter : int
            Smallest acceptable duration of an interval in frames

    largest_accep_inter : int
            Largest acceptable duration of an interval in frames

    pre_stim_inter : int 
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info

    Returns
    -------

    epochs : 2D array(float)
            X positions prior to laser stimulation aligned in rows. n_col = pre_stim_inter

    '''

    epochs = np.empty((0, pre_stim_inter))
    n_files = len(files_list)

    for i, (f_DLC, f_laser) in enumerate( zip(files_list, files_list_laser)):

        print('session {} out of {}'.format(i + 1, n_files))

        file_path_DLC = os.path.join(pre_direct, 
                                     stim_loc, 
                                     stim_type, 
                                     'DLC', 
                                     f_DLC)

        df = read_DLC(file_path_DLC, scale_pix_to_cm)

        study_param_dict = {
                            'cor': 'x', 
                            'body_part': ['Tail', 'Nose'], 
                            'plot_param': 'position'
                            }

        x_average = average_position_r_l(df, 
                                         window_pos, 
                                         misdetection_dict, 
                                         **study_param_dict)  # x positions

        file_path_Laser = os.path.join( direct, 
                                       stim_loc, 
                                        stim_type, 
                                        'Laser', 
                                        f_laser)
        laser_t = read_laser(file_path_Laser, file_path_DLC)

        bins = np.copy(laser_t.values).astype(int)
        duration = bins[:, 1] - bins[:, 0]
        acceptable = np.logical_and(
                                    duration > smallest_accep_inter, 
                                    duration < largest_accep_inter
                                    )
        
        print(len(acceptable) - sum(acceptable),
             ' trials discarded')

        bins = bins[acceptable, :]

        # make an array with indices of laser ON
        take = np.hstack([np.arange(i[0] - pre_stim_inter, i[0])
                         for i in bins[:-1]])

        pre_x_series = x_average[take]
        # calculate the x position relative to the front edge of the treadmill
        epochs_trial = np.absolute(
                                pre_x_series - max(pre_x_series)).\
                                reshape(len(bins) - 1, pre_stim_inter)
        epochs = np.append(epochs, epochs_trial, axis=0)

    return epochs


def plot_every_mouse_mean(epochs_mean_each_mouse):
    '''plot mean of each mouse for epochs OFF-ON-OFF'''

    for i in range(epochs_mean_each_mouse.shape[0]):
        plt.plot(
            median, epochs_mean_each_mouse[i, :], '-', color='gray', marker='.')


def violin_plot_summary(data_init, names, measure):
    ''' plot violin plot for different columns in data'''

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Time Epoch')

    # transform the data into list as to be fed to the violin plot function
    data = list([data_init[:, i] for i in range(data_init.shape[1])])
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(20, 10), sharey=True)

    ax2.set_title(mouse_type)
    parts = ax2.violinplot(data, widths=0.3, showmeans=False,
                           showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('teal')
        pc.set_edgecolor('None')
        pc.set_alpha(.4)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)

    inds = np.arange(1, len(medians) + 1)
    confidence_inter = np.empty((0, 2), int)

    # calculate the two sided confidence interval for every timestep
    for i in range(data_init.shape[1]):
        m = [sms.DescrStatsW(data_init[:, i]).tconfint_mean(
            alpha=0.05, alternative='two-sided')]
        confidence_inter = np.append(
            confidence_inter, [[m[0][0], m[0][1]]], axis=0)

    ax2.scatter(inds, np.average(data_init, axis=0), marker='o',
                color='white', s=20, zorder=3, label='Means')  # measns
    ax2.vlines(inds, quartile1, quartile3, color='k',
               linestyle='-', lw=2, label='Quartiles')  # quartiles

    ax2.errorbar(inds, np.average(data_init, axis=0), yerr=confidence_inter.T, fmt='none', elinewidth=2,
                 capsize=7, marker='o', color='r', label='95% Confidence interval')  # confidence interval

#     ax2.boxplot(inds, data)
#     ax2.vlines(inds, np.min(data_init,axis=0), np.max(data_init,axis=0), color='r', linestyle='-', lw=1,label = 'Whiskers')
#
    # set style for the axes
    set_axis_style(ax2, names)

    # plot each mouse data individually on top of the summary of all
    if measure == 'Mean':
        for i in range(epochs_mean_each_mouse.shape[0]):
            plt.plot(inds, epochs_mean_each_mouse[i, :], '-', color='gray',
                     marker='o', fillstyle='none', markersize=10, linewidth=2, alpha=0.6)
    elif measure == 'Min':
        for i in range(epochs_min_each_mouse.shape[0]):
            plt.plot(inds, epochs_min_each_mouse[i, :], '-', color='gray',
                     marker='o', markersize=10, linewidth=2, alpha=0.6)

    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.ylabel(" Velocity (cm/s)", fontproperties=font_label)
    plt.xlabel("Epoch", fontproperties=font_label)
    plt.ylim(-16, 30)  # set limits
    plt.title(cor + " "+measure+" Velocity"+"\n"+mouse_type +
              ' '+opto_par + "\n"+ stim_loc + ' ' + stim_type, fontproperties=font)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(pre_direct, 'Summary', 'Aver_of_pre_stim_post'+opto_par+'_'+ stim_loc + ' ' + stim_type+'_'+mouse_type +
                             '_' + cor+'_'+measure + '_Velociy_' + str(n_timebin)+'_violin_plot'+'.png'), bbox_inches='tight', orientation='landscape', dpi=400)


def plot_two_protocols_with_mouse_distinction(mouse_type, mouse_list, stim_loc, stim_type1, stim_type2, opto_par):
    '''Extract data over all mice of one group in two stim_types intensity and plot with subplots of individual animals'''

    epochs_spont_all_mice = np.empty(
        (0, pre_interval+interval+post_interval+1))
    epochs_all_mice = np.empty((0, pre_interval+interval+post_interval+1))
#     epochs_mean_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice
    epochs_spont_all_mice_1 = np.empty(
        (0, pre_interval+interval+post_interval+1))
    epochs_all_mice_1 = np.empty((0, pre_interval+interval+post_interval+1))
#     epochs_mean_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice

#     epochs_min_each_mouse = np.empty((0,3)) # array storing the average of each period (OFF-ON-OFF) for all the mice
    plt.figure(2)
    fig = plt.figure(figsize=(20, 15))
    nrows = 3
    ncols = 4
#     fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharey=True)

    count = 0
    if opto_par == "Control":
        loop_list = mouse_list[1]
    else:
        loop_list = mouse_list[0]

    for n in loop_list:  # Run over all the mice
        count += 1
        start = timeit.default_timer()
        global mouse_no
        mouse_no = n
        print(mouse_type+" # ", n)
        # directory to the stim_type for each mouse
        direct = os.path.join(pre_direct, mouse_type,
                              opto_par, 'Mouse_' + str(mouse_no))

        files_list_DLC1 = get_DLC_files(
            os.path.join(direct, stim_loc, stim_type1, 'DLC'))
#         convert_csv_to_xlsx(direct+stim_type+'/Laser') # files might be given in csv, this is to unify
        files_list_Laser1 = get_laser_files(
            os.path.join(direct, stim_loc, stim_type1, 'Laser'))
        files_list_spont1 = get_DLC_files(
            os.path.join(direct, 'Spontaneous', 'DLC'))
        files_list_DLC2 = get_DLC_files(
            os.path.join(direct, stim_loc, stim_typ2e, 'DLC'))
#         convert_csv_to_xlsx(direct+stim_type+'/Laser') # files might be given in csv, this is to unify
        files_list_Laser2 = get_laser_files(
            os.path.join(direct, stim_loc, stim_type2, 'Laser'))
        files_list_spont2 = get_DLC_files(
            os.path.join(direct, 'Spontaneous', 'DLC'))
#         files_list_spont = []
        if len(files_list_DLC1) == 0:
            print("No files for mouse # ", n)
            continue
        elif len(files_list_Laser1) == 0:
            print("No Laser detection for mouse # ", n)
            continue
        else:

            epochs, = extract_epochs_over_trials(files_list_DLC1, files_list_Laser1, direct, stim_loc, stim_type1, scale_pix_to_cm, False, accep_interval_range,
                                                 spont_trial_dict, misdetection_dict, study_param_dict, **intervals_dict, **t_window_dict)
            print('total of {} trials'.format(epochs.shape[0]))
            # construct an array of all the trial epochs of all mice
            epochs_all_mice = np.append(epochs_all_mice, epochs, axis=0)
            # number of epochs extracted for the mouse
            n_epochs = epochs.shape[0]

            # number of spont sessions
            n_spont_sessions = len(files_list_spont1)
            global n_samples
            # number of repeats
            n_samples = int(n_epochs/(n_spont_sessions*n_trials_spont))+1
            # over a spont file to get the same number of epochs as laser session
            epochs_spont, = extract_epochs_over_trials(files_list_spont1, files_list_Laser1, direct,
                                                       'Spontaneous', scale_pix_to_cm, 'y', accep_interval_range,
                                                       spont_trial_dict, misdetection_dict, study_param_dict, **intervals_dict, **t_window_dict)
            # construct an array of all the spont epochs
            epochs_spont_all_mice = np.append(
                epochs_spont_all_mice, epochs_spont, axis=0)

            ax = fig.add_subplot(3, 4, count)
            plot_pre_on_post(ax, pre_direct, mouse_type, opto_par, stim_loc, stim_type, epochs, epochs_spont, treadmil_velocity, ylim, **t_window_dict, **study_param_dict,
                             **intervals_dict, average='Averg_trials', c_laser='deepskyblue', c_spont='k', save_as_format='.pdf')

            epochs, = extract_epochs_over_trials(files_list_DLC2, files_list_Laser2, direct, stim_loc, stim_type2, scale_pix_to_cm, False, accep_interval_range,
                                                 spont_trial_dict, misdetection_dict, study_param_dict, **intervals_dict, **t_window_dict)
            print('total of {} trials'.format(epochs.shape[0]))
            # construct an array of all the trial epochs of all mice
            epochs_all_mice = np.append(epochs_all_mice, epochs, axis=0)
            # number of epochs extracted for the mouse
            n_epochs = epochs.shape[0]

            # number of spont sessions
            n_spont_sessions = len(files_list_spont2)

            # number of repeats
            n_samples = int(n_epochs/(n_spont_sessions*n_trials_spont))+1
            # over a spont file to get the same number of epochs as laser session
            epochs_spont, = extract_epochs_over_trials(files_list_spont2, files_list_Laser2, direct,
                                                       'Spontaneous', scale_pix_to_cm, 'y', accep_interval_range,
                                                       spont_trial_dict, misdetection_dict, study_param_dict, **intervals_dict, **t_window_dict)
            # construct an array of all the spont epochs
            epochs_spont_all_mice = np.append(
                epochs_spont_all_mice, epochs_spont, axis=0)

            plot_pre_on_post(ax, pre_direct, mouse_type, opto_par, stim_loc, stim_type, epochs, epochs_spont, treadmil_velocity, ylim, **t_window_dict, **study_param_dict,
                             **intervals_dict, average='Averg_trials', c_laser='mediumseagreen', c_spont='hotpink', save_as_format='.pdf')

    plt.tight_layout()

    fig.suptitle(cor + "-Velocity ("+mouse_type+") "+opto_par +
                 "\n"+" interval = " + str(n_timebin), fontproperties=font)
    plt.savefig(os.path.join(pre_direct, 'Subplots', 'All_together_'+ stim_loc + '_' + stim_type+"_"+opto_par+'_'+mouse_type+'_' + cor + '_'+body_part+'_timebin=' + str(n_timebin) + "_moving_aver_win="
                             + str(window) + '_spont_sampl='+str(n_samples)+'_pre_post_stim.pdf'), bbox_inches='tight', orientation='landscape', dpi=200)
    plt.show()


def MWW_test(result, result_Ctr, mouse_type):
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
    x = result_Ctr[(result_Ctr['epoch'] == 'ON') & (
        result_Ctr['mouse_type'] == mouse_type)]['mean_velocity'].values
    y = result[(result['epoch'] == ('ON'+mouse_type)) &
               (result['mouse_type'] == mouse_type)]['mean_velocity'].values
    stat = stats.mannwhitneyu(x, y)
    print("MWW ChR2 vs. Ctr "+mouse_type+" = ", stat)
    return statt


def read_npz_return_data_frame(file_path_list, pre_interval, interval, post_interval, pre_stim_inter):
    '''Read the saved .npz file and produce a data frame with the following columns.

    Parameters
    ----------

    file_path_list : list(str)
            List of full paths to .npz files to be read

    pre_interval : int 
            Number of time bins to be taken into account before laser onset 

    interval : int 
            Laser duration in timebins.

    post_interval : int
            Number of time bins to be taken into account after laser onset 

    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info

    Returns
    -------

    result : dataframe
            dataframe with following columns: ['mean_velocity', 'min_velocity', 'mouse_type', 'optogenetic expression', 
                            'pulse_type','intensity_mW','epoch','pre_velocity_pos_neg','pre_x','pre_x_front_back','pre_accel','pre_accel_pos_neg']


    '''

    col_names = ['mean_velocity', 'min_velocity', 'mouse_type', 'optogenetic expression',
                 'pulse_type', 'intensity_mW', 'epoch', 'pre_velocity_pos_neg', 'pre_x', 'pre_x_front_back', 'pre_accel', 'pre_accel_pos_neg']
    result = pd.DataFrame(columns=col_names)
    for file in file_path_list:
        print(file)
        dat = np.load(file)
        properties = file.split("_")
        epochs = dat['epochs_all_mice']
        n_epochs = epochs.shape[0]
        # set the variables of epoch/optogen/pulse type/mouse type/velocity and x of pre stim
        pre = epochs[:, :pre_interval]
        on = epochs[:, pre_interval+1:pre_interval+interval+1]
        mouse_type_ = [properties[0]] * n_epochs*2
        opto_par_ = [properties[1]] * n_epochs*2
        pulse_ = [properties[2]] * n_epochs*2
        inten_ = [properties[3]] * n_epochs*2

        x_ = np.concatenate(
            (dat['avg_pre_stim_position'], dat['avg_pre_stim_position']), axis=0)
        Velocity_ = np.concatenate(
            (dat['avg_pre_stim_velocity'], dat['avg_pre_stim_velocity']), axis=0)
        accel_ = np.concatenate(
            (dat['avg_pre_stim_acc'], dat['avg_pre_stim_acc']), axis=0)
        print(dat['avg_pre_stim_position'].shape, epochs.shape)
        try:
            off_vel = np.average(pre, axis=1)
            on_vel = np.average(on, axis=1)
            all_mean = np.concatenate((off_vel, on_vel), axis=0)
        except ZeroDivisionError:
            all_mean = np.empty((0))

        all_min = np.concatenate(
            (np.min(pre, axis=1), np.min(on, axis=1)), axis=0)
        epoch_off = ['OFF'] * n_epochs
        epoch_on = ['ON'] * n_epochs
        epoch_ = epoch_off+epoch_on
        # append the data of each mouse to a unit dataframe
        df = pd.DataFrame(({'mean_velocity': all_mean, 'min_velocity': all_min,
                            'mouse_type': mouse_type_, 'optogenetic expression': opto_par_, 'pulse_type': pulse_,
                            'intensity_mW': inten_, 'epoch': epoch_, 'pre_velocity_pos_neg': Velocity_, 'pre_x': x_, 'pre_x_front_back': x_,
                            'pre_accel': accel_, 'pre_accel_pos_neg': accel_}))
        frames = [result, df]
        result = pd.concat(frames, ignore_index=True)
    return result


def categorize_pre_x_and_v(result, back_front_boundary, v_threshold, pre_stim_inter):
    '''Set threshold to velocity and x position averaged over pre_stim_inter.

    Parameters
    ----------

    result : dataframe
            dataframe containing information of all animals 

    back_front_boundary : float
            Threshold to set the position of the animal before the laser onset as either 'back' or 'front'

    v_threshold : float
            Threshold to set the velocity of the animal before the laser onset as either 'pos' or 'neg'

    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info

    Returns
    -------

    result : dataframe
            the dataframe as the input but with the difference that pre_X_front_back and 
            pre_velocity_pos_neg set to binary(str) values accorfing to values.

    '''

    ind_0 = result['pre_velocity_pos_neg'] < v_threshold
    ind_1 = result['pre_velocity_pos_neg'] > v_threshold
    ind_2 = result['pre_x'] < back_front_boundary
    ind_3 = result['pre_x'] > back_front_boundary
    ind_4 = result['pre_accel'] < v_threshold
    ind_5 = result['pre_accel'] > v_threshold
    result.loc[ind_0, 'pre_velocity_pos_neg'] = 'neg'
    result.loc[ind_1, 'pre_velocity_pos_neg'] = 'pos'
    result.loc[ind_4, 'pre_accel_pos_neg'] = 'neg'
    result.loc[ind_5, 'pre_accel_pos_neg'] = 'pos'
    result.loc[ind_2, 'pre_x_front_back'] = 'front'
    result.loc[ind_3, 'pre_x_front_back'] = 'back'
    return result


def Plot_ON_OFF_X_V_mean(result, path, mouse_type, stim_loc, stim_type, fps, back_front_boundary, v_threshold, pre_stim_inter, ylim=[-20, 15], save_as_format='.pdf'):
    '''Plot the mean laser-ON laser-OFF vellociry with distinction of velocity and x prior to laser stim.

    Parameters
    ----------

    result : dataframe
            dataframe containing information of all animals 

    path : str
            Path to save the figure
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   
    stim_type : str
            Folder containing the experiment sessions

    back_front_boundary : float
            Threshold to set the position of the animal before the laser onset as either 'back' or 'front'

    v_threshold : float
            Threshold to set the velocity of the animal before the laser onset as either 'pos' or 'neg'

    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info

    ylim : list(float), optional
            Y axis limits for the figure, default set to [-20,15]

    save_as_format : str, optional
            Format to save the figure default is '.pdf'
    '''

    result_pos = result[result['pre_velocity_pos_neg'] == 'pos']
    result_neg = result[result['pre_velocity_pos_neg'] == 'neg']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)

    y = [np.average(result_pos[(result_pos['pre_x_front_back'] == 'front') & (result_pos['epoch'] == 'OFF')]['mean_velocity']),
         np.average(result_pos[(result_pos['pre_x_front_back'] == 'front') & (result_pos['epoch'] == 'ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0], 2))

    plt.plot(['OFF', 'ON'], y, '-',
             color='g', marker='o', markersize=10, linewidth=4, alpha=1, label='V>0 - front-N=' +
             str(len(result_pos[(result_pos['pre_x_front_back'] == 'front') & (result_pos['epoch'] == 'ON')]['mean_velocity'])) +
             ' $\delta$='+delta)

    y = [np.average(result_pos[(result_pos['pre_x_front_back'] == 'back') & (result_pos['epoch'] == 'OFF')]['mean_velocity']),
         np.average(result_pos[(result_pos['pre_x_front_back'] == 'back') & (result_pos['epoch'] == 'ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0], 2))
    plt.plot(['OFF', 'ON'], y, '-',
             color='r', marker='o', markersize=10, linewidth=4, alpha=1, label='V>0 - back-N=' +
             str(len(result_pos[(result_pos['pre_x_front_back'] == 'back') & (result_pos['epoch'] == 'ON')]['mean_velocity'])) +
             ' $\delta$='+delta)

    y = [np.average(result_neg[(result_neg['pre_x_front_back'] == 'front') & (result_neg['epoch'] == 'OFF')]['mean_velocity']),
         np.average(result_neg[(result_neg['pre_x_front_back'] == 'front') & (result_neg['epoch'] == 'ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0], 2))
    plt.plot(['OFF', 'ON'], y, '-',
             color='b', marker='o', markersize=10, linewidth=4, alpha=1, label='V<0 - front-N=' +
             str(len(result_neg[(result_neg['pre_x_front_back'] == 'front') & (result_neg['epoch'] == 'ON')]['mean_velocity'])) +
             ' $\delta$='+delta)

    y = [np.average(result_neg[(result_neg['pre_x_front_back'] == 'back') & (result_neg['epoch'] == 'OFF')]['mean_velocity']),
         np.average(result_neg[(result_neg['pre_x_front_back'] == 'back') & (result_neg['epoch'] == 'ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0], 2))
    plt.plot(['OFF', 'ON'], y, '-',
             color='k', marker='o', markersize=10, linewidth=4, alpha=1, label='V<0 - back-N=' +
             str(len(result_neg[(result_neg['pre_x_front_back'] == 'back') & (result_neg['epoch'] == 'ON')]['mean_velocity'])) +
             ' $\delta$='+delta)
    inten = result['intensity_mW'][0]
    legend = plt.legend(loc='upper right', fontsize=12)
    plt.xlabel("Laser", fontsize=15)  # .set_fontproperties(font_label),
    # .set_fontproperties(font_label)
    plt.ylabel("Average velocity (cm/s)", fontsize=15)
    plt.suptitle(mouse_type + '  ' + stim_loc + ' ' + stim_type + '(I='+inten+')'+'\n'+'pre-stim-inetrval = ' +
                 str(int(pre_stim_inter*1000/fps))+' ms', fontsize=20, y=1)
    # set_ticks(ax)
    plt.ylim(ylim[0], ylim[1])
    ax.set_facecolor((0.8, 1.0, 1.0))
    plt.savefig(os.path.join(path, 'X_V_distinction_mean_' + stim_loc + '_' + stim_type+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps)) +
                             '_inten='+inten+save_as_format), bbox_inches='tight', orientation='landscape', dpi=350)


def Plot_ON_OFF_V_mean(result, path, mouse_type, stim_loc, stim_type, fps, back_front_boundary, v_threshold, pre_stim_inter, ylim=[-20, 15], save_as_format='.pdf'):
    '''Plot the mean laser-ON laser-OFF vellociry with distinction of velocity prior to laser stim.

    Parameters
    ----------

    result : dataframe
            dataframe containing information of all animals 

    path : str
            Path to save the figure
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   
    stim_type : str
            Folder containing the experiment sessions

    back_front_boundary : float
            Threshold to set the position of the animal before the laser onset as either 'back' or 'front'

    v_threshold : float
            Threshold to set the velocity of the animal before the laser onset as either 'pos' or 'neg'

    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info

    ylim : list(float), optional
            Y axis limits for the figure, default set to [-20,15]

    save_as_format : str, optional
            Format to save the figure default is '.pdf'
    '''

    result_pos = result[result['pre_velocity_pos_neg'] == 'pos']
    result_neg = result[result['pre_velocity_pos_neg'] == 'neg']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)

    y = [np.average(result_pos[(result_pos['epoch'] == 'OFF')]['mean_velocity']),
         np.average(result_pos[(result_pos['epoch'] == 'ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0], 2))
    yerr = [np.std(result_pos[(result_pos['epoch'] == 'OFF')]['mean_velocity']),
            np.std(result_pos[(result_pos['epoch'] == 'ON')]['mean_velocity'])]
    plt.errorbar(['OFF', 'ON'], y, yerr, marker='o', markersize=10, linewidth=2, capsize=10, capthick=3,
                 color='r', label='V>0 -N=' +
                 str(len(result_pos[(result_pos['epoch'] == 'ON')]['mean_velocity'])) +
                 ' $\delta$='+delta)

    y = [np.average(result_neg[(result_neg['epoch'] == 'OFF')]['mean_velocity']),
         np.average(result_neg[(result_neg['epoch'] == 'ON')]['mean_velocity'])]
    yerr = [np.std(result_neg[(result_neg['epoch'] == 'OFF')]['mean_velocity']),
            np.std(result_neg[(result_neg['epoch'] == 'ON')]['mean_velocity'])]
    delta = str(round(y[1]-y[0], 2))
    plt.errorbar(['OFF', 'ON'], y, yerr, marker='o', markersize=10, linewidth=2, capsize=10, capthick=3,
                 color='k', label='V<0 -N=' +
                 str(len(result_neg[(result_neg['epoch'] == 'ON')]['mean_velocity'])) +
                 ' $\delta$='+delta)

    inten = result['intensity_mW'][0]
    legend = plt.legend(loc='upper right', fontsize=12)
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel("Laser", fontsize=15)  # .set_fontproperties(font_label),
    # .set_fontproperties(font_label)
    plt.ylabel("Average velocity (cm/s)", fontsize=15)
    plt.suptitle(mouse_type+'  '+stim_type+'(I='+inten+')'+'\n'+'pre-stim-inetrval = ' +
                 str(int(pre_stim_inter*1000/fps))+' ms', fontsize=20, y=1)
    # set_ticks(ax)
    ax.set_facecolor((1, 1.0, .8))
    plt.savefig(os.path.join(path, 'X_V_distinction_mean_' + stim_loc + '_' + stim_type+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps)) +
                             '_inten='+inten+save_as_format), bbox_inches='tight', orientation='landscape', dpi=350)


def violin_plot_X_V_distiction(result, path, mouse_type, stim_loc, stim_type, fps, back_front_boundary, v_threshold, pre_stim_inter, ylim=[-30, 30], save_as_format='.pdf'):
    """Plot violin plots with columns for pre_laser distinction and hue for back and front distinction on the treadmill.

    Parameters
    ----------

    result : dataframe
            dataframe containing information of all animals 
    path : str
            Path to save the figure
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   

    stim_type : str
            Folder containing the experiment sessions
    back_front_boundary : float
            Threshold to set the position of the animal before the laser onset as either 'back' or 'front'
    v_threshold : float
            Threshold to set the velocity of the animal before the laser onset as either 'pos' or 'neg'
    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info
    ylim : list(float), optional
            Y axis limits for the figure, default set to [-30,30]
    save_as_format : str, optional
            Format to save the figure default is '.pdf'

    Returns
    -------
    """

    g = sns.catplot(x="epoch", y="mean_velocity", hue="pre_x_front_back", col="pre_velocity_pos_neg",
                    data=result, kind="violin", split=True, palette=sns.color_palette("Set2", n_colors=2, desat=.5),
                    scale_hue=False, linewidth=2, inner="quartile", scale='area',
                    hue_order=['front', 'back'], col_order=['pos', 'neg'], legend=False)

    ax1, ax2 = g.axes[0]

    sns.set(font_scale=2)
    sns.set_style("white")
    plt.ylim(ylim[0], ylim[1])

    ax1.axhline(y=0, ls='-', c='y', linewidth=3)
    ax2.axhline(y=0, ls='-', c='y', linewidth=3)

    ax1.axhline(y=-15, ls='--', c='r', linewidth=3)
    ax2.axhline(y=-15, ls='--', c='r', linewidth=3)

    ax1.set_title('V > 0', y=0.95, fontsize=25)
    ax2.set_title('V < 0', y=0.95, fontsize=25)

    ax1.set_xlabel('Laser', fontsize=25)
    ax2.set_xlabel('Laser', fontsize=25)

    ax1.set_ylabel('Average velocity (cm/s)', fontsize=25)

    ax1.get_xaxis().set_tick_params(direction='out', labelsize=20, length=10)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.get_yaxis().set_tick_params(direction='out', labelsize=20, length=10)
    ax1.yaxis.set_ticks_position('left')

    ax2.get_xaxis().set_tick_params(direction='out', labelsize=20, length=10)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.get_yaxis().set_tick_params(direction='out', labelsize=20, length=10)
    ax2.yaxis.set_ticks_position('left')

    inten = result['intensity_mW'][0]
    #g.set_axis_labels("Laser", "Average velocity (cm/s)")
    plt.suptitle(mouse_type+'  '+stim_type+' ('+inten+'mW)'+' pre-stim-interval = ' +
                 str(int(pre_stim_inter*1000/fps))+' ms', fontsize=30, y=1)
    g.fig.set_figwidth(20.0)
    g.fig.set_figheight(12)
    legend = plt.legend(loc='upper right',
                        title='position on treadmill', fontsize=20)

    plt.savefig(os.path.join(path, mouse_type+'_X_V_violin_plot_'+stim_type+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps))
                             + '_inten='+inten+save_as_format), bbox_inches='tight', orientation='landscape', dpi=350)


def plot_phase_space_V(result, path, mouse_type, stim_loc, stim_type, fps, back_front_boundary, v_threshold, pre_stim_inter, xlim=[-25, 32], ylim=[-30, 40], save_as_format='.pdf'):
    '''Plot the phase space of laser-ON vs. laser-OFF velocity for all trials of all mice.

    Parameters
    ----------

    result : dataframe
            dataframe containing information of all animals 

    path : str
            Path to save the figure

    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   

    stim_type : str
            Folder containing the experiment sessions

    back_front_boundary : float
            Threshold to set the position of the animal before the laser onset as either 'back' or 'front'

    v_threshold : float
            Threshold to set the velocity of the animal before the laser onset as either 'pos' or 'neg'

    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info

    xlim : list(float), optional
            X axis limits for the figure, default set to [-25,32]

    ylim : list(float), optional
            Y axis limits for the figure, default set to [-20,15]

    save_as_format : str, optional
            Format to save the figure default is '.pdf'
    '''

    result_pos = result[result['pre_velocity_pos_neg'] == 'pos']
    result_neg = result[result['pre_velocity_pos_neg'] == 'neg']
#     result_zero = result[(result['velocity'] != 'neg') & (result['velocity'] != 'pos')]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
    OFF_vel = result_pos[(result_pos['epoch'] == 'OFF')]['mean_velocity']
    plt.scatter(result_pos[(result_pos['epoch'] == 'OFF')]['mean_velocity'],
                result_pos[(result_pos['epoch'] == 'ON')]['mean_velocity'], c='navy', label=r'$V > 0$')
    plt.scatter(result_neg[(result_neg['epoch'] == 'OFF')]['mean_velocity'],
                result_neg[(result_neg['epoch'] == 'ON')]['mean_velocity'], c='purple', label=r'$V < 0$')

    inten = result['intensity_mW'][0]
    plt.plot([-40, 40], [-40, 40], '--', c='k', label=r'$ V_{ON}=V_{OFF}$')
    legend = plt.legend(loc='upper right', fontsize=20)
    plt.xlabel("Velocity OFF (cm/s)").set_fontproperties(font_label),
    plt.ylabel("Velocity ON (cm/s)").set_fontproperties(font_label)
    plt.suptitle(mouse_type+'  '+ stim_loc + ' ' + stim_type+'(I='+inten+')'+'\n'+'pre-stim-interval = ' +
                 str(int(pre_stim_inter*1000/fps))+' ms', fontsize=30, y=1)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    set_ticks(ax)
#     ax.set_facecolor((0.8, 1.0, 1.0))
    plt.savefig(os.path.join(path, mouse_type+'_V_phase_space_'+ stim_loc + '_' + stim_type+'_pre_stim_t='+str(int(pre_stim_inter*1000/fps)) +
                             '_inten='+inten+save_as_format), bbox_inches='tight', orientation='landscape', dpi=350)


def min_velocity_diff_inten_box_plot(file_path_list, path_to_save, intervals_dict, pre_x_v_dict, ylim=[-40, 15], save_as_format='.pdf'):
    """Compare min velocity during laser for trials with positive and negative pre laser velocity. 

    Box plots to show this comparison for different intensity in same animal types.

    Parameters
    ----------

    file_path_list : list(str)
            list of paths to .npz files 
    path_to_save : str
            Path to save figure 
    intervals_dict : dict-like
            dict-like of interals:{'pre_interval' : int(.5*fps), # interval before laser onset
                                                            'interval' : int(interval_in_sec * fps), # number of timebins of stimulation
                                                            'post_interval' : int(.5*fps*2), # interval after laser onset
                                                            'pre_stim_inter' : pre_stim_inter }
    pre_x_v_dict : dict-like
            {'back_front_boundary' : (treadmill_len-elec_shock)/2, # set the limit below which is considered back of the treadmill
                            'v_threshold' : 0,
                            'pre_stim_inter' : pre_stim_inter # number of timebins in the pre-stimulus period}    
    ylim : list(float), optional
            y axis limits for figure. Default is set to [-40,15]
    save_as_format : str, optional
            format to save the figure. Default is set to '.pdf'

    Returns
    -------

    """

    n_subplots = len(file_path_list)

    fig, axes = plt.subplots(nrows=1, ncols=n_subplots,
                             figsize=(4*n_subplots, 8))
    for count in range(1, n_subplots+1):
        path = file_path_list[count-1]
        print(Path(path).name)
        result_val = read_npz_return_data_frame([path], **intervals_dict)
        result = categorize_pre_x_and_v(result_val, **pre_x_v_dict)
        result = result[result['epoch'] == 'ON']
        mouse_type = Path(path).name.split("_")[0]
        inten = Path(path).name.split("_")[2:4]

        s = ' '
        inten = s.join(inten)
        ax = axes[count-1]
        ax = plt.subplot(100+n_subplots*10+count)
        set_ticks(ax)

        sns.stripplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], dodge=True, data=result, jitter=True,
                      marker='o', edgecolor='k', linewidth=1, size=3,
                      alpha=0.5)
        g = sns.boxplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], dodge=False, width=0.4,
                        data=result, fliersize=0)

        add_stat_annotation(g, data=result, x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], box_pairs=[("pos", "neg")],
                            test='Mann-Whitney', text_format='star', loc='outside', verbose=2, fontsize=20)
        plt.plot([0.3, 0.7], [np.average(result[result['pre_velocity_pos_neg'] == 'pos']['min_velocity']),
                              np.average(result[result['pre_velocity_pos_neg'] == 'neg']['min_velocity'])],
                 '-', lw=3, c='r', alpha=0.5, markersize=12)
        for patch in g.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.ylim(ylim[0], ylim[1])
        # get legend information from the plot object
        # handles, labels = ax.get_legend_handles_labels()
        # specify just one legend
        # plt.legend(handles[0:2], labels[0:2], fontsize = 20)
        plt.ylabel('').set_fontproperties(font_label)
        plt.xlabel(r'$min(V_{laser-On})$').set_fontproperties(font_label)
        plt.title(inten, fontsize=18, pad=75)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.suptitle(mouse_type, y=1.15, fontproperties=font)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
    plt.ylabel('Velocity (cm/s)', fontsize=25,
               labelpad=45).set_fontproperties(font_label)
    plt.savefig(os.path.join(path.replace(Path(path).name, ''), 'Min_V_diff_inten_' +
                mouse_type+save_as_format), bbox_inches='tight', orientation='landscape', dpi=200)


def min_velocity_mouse_type_box_plot(file_path_list, path_to_save, intervals_dict, pre_x_v_dict, ylim=[-40, 15], save_as_format='.pdf'):
    """Compare min velocity during laser.

    Box plots to show this comparison FOR trials with positive and negative
    pre laser velocity for same intensity in different animal types.

    Parameters
    ----------

    file_path_list : list(str)
            list of paths to .npz files 
    path_to_save : str
            Path to save figure 
    intervals_dict : dict-like
            dict-like of interals:{'pre_interval' : int(.5*fps), # interval before laser onset
                                                            'interval' : int(interval_in_sec * fps), # number of timebins of stimulation
                                                            'post_interval' : int(.5*fps*2), # interval after laser onset
                                                            'pre_stim_inter' : pre_stim_inter }
    pre_x_v_dict : dict-like
            {'back_front_boundary' : (treadmill_len-elec_shock)/2, # set the limit below which is considered back of the treadmill
                            'v_threshold' : 0,
                            'pre_stim_inter' : pre_stim_inter # number of timebins in the pre-stimulus period}
    ylim : list(float), optional
            y axis limits for figure. Default is set to [-40,15]
    save_as_format : str, optional
            format to save the figure. Default is set to '.pdf'

    Returns
    -------

    """

    n_subplots = len(file_path_list)

    fig, axes = plt.subplots(nrows=1, ncols=n_subplots,
                             figsize=(4*n_subplots, 8))
    for count in range(1, n_subplots+1):
        path = file_path_list[count-1]
        print(path)

        result_val = read_npz_return_data_frame([path], **intervals_dict)
        result = categorize_pre_x_and_v(result_val, **pre_x_v_dict)
        result = result[result['epoch'] == 'ON']
        mouse_type = Path(path).name.split("_")[0]
        ax = axes[count-1]
        ax = plt.subplot(100+n_subplots*10+count)
        set_ticks(ax)

        sns.stripplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], dodge=True, data=result, jitter=True,
                      marker='o', edgecolor='k', linewidth=1, size=3,
                      alpha=0.5)
        g = sns.boxplot(x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], dodge=False, width=0.4,
                        data=result, fliersize=0)

        add_stat_annotation(g, data=result, x="pre_velocity_pos_neg", y="min_velocity", order=["pos", "neg"], box_pairs=[("pos", "neg")],
                            test='Mann-Whitney', text_format='star', loc='outside', verbose=2, fontsize=20)
        plt.plot([0.3, 0.7], [np.average(result[result['pre_velocity_pos_neg'] == 'pos']['min_velocity']),
                              np.average(result[result['pre_velocity_pos_neg'] == 'neg']['min_velocity'])],
                 '-', lw=3, c='r', alpha=0.5, markersize=12)
        for patch in g.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.ylim(ylim[0], ylim[1])
        # get legend information from the plot object
        # handles, labels = ax.get_legend_handles_labels()
        # specify just one legend
        # plt.legend(handles[0:2], labels[0:2], fontsize = 20)
        plt.ylabel('').set_fontproperties(font_label)
        plt.xlabel(r'$min(V_{laser-On})$').set_fontproperties(font_label)
        plt.title(mouse_type, fontsize=18, pad=75)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.suptitle(stim_loc + ' ' + stim_type, y=1.15, fontproperties=font)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
    plt.ylabel('Velocity (cm/s)', fontsize=25,
               labelpad=45).set_fontproperties(font_label)
    plt.savefig(os.path.join(path, 'Min_V_diff_mouse_type_'+ stim_loc + '_' + stim_type +
                save_as_format), bbox_inches='tight', orientation='landscape', dpi=200)


def extract_epochs(bins, x, smallest_accep_inter, largest_accep_inter, pre_interval, interval, post_interval, session_length):
    '''Extract the (pre | Laser ON | post) epochs.

    Check reported (start,end) of trials from bins and discard trial with unacceptable duration. 
    Return the stacked the corresponding frames of all trials from the measurment array x.

    Parameters
    ----------

    bins : 2D-array(int)
            array of start and end times for trials

    smallest_accep_inter : int
            Smallest acceptable duration of an interval in frames

    largest_accep_inter : int
            Largest acceptable duration of an interval in frames

    pre_interval : int 
            Number of time bins to be taken into account before laser onset 

    interval : int 
            Laser duration in timebins.

    post_interval : int
            Number of time bins to be taken into account after laser onset 

    Returns
    -------

    epochs : 2D array(float)
            Trials containtning laser stimulation aligned in rows. n_col = pre_interval + interval + post_interval

    n_trials : int
            Number of trials

    take : 1D array(int)
            Array of indices corresponding to pre | on | post epoch frmaes.
    '''
    bins_in = np.copy(bins)
    # remove the unacceptable epochs
    duration = bins_in[:, 1] - bins_in[:, 0]
    acceptable = np.logical_and(
                                duration > smallest_accep_inter, 
                                duration < largest_accep_inter
                                )
    print(len(acceptable) - sum(acceptable),
         ' trials discarded')
    
    bins_in = bins[acceptable, :]


    # find the epochs != interval
    # find the exterior intervals to the standard interval
    larger_intervals = (duration[acceptable]) > interval
    # find the inferior intervals to the min interval
    smaller_intervals = (duration[acceptable]) < interval

    # remove or add the extra frames  to make it uniform along the different stimuli
    bins_in[larger_intervals, 1] = bins_in[larger_intervals, 1]  \
                                   - (
                                       bins_in[larger_intervals, 1]
                                      - bins_in[larger_intervals, 0]
                                      - interval
                                      )
    bins_in[smaller_intervals, 1] = bins_in[smaller_intervals, 1] \
                                    + (
                                        interval 
                                        - (bins_in[smaller_intervals, 1] 
                                          - bins_in[smaller_intervals, 0]
                                          )
                                       )
    bins_in = remove_epochs_with_unaccep_pre_post_interval(bins_in, pre_interval, post_interval, session_length)
    bins_in[:, 1] = bins_in[:, 0] \
                    + interval \
                    + post_interval
    # extend the interval to pre and post
    bins_in[:, 0] = bins_in[:, 0] - pre_interval
    
    n_trials = len(bins_in) - 1
    

    # make an array with indices of laser ON timebins
    take = np.hstack([np.arange(i[0], i[1] + 1) for i in bins_in[:-1]])
    epochs = x[take].reshape(n_trials, 
                             pre_interval + post_interval + interval + 1)
    
    return epochs, n_trials, take

        
def choose_n_from_colormap(cmap, minval=0.0, maxval=1.0, n_colors = 100):
    
    """Return 'n' equally distanced colors from the colormap"""
    
    return  [cmap(i) for i in np.linspace(minval, maxval, num = n_colors, endpoint=True)]        

def remove_epochs_with_unaccep_pre_post_interval(bins, pre_interval, post_interval, session_length):
    
    ''' Remove trials where extending the width to pre interval and post
    interval exceeds the session boundaries
    '''
    
    ind_unacc = np.logical_or(bins[:, 1] + post_interval > session_length, bins[:, 1] - pre_interval < 0)
    
    return bins[~ind_unacc, :]

def extract_epochs_over_trials(files_list_DLC, files_list_laser, direct, stim_loc, stim_type, scale_pix_to_cm, spont, accep_interval_range,
                               spont_trial_dict, misdetection_dict, study_param_dict, pre_interval, interval, post_interval, pre_stim_inter,
                               fps, n_timebin, window_veloc, window_pos, n_samples=25):
    """Return all the epochs of all similar trials for one mouse.

    Parameters
    ----------
    files_list_DLC : list(str)
            List of paths to all DLC .csv files for sessions    
    files_list_laser : list(str)
            List of paths to all .csv files that have (start,end) laser times.
    direct : str
            Path to the parent directory containing all stim_types
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   

    stim_type : str
            folder containing sessions for this particular protocol
    scale_pix_to_cm : float
            Scaling coefficient converting pixels to cm
    spont : str
            Either False --> for trials containing laser stimulation
            or 'y' --> for spontaneous trials with no laser
    accep_interval_range: tuple(int)
            Smallest and largest acceptable trial sizes in timebins
    spont_trial_dict : dict-like
            {'max_distance' : int (max n_timebins between sampled epochs for spontaneous),
            'min_distance' : int (min #n_timebines between sampled epochs for spontaneous),
            'n_trials_spont' : int (the average number of trials extracted from one spontaneous session)}
    misdetection_dict : dict-like 
            Dictionary of constants used in the misdetection correction algorithm:
            {'acc_deviance' : float  (acceptable deviance between right and left detections),
            'internal_ctr_dev' : float,
            'percent_thresh_align' : int,
            't_s' : int (number of time steps before and after to look at),
            'n_iter_jitter' : int (how many times go over data to find jitters and clear them out),
            'jitter_threshold' : int (max_speed/fps)}
    study_param_dict : dict-like
            Dictionary of values regarding coordinate (x,y), body part (list of body parts) and plot parameter 
            (position, velocity, acceleration) packed from input of user
    pre_interval : int 
            Number of time bins to be taken into account before laser onset 
    interval : int 
            Laser duration in timebins.
    post_interval : int
            Number of time bins to be taken into account after laser onset 
    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info
    fps : int
            Frame per second of the video
    n_timebin : int
            Number of frames as timebins for the x derivative to yields velocity
    window_veloc : int
            Velocity moving average window in frames
    window_pos : int
            Position moving average window in frames
    n_samples : int, optional
            Number of samples to extract from one spontaneous session. Only necessary for spont = 'y' cases.

    Returns
    -------
    epochs : 2D array(float)
            (pre | Laser ON | post) epochs with shape (n_trials, pre_interval+post_interval+interval+1)
    pre_info : 2D array(float)
            Three columns with averages of : pre_x , pre_v, pre_acceleration over pre_stim_interval duration.
            with shape (n_trials, 3)

    """
    accep_interval_range = correct_accep_interval_range_for_beta(accep_interval_range, stim_type)
    
    if pre_interval > pre_stim_inter:

        pre_era = pre_interval
    
    else:
        
        pre_era = pre_stim_inter

    plot_param = study_param_dict['plot_param']
    i = 0

    epochs = np.empty((0, pre_interval+interval+post_interval+1))
    epochs_pos = np.empty((0, pre_stim_inter+interval+post_interval+1))
    epochs_veloc = np.empty((0, pre_stim_inter+interval+post_interval+1))
    epochs_acc = np.empty((0, pre_stim_inter+interval+post_interval+1))
    pre_info = np.empty((0, 3))
    
    # print(files_list_DLC, files_list_laser)
    for i in range(0, len(files_list_DLC)):

        print('\n session {} out of {}'.format(i+1, len(files_list_DLC)))
        
        file_path_DLC = os.path.join(direct, stim_loc, stim_type, 'DLC', files_list_DLC[i])
        df = read_DLC(file_path_DLC, scale_pix_to_cm)
        # print(file_path_DLC)
        position = average_position_r_l(
            df, window_pos, misdetection_dict, **study_param_dict)
        session_length = len(position)
        velocity = derivative_mov_ave(position, n_timebin, window_veloc, fps)
        acceleration = derivative(velocity, n_timebin, fps)

        # if only onse side is needed
        # variable,left_side = position_r_l(df, which_plot,where_plot)
        if plot_param == 'position':
            variable = position

        elif plot_param == 'velocity':
            variable = velocity   # velocity

        elif plot_param == 'acceleration':
            variable = acceleration   # velocity

        if spont == 'y':  # if it's a spontaneous reading extract epochs randomly
            bins = produce_random_bins_for_spont(len(variable), n_samples, pre_interval, interval, post_interval,
                                                 **spont_trial_dict)

        else:  # if a normal trial read bins from laser times
            file_path_Laser = os.path.join(
                direct, stim_loc, stim_type, 'Laser', files_list_laser[i])
            laser_t = read_laser(file_path_Laser, file_path_DLC)
            bins = np.copy(laser_t.values).astype(int)
        epochs_trial, n_trials, take = extract_epochs(
            bins, variable, *accep_interval_range, pre_era, interval, post_interval, session_length)
        epochs_pos = position[take].reshape(
            n_trials, pre_era+post_interval+interval+1)
        epochs_veloc = velocity[take].reshape(
            n_trials, pre_era+post_interval+interval+1)
        epochs_acc = acceleration[take].reshape(
            n_trials, pre_era+post_interval+interval+1)
        if pre_era > pre_interval:

            pre_info = np.append(pre_info,
                                 np.concatenate((np.average(epochs_pos[:, :pre_stim_inter], axis=1).reshape(-1, 1),
                                                 np.average(
                                                     epochs_veloc[:, :pre_stim_inter], axis=1).reshape(-1, 1),
                                                 np.average(epochs_acc[:, :pre_stim_inter], axis=1).reshape(-1, 1)),  axis=1),    axis=0)
            epochs = np.append(
                epochs, epochs_trial[:, pre_stim_inter-pre_interval:], axis=0)
        else:

            pre_info = np.append(pre_info,
                                 np.concatenate((np.average(epochs_pos[:, pre_interval - pre_stim_inter:pre_era], axis=1).reshape(-1, 1),
                                                 np.average(
                                                     epochs_veloc[:, pre_interval - pre_stim_inter:pre_era], axis=1).reshape(-1, 1),
                                                 np.average(epochs_acc[:, pre_interval - pre_stim_inter:pre_era], axis=1).reshape(-1, 1)),    axis=1),    axis=0)
            epochs = np.append(epochs, epochs_trial, axis=0)

    if plot_param == 'position':
        return epochs - np.repeat(epochs[:, pre_interval].reshape(epochs.shape[0], 1), epochs.shape[1], axis=1)
    else:
        return epochs, pre_info

def correct_accep_interval_range_for_beta(accep_interval_range, stim_type):
    """ beta stimulation doesn't include the last half cycle so 
    it's shorter. Therefore the acceptable range is coorected to be
    smaller 
    """
    if 'beta' in stim_type and 'square' not in stim_type:
        
        return tuple([i * (118/125) for i in accep_interval_range])
    
    else:
        
        return accep_interval_range
    
def run_one_intensity_save_data(pre_direct, scale_pix_to_cm, mouse_type, mouse_dict, stim_loc, stim_type, opto_par, treadmil_velocity,
                                ylim, spont_trial_dict, misdetection_dict, intervals_dict, t_window_dict, accep_interval_range, study_param_dict,
                                max_distance, min_distance, n_trials_spont, c_laser = 'deepskyblue', c_spont = 'k', fig = None,
                                outer = None, n_inner = 0, inner = None, axes = [], remove_empty_ax = True, label = None, annotate_n = False,
                                title_manually = True, plot_spont = True, save_fig = True, suptitle_y = 0.95):
    
    """Save data of epochs and individal mice to a npz file.

    by running over all mice of one group and one intensity. 

    Parameters
    ----------
    pre_direct : str
            path to rooot directory
    scale_pix_to_cm : float
            SCaling coefficient for converting pixels to cm.
    mouse_type : str
            mouse type such as `FoxP2` etc
    mouse_list : list(int)
            list of mouse identification numbers 
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".      
    stim_type : str
            the folder with the corresponding experiment protocol e.g. "Square_1_mW".
    opto_par: str
            `ChR2` for ChR2 injected animals `Control` for Control group
    treadmil_velocity : float
            Treadmill velocity in `cm/s`
    ylim : list(float)
    spont_trial_dict : dict-like
            dict-like with values : {'max_distance' : int(4.5*fps), # max #n_timebines between sampled epochs for spontaneous
                                                              'min_distance' : int(3*fps), # min #n_timebines between sampled epochs for spontaneous
                                                              'n_trials_spont' : 25 # the average number of trials extracted from one spontaneous session}
    misdetection_dict : dict-like 
            dictionary of constants used in the misdetection correction algorithm:
    intervals_dict : dict-like
            dictionary of intervals
    t_window_dict: dict-like
            dict-like containing information about time contants of the project:
            t_window_dict = {`fps` : int, 'n_timebin' : int, 'window_pos' : int , 'window_veloc' : int}
    accep_interval_range: tuple(int)
            Smallest and largest acceptable trial sizes in timebins
    study_param_dict : dictionray
            dict-like of values regarding coordinate (x,y), body part (list of body parts) and plot parameter 
            (position, velocity, acceleration) packed from input of user
    max_distance : int
            maximum number of frames between sampled epochs for spontaneous
    min_distance : int
            mainimum number of frames between sampled epochs for spontaneous
    n_trials_spont : int
            number of spontaneous trials

    Returns
    -------

    """
    epochs_spont_all_mice = np.empty((0, intervals_dict['pre_interval']
                                         + intervals_dict['interval'] 
                                         + intervals_dict['post_interval']
                                         + 1))
    epochs_all_mice = np.empty_like(epochs_spont_all_mice)
    
    # array storing the average of each period (OFF-ON-OFF) for all the mice
    epochs_mean_each_mouse = np.empty((0, 3))
    all_pre_info = np.empty((0, 3))
    epochs_mean_pre_v = np.empty((0, 1))

    
    n_subplots = len(mouse_dict[opto_par])
    # fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5), sharey = True)

    fig = fig or plt.figure(figsize=(5 * n_subplots, 5))
    outer = outer or gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
    inner = inner or gridspec.GridSpecFromSubplotSpec(1, n_subplots,
                    subplot_spec=outer[0], wspace=0.1, hspace=0.1)#,sharex = True)

    if len(axes) == 0:
        
        axes = np.empty(n_subplots, object)
        
        for i in range(n_subplots):
            
            axes[i] = plt.Subplot(fig, inner[i])   
            fig.add_subplot(axes[i])
            
        axvspan = True
        
    else:
        
        axvspan = False

    
    global mouse_no
    global n_samples
    exist_count = 0
    
    for count, mouse_no  in enumerate(mouse_dict[opto_par]):  # Run over all the mice

        start = timeit.default_timer()

        print("Type : ", mouse_type + " # ", mouse_no)
        # directory to the stim_type for each mouse
        direct = os.path.join(pre_direct, mouse_type,
                              opto_par, 'Mouse_' + str(mouse_no))

        files_list_DLC = get_DLC_files(
            os.path.join(direct, stim_loc, stim_type, 'DLC'))

        files_list_Laser = get_laser_files(
            os.path.join(direct, stim_loc, stim_type, 'Laser'))

        if os.path.exists(os.path.join(direct, 'Spontaneous', 'DLC')) :

            files_list_spont = get_DLC_files(
                os.path.join(direct, 'Spontaneous', 'DLC'))
            
        else:

            files_list_spont = []
        
        
        if no_files(files_list_DLC, mouse_no, dtype = 'DLC') or \
           no_files(files_list_Laser, mouse_no, dtype = 'Laser') :
            
            if remove_empty_ax:
                axes[count].axis('Off')
            continue
        
        else:
            
            exist_count += 1
            epochs, pre_info = extract_epochs_over_trials(files_list_DLC,
                                                          files_list_Laser,
                                                          direct, stim_loc, 
                                                          stim_type,
                                                          scale_pix_to_cm,
                                                          False,
                                                          accep_interval_range,
                                                          spont_trial_dict,
                                                          misdetection_dict,
                                                          study_param_dict,
                                                          **intervals_dict,
                                                          **t_window_dict)
            n_epochs = epochs.shape[0]
            print(n_epochs, 'laser trials')
            
            # fig.add_subplot(axes[count])

            # construct an array of all the trial epochs of all mice
            epochs_all_mice = np.append(epochs_all_mice, epochs, axis=0)
            all_pre_info = np.append(all_pre_info, pre_info, axis=0)


            if len(files_list_spont) == 0:  # if no spont trials recorded set it to zero

                epochs_spont = np.zeros(np.shape(epochs))
                # construct an array of all the spont epochs
                epochs_spont_all_mice = np.append(
                    epochs_spont_all_mice, epochs_spont, axis=0)
                n_samples = 0

            else:

                
                n_spont_sessions = len(files_list_spont) # number of spont sessions
                
                n_samples = int(n_epochs 
                                / (n_spont_sessions 
                                   * n_trials_spont)) + 1 # number of repeats over a spont file to get the same number of epochs as laser session

                epochs_spont, _ = extract_epochs_over_trials(files_list_spont,
                                                                files_list_Laser,
                                                                direct,
                                                                'Spontaneous',
                                                                scale_pix_to_cm,
                                                                'y',
                                                                accep_interval_range,
                                                                spont_trial_dict,
                                                                misdetection_dict,
                                                                study_param_dict,
                                                                **intervals_dict,
                                                                **t_window_dict,
                                                                n_samples=n_samples)

                print('Spontaneous session available. {} trials extracted'.format(
                    epochs_spont.shape[0]))

            # construct an array of all the spont epochs
            epochs_spont_all_mice = np.append(
                epochs_spont_all_mice, epochs_spont, axis=0)
            # get the mean value of velocity for pre-on-post intervals
            temp = min_and_mean_on_off(epochs, 'Mean', **intervals_dict)
            average_of_on_off_on = np.average(
                temp, axis=0).reshape(1, 3)  # average over trialas
            # construct an array with these 3values for all the mice
            epochs_mean_each_mouse = np.append(
                epochs_mean_each_mouse, average_of_on_off_on, axis=0)


            ax = plot_pre_on_post(axes[count],
                             pre_direct,
                             mouse_type,
                             opto_par,
                             stim_loc, 
                             stim_type,
                             epochs,
                             epochs_spont,
                             treadmil_velocity,
                             ylim,
                             **t_window_dict,
                             **study_param_dict,
                             **intervals_dict,
                             average='Averg_trials',
                             c_laser= c_laser,
                             c_spont = c_spont,
                             save_as_format='.pdf',
                             title = False,
                             label = label, 
                             annotate_n = annotate_n,
                             plot_spont=plot_spont,
                             axvspan = axvspan)
            
            axes[count].set_title('# {}'.format(mouse_no)).set_fontproperties(font)


            if count == 0:
                axes[count].set_title(stim_loc + '\n # {}'.format(mouse_no)).set_fontproperties(font)
                
            if not title_manually:
                rm_ax_unnecessary_labels_in_subplots(count, n_subplots, axes[count])

            stop = timeit.default_timer()
            print('runtime = ', int(stop-start), " s")
            
        

    
    if exist_count > 1:


        Directory.create_dir_if_not_exist(os.path.join(pre_direct, 'Subplots', stim_loc, stim_type ))
        if title_manually:
            fig.suptitle("(" + mouse_type 
                         + " " + opto_par + ") " 
                         +  stim_loc 
                         + " " + stim_type.replace('_', ' ').replace('-','.')).set_fontproperties(font)
        else:
            fig.suptitle("(" + opto_par
                         + " " + mouse_type + ") " 
                         + " " + stim_type.split('_')[0], y= suptitle_y ).set_fontproperties(font)
        
     
        if save_fig:
            save_pdf_png(fig, os.path.join(pre_direct, 'Subplots', stim_loc, stim_type,
                                     'All_together_' 
                                     + opto_par 
                                     + '_' + mouse_type 
                                     + '_' + study_param_dict['cor']
                                     + '_' + study_param_dict['body_part'][0] 
                                     + '_timebin=' + str(t_window_dict['n_timebin']) 
                                     + "_moving_aver_win=" + str(t_window_dict['window_pos']) 
                                     + '_spont_sampl=' + str(n_samples) 
                                     + '_pre_post_stim'))
        save_npz(pre_direct, 
                 mouse_type, 
                 opto_par, 
                 stim_loc, 
                 stim_type, 
                 stim_type.split('_')[-1],
                 t_window_dict['fps'], 
                 t_window_dict['window_pos'], 
                 t_window_dict['n_timebin'], "",
                 epochs_all_mice, 
                 epochs_mean_each_mouse, 
                 epochs_spont_all_mice, 
                 all_pre_info, 
                 **study_param_dict)
        

    

    
    return axes

def save_pdf_png(fig, figname, size = (8,6)):
    
    '''Save fig as pdf and png format'''
    # fig.set_size_inches(size, forward=False)
    fig.savefig(figname + '.png', dpi = 500, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    fig.savefig(figname + '.pdf', dpi = 500, #facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True ,bbox_inches = "tight", pad_inches=0.1)
    
def create_pulse_specific_color_lists(pulse_cmap = {'betapulse' : 'copper', 'squarepulse': 'winter'}, 
                                      numbers = {'betapulse' : 4, 'squarepulse': 4}):
    
    cmap_dict = {pulse: choose_n_from_colormap(plt.get_cmap(cmap), minval=0.0, maxval=1.0, n_colors = numbers[pulse])
                          for pulse, cmap in pulse_cmap.items()}
    return cmap_dict

def get_key_that_contains(dictionary, substr):
    
    return [key for key, value in dictionary.items() if substr in key.lower()]

def superimpose_intensities(cmap_dict, opto_par_list, stim_type_dict, stim_loc_list, pre_direct, scale_pix_to_cm, 
                            mouse_type, mouse_dict, treadmil_velocity,
                            ylim, spont_trial_dict, misdetection_dict, intervals_dict, t_window_dict, 
                            accep_interval_range, study_param_dict,
                            max_distance, min_distance, n_trials_spont, plot_spont = True, 
                            ylabel_x = 0, xlabel_y = 0.07, suptitle_y = 0.98):

    
    for opto_par in opto_par_list:
        
        for pulse, stim_type_list in stim_type_dict.items():
            
            n_mice = len(mouse_dict[opto_par])
            fig = plt.figure(figsize = (5 * len(stim_loc_list), n_mice * 5))
            outer = gridspec.GridSpec(1, len(stim_loc_list), wspace=0.2, hspace=0.2)
            
            for i, stim_loc in enumerate(stim_loc_list):
                
                inner = gridspec.GridSpecFromSubplotSpec(n_mice, 1,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
                axes = []
                
                for j, stim_type in enumerate(stim_type_list):
                

                    c_laser = cmap_dict[pulse][j]  
                    label = ' '.join(stim_type.split('_')[1:]).replace('-', '.')
                    axes = run_one_intensity_save_data(pre_direct, scale_pix_to_cm, mouse_type, mouse_dict, stim_loc, stim_type, opto_par, treadmil_velocity,
                                                    ylim, spont_trial_dict, misdetection_dict, intervals_dict, t_window_dict, accep_interval_range, study_param_dict,
                                                    max_distance, min_distance, n_trials_spont, c_laser = c_laser, remove_empty_ax= False, annotate_n = False,
                                                    c_spont = 'k', fig = fig, outer = outer, inner = inner, axes = axes, label = label,
                                                    title_manually = False, plot_spont = False, save_fig = False, suptitle_y = suptitle_y)


            axes = rm_repeated_legend(axes)
            fig.text(0.5, xlabel_y, 'time (ms)', ha='center',
                      va='center').set_fontproperties(font)
            fig.text(ylabel_x, 0.5, study_param_dict['plot_param'], ha='center', va='center',
                      rotation='vertical').set_fontproperties(font)
            
            save_pdf_png(fig, os.path.join(pre_direct, 'Subplots', 
                                     'All_stim_type_and_loc_' 
                                     + opto_par 
                                     + '_' + pulse
                                     + '_' + mouse_type 
                                     + '_' + study_param_dict['cor']
                                     + '_' + study_param_dict['body_part'][0] 
                                     + '_timebin=' + str(t_window_dict['n_timebin']) 
                                     + "_moving_aver_win=" + str(t_window_dict['window_pos']) 
                                     + '_spont_sampl=' + str(n_samples) 
                                     + '_pre_post_stim'))
    return fig

def rm_shared_axis_labels(axes):
    
    """ Remove repeated axis labels"""
    
    for ax in axes.flat:
        ax.label_outer()

    return axes
def rm_repeated_legend(axes):
    
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        leg = ax.legend(by_label.values(), by_label.keys(),
                  frameon = False, fontsize = 16)
        p = [l.set_linewidth(3) for l in leg.legendHandles ]
    return axes

def no_files(files, mouse_no, dtype = 'DLC'):
    
    """ Report True the there are no 'dtype' 
    files for this mouse"""
    
    if len(files) == 0:
        
        print("No {} files for mouse # {}".format(dtype, mouse_no))
    
        return True
    
    else:
        
        return False
    
def save_npz_limb_and_tail(pre_direct, scale_pix_to_cm, mouse_type, mouse_list, stim_loc, stim_type, opto_par,
                           misdetection_dict, pre_interval, interval, post_interval, t_window_dict, accep_interval_range, cor_list, body_part_list, plot_param_list):
    """Save data of epochs and individal mice to a npz file.

    Run over all mice of one group and one intensity append epochs. 


    Parameters
    ----------
    pre_direct : str
            path to rooot directory
    scale_pix_to_cm : float
            Scaling coeficient for pixels to cm.
    mouse_type : str
            mouse type such as `FoxP2` or etc
    mouse_list : list(int)
            list of mouse identification numbers
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".           
    stim_type : str
            the folder with the corresponding experiment protocol e.g. `Square_1_mW`
    opto_par: str
            `ChR2` for ChR2 injected animals, `Control` for Control group
    misdetection_dict : dict-like 
            dictionary of constants used in the misdetection correction algorithm
    pre_interval : int 
            Number of time bins to be taken into account before laser onset 
    interval : int 
            Laser duration in timebins.
    post_interval : int
            Number of time bins to be taken into account after laser onset 
    t_window_dict : dict-like
            dictionary containing information about time contants of the project
    accep_interval_range: tuple(int)
            Smallest and largest acceptable trial sizes in timebins
    cor_list : list(str)
            list of available coordinates e.g. `x`, `y`
    body_part_list : list(str)
            list of DLC labeled body parts 
    plot_param_list : list(str)
            List of available parameters for analysis e.g position or velocity

    Returns
    -------

    """

    print(" 1. X \n 2. Y ")
    which_plot = int(input())-1  # ask what body part to plot
    print(" 1. Position \n 2. Velocity \n ")
    what_plot = int(input())-1  # ask what body part to plot
    cor = cor_list[which_plot]
    plot_param = plot_param_list[what_plot]

    epochs_limb = np.empty((0, pre_interval+interval+post_interval+1))
    epochs_tail = np.empty_like(epochs_limb)
    count = 0
    if opto_par == "Control":
        loop_list = mouse_list[1]
    else:
        loop_list = mouse_list[0]

    for n in loop_list:  # Run over all the mice
        count += 1

        #global mouse_no
        mouse_no = n
        print("Type : ", mouse_type+" # ", n)
        # directory to the stim_type for each mouse
        direct = os.path.join(pre_direct, mouse_type,
                              opto_par, 'Mouse_' + str(mouse_no))

        files_list_DLC = get_DLC_files(
            os.path.join(direct, stim_loc, stim_type, 'DLC'))
#         convert_csv_to_xlsx(direct+stim_type+'/Laser') # files might be given in csv, this is to unify
        files_list_Laser = get_laser_files(
            os.path.join(direct, stim_loc, stim_type, 'Laser'))
        if len(files_list_DLC) == 0:
            print("No files for mouse # ", n)
            continue
        elif len(files_list_Laser) == 0:
            print("No Laser detection for mouse # ", n)
            continue
        else:
            study_param_dict = {'cor': cor, 'body_part': [
                'HL'], 'plot_param': plot_param}
            epochs = extract_epochs_over_trials_one_side_one_body_part(files_list_DLC, files_list_Laser, direct, stim_loc, stim_type, scale_pix_to_cm, accep_interval_range,
                                                                       misdetection_dict, pre_interval, interval, post_interval,
                                                                       **t_window_dict, **study_param_dict)
            # construct an array of all the trial epochs of all mice
            epochs_limb = np.append(epochs_limb, epochs, axis=0)

            study_param_dict = {'cor': cor, 'body_part': [
                'Tail'], 'plot_param': plot_param}
            epochs = extract_epochs_over_trials_one_side_one_body_part(files_list_DLC, files_list_Laser, direct, stim_loc, stim_type, scale_pix_to_cm, accep_interval_range,
                                                                       misdetection_dict, pre_interval, interval, post_interval,
                                                                       **t_window_dict, **study_param_dict)
            # construct an array of all the trial epochs of all mice
            epochs_tail = np.append(epochs_tail, epochs, axis=0)

    file_name = (mouse_type+'_'+opto_par+'_'+stim_type+"_mov_aver="+str(int(t_window_dict['window_pos']/t_window_dict['fps']*1000)) +
                 "_n_t="+str(int(t_window_dict['n_timebin']/t_window_dict['fps']*1000))+'_'+cor+'_'+plot_param+'_pre_inter_'+str(int(pre_interval/t_window_dict['fps']*1000)) +
                 '_post_inter_'+str(int(post_interval/t_window_dict['fps']*1000)))

    path = os.path.join(pre_direct, 'data_npz', stim_type, opto_par, 'Limb_Tail')

    if not os.path.exists(path):
        os.makedirs(path)

    np.savez(os.path.join(path, file_name),
             epochs_all_mice_limb=epochs_limb,
             epochs_all_mice_tail=epochs_tail,
             cor=[cor],
             body_part=['HL', 'Tail'],
             plot_param=[plot_param])


def extract_epochs_over_trials_one_side_one_body_part(files_list_DLC, files_list_Laser, direct, stim_loc, stim_type, scale_pix_to_cm,
                                                      accep_interval_range, misdetection_dict, pre_interval, interval, post_interval,
                                                      fps, n_timebin, window_veloc, window_pos, cor, body_part, plot_param, left_or_right='right'):
    '''Return all the epochs of all simillar trials for one mouse.

    Parameters
    ----------
    files_list_DLC : list(str)
            List of paths to all DLC .csv files for sessions
    files_list_laser : list(str)
            List of paths to all .csv files that have (start,end) laser times.
    direct : str
            Path to the parent directory containing all stim_types
    stim_loc : str
            the folder with the corresponding experiment brain location e.g. "STR".   
    stim_type : str
            Folder containing sessions for this protocol  
    scale_pix_to_cm : float
    accep_interval_range: tuple(int)
            Smallest and largest acceptable trial sizes in timebins
    misdetection_dict : dict-like 
            dictionary of constants used in the misdetection correction algorithm:
            {'acc_deviance' : float  (acceptable deviance between right and left detections),
            'internal_ctr_dev' : float,
            'percent_thresh_align' : int,
            't_s' : int (number of time steps before and after to look at),
            'n_iter_jitter' : int (how many times go over data to find jitters and clear them out),
            'jitter_threshold' : int (max_speed/fps)}
    pre_interval : int 
            Number of time bins to be taken into account before laser onset 
    interval : int 
            Laser duration in timebins.
    post_interval : int
            Number of time bins to be taken into account after laser onset 
    pre_stim_inter : int
            Pre interval for averaging the positions/velocity and accelaration and report as pre_info
    fps : int
            Frame per second of the video
    n_timebin : int
            Number of frames as timebins for the x derivative to yields velocity
    window_veloc : int
            Velocity moving average window in frames
    window_pos : int
            Position moving average window in frames
    left_or_right : str, optional
            Either left side or right side. Default value is set to 'right'.

    Returns
    -------

    epochs : 2D-array
            (pre | Laser ON | post) epochs with shape (n_trials, pre_interval+post_interval+interval+1)

    '''

    epochs = np.empty((0, pre_interval+interval+post_interval+1))

    for i in range(0, len(files_list_DLC)):

        print('session {} out of {}'.format(i+1, len(files_list_DLC)))
        file_path_DLC = os.path.join(direct, stim_loc, stim_type, 'DLC', files_list_DLC[i])
        df = read_DLC(file_path_DLC, scale_pix_to_cm)

        right, left = position_r_l(
            df, window_pos, misdetection_dict, cor, body_part, plot_param)
        
        session_length = len(right)
        if left_or_right == 'left':
            variable = left
        else:
            variable = right
        file_path_Laser = os.path.join(
            direct, stim_loc, stim_type, 'Laser', files_list_Laser[i])
        laser_t = read_laser(file_path_Laser, file_path_DLC)
        bins = np.copy(laser_t.values).astype(int)
        epochs_trial, n_trials, take = extract_epochs(
            bins, variable, *accep_interval_range, pre_interval, interval, post_interval, session_length)

        epochs = np.append(epochs, epochs_trial, axis=0)

    return epochs


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
