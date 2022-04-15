# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import csv
import numpy as np
import pandas as pd
from os import walk
import os, sys
import subprocess 
import timeit
import pickle
try:
    import cv2
except ImportError or ModuleNotFoundError:

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])

# def conda_install(environment, packages):
#     proc = subprocess.run(["conda", "install", "--quiet"] + packages,
#                 text=True, capture_output=True)
#     return json.loads(proc.stdout)

# conda_install("base", ["opencv"])
# from threading import Thread
import cv2
# try:
#     import cv2
# except ImportError or ModuleNotFoundError:
#     run.check_call([sys.executable, '-m', 'conda', 'install', 'cv2'])

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

        
class Frame:
    """class for frames
    """
    def __init__(self):
        
        self.no = None
        self.gray_cropped = None # gray scaled and cropped frame
        self.LED_centers=None
        self.LED_radius = None
        self.LED_inten = None
        self.width=None #vidProp_dict['videoWidth']=cap.get(3)
        self.height=None #vidProp_dict['videoHeight']=cap.get(4)   
        self.On_LEDs_frame = None # frame circling the ON detected LEDs
        self.x_crop_lim = None
        self.y_crop_lim = None
        
    def set_properties(self,x0,x1,y0,y1):
        ''' set the height and width of frames by getting the cropping limits'''
        
        self.x_crop_lim = [x0 , x1]
        self.y_crop_lim = [y0 , y1]
        self.width = self.x_crop_lim[1]- self.x_crop_lim[0]
        self.height = self.y_crop_lim[1]- self.y_crop_lim[0]
        
class LED_class:
    ''' class holding masks of LEDs on the frame '''
    
    def __init__(self):
        
        self.mask = dict([('Cue',None),('R_pad',None), ('L_pad',None), ('laser',None), ('reward', None)])
        self.intensity = np.zeros((len(self.mask)))
        self.on_thresh = np.zeros((len(self.mask))) # the threshold above which the LED is presumed to be on
        self.on_range = np.full((len(self.mask)), 3/4) # the range between min and max intensity 
        # where the LED is considered on. Now on-thresh is min+(1-x)(max-min) where x is 3/4 for all LEDs
        self.switch = np.zeros((len(self.mask)))
        
    def get_pix(self,circles_cor,frame):
        
        count = 0
        for count, led in enumerate( self.mask.keys() ):
            
            i = circles_cor[count,:]
            self.mask[led] = np.zeros((frame.height,frame.width))
            cv2.circle(self.mask[led],(i[0],i[1]),i[2],1,thickness = -1) #find the pixels inside the circle

def get_one_frame_from_video(videoPath):
    
    video=Video()
    video.capture = cv2.VideoCapture(videoPath)
    ret, video.currentFrame = video.capture.read()
#    this_frame = cv2.cvtColor(video.currentFrame[y0:y1 , x0:x1], cv2.COLOR_BGR2GRAY)   
    this_frame = cv2.cvtColor(video.currentFrame, cv2.COLOR_BGR2GRAY)   

    video.capture.release()
    
    return this_frame

def get_circles(img):
    ''' get the center and one point on the edge for each circle you want to 
    detect with mouse clicks on the frame shown in a pop up window '''
    
    global press, inputs, n_circles
    
    n_circles = 5 # number of circles to be identified
    press = False
    inputs = []
    
    def get_circles_by_click(event, x, y, flags, param):
        # grab references to the global variables
        global press, inputs
        for n in range(n_circles):
        	if event == cv2.EVENT_LBUTTONDOWN and press == False:
        		inputs.append([x, y])
        		press = True
        	# check to see if the left mouse button was released
        	elif event == cv2.EVENT_LBUTTONUP:
        		press = False
        
    print("Specify the center and one point on the edge for each LED. \n")    
    cv2.namedWindow("Specify circles")
    cv2.setMouseCallback("Specify circles", get_circles_by_click)
    
    # keep looping until the 'q' key is pressed
    while True:
    	# display the image and wait for a keypress
        cv2.imshow("Specify circles", img)
        
        if cv2.waitKey(1)==27 or len(inputs)== int(2*n_LED):
            
            break
    
    circles_cor = np.zeros((n_circles,3))
    inputs = np.int16(np.array(inputs))
    
    for i in range (n_circles):
        center = inputs[i*2]
        edge = inputs[i*2+1]
        r = int(round(pow((pow((edge[0]-center[0]),2)+pow((edge[1]-center[1]),2)),0.5)))#.astype(int)
#        print(r.type())
        circles_cor[i] = [center[0],center[1], r]
    circles_cor = np.uint16(np.around(circles_cor))
    print("Circle centers and radii ([X, Y, R]): \n", circles_cor)
    
    draw_circles_on_img(circles_cor, img)
    return circles_cor

def draw_circles_on_img(circles_cor, img):
    
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    
    for i in circles_cor:
    #         draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        
    print("Press esc when you're done to close the windows. \n")
    
    while True:
        
    	# display the image and wait for a keypress
        cv2.imshow("Specified circles", cimg)
        
        if cv2.waitKey(1)==27:
            
            break
        
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
def crop_with_user_specified_corners(img, show_circles = False, circles_cor = ""):
    
    ''' get the center and one point on the edge for each circle you want to 
    detect with mouse clicks on the frame shown in a pop up window '''
    
    global press, inputs, n_points
    
    while True: # To give the chance to user to try cropping multiple times
    
        n_points = 2 # number of points to be identified
        press = False
        inputs = []
        
        def get_corners_by_click(event, x, y, flags, param):
            
            # grab references to the global variables
            global press, inputs
            
            for n in range(n_points):
    
                if event == cv2.EVENT_LBUTTONDOWN and press == False:
                    inputs.append([x, y])
                    press = True
            	# check to see if the left mouse button was released
                elif event == cv2.EVENT_LBUTTONUP:
                    press = False
    
        print("Now specify the upper left and lower right corner for cropping. \n")    
        cv2.namedWindow("Specify corners to crop")
        cv2.setMouseCallback("Specify corners to crop", get_corners_by_click)
    
        # keep looping until the 'q' key is pressed
        while True:
            
        	# display the image and wait for a keypress
            cv2.imshow("Specify corners to crop", img)
            
            if cv2.waitKey(1)==27 or len(inputs)==n_points:
                
                break
    
        crop_cor = np.zeros((n_points,2))
        inputs = np.int16(np.array(inputs))
        crop_cor = inputs[:2]
    
        cimg = cv2.cvtColor(img[crop_cor[0,1]:crop_cor[1,1], crop_cor[0,0]:crop_cor[1,0] ],cv2.COLOR_GRAY2BGR)
        
        print("Corners:\n", crop_cor[0],crop_cor[1])
        print("Press esc when you're done to close the windows. \n ")
        
        while True:
            
        	# display the image and wait for a keypress
            cimg_ful = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            cv2.rectangle(cimg_ful, (inputs[0][0],inputs[0][1]), (inputs[1][0],inputs[1][1]), (0, 255, 0), 2)
            cv2.imshow("Specify corners to crop",cimg_ful)
            cv2.imshow("Cropped frame", cimg)
            
            if cv2.waitKey(1)==27:
                
                break
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        x0, x1, y0, y1 = crop_cor[0,0], crop_cor[1,0], crop_cor[0,1], crop_cor[1,1]
        cropped_image = crop_frame(img, x0, y0, x1, y1)
        
        if show_circles:
            
                draw_circles_on_img(circles_cor, cropped_image)
                
        if_recrop = input("If you're not satisfied, do you want to crop again (y/[n])?")
        
        if if_recrop in ['n', 'N', '']:
            
            break

    # x0,x1,y0,y1 = crop_cor[0,0],crop_cor[1,0],crop_cor[0,1],crop_cor[1,1]
    return x0,x1,y0,y1

def crop_frame(image, x0,y0,x1,y1):
    
    return image[y0:y1,x0:x1]

def get_pad_and_lever_from_user(img):
    
    ''' get the center and one point on the edge for each circle you want to 
    detect with mouse clicks on the frame shown in a pop up window '''
    
    global press, inputs, n_points
    
    n_points = 3 # number of points to be identified
    press = False
    inputs = []
    cimg_ful = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    
    def get_corners_by_click(event, x, y, flags, param):
        
        # grab references to the global variables
        global press, inputs
        
        for n in range(n_points):

            if event == cv2.EVENT_LBUTTONDOWN and press == False:
                
                inputs.append([x, y])
                press = True
        	# check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                
                press = False


                  
        
    print("Specify the left and right end of the pad and the lever location, resepectively. \n ")    
    cv2.namedWindow("Specify pad and lever positions")
    cv2.setMouseCallback("Specify pad and lever positions", get_corners_by_click)
    
    # keep looping until the 'q' key is pressed
    while True:
    	# display the image and wait for a keypress
        cv2.imshow("Specify pad and lever positions", img)
        if cv2.waitKey(1)==27 or len(inputs)==n_points:
            break
    
    inputs = np.int16(np.array(inputs))
    pad_ends = inputs[:2]
    lever = inputs[2:3]

    print("Pad edges (x,y): \n", pad_ends[0],pad_ends[1])
    print("Lever location (x,y): \n", lever[0])
    print("Press esc when you're done to close the windows. \n ")
    
    while True:
    	# display the image and wait for a keypress
#        cv2.imshow("cropped frame", cimg)
        cv2.circle(cimg_ful,(pad_ends[0][0],pad_ends[0][1]),5,(0,0,255),-1)
        cv2.circle(cimg_ful,(pad_ends[1][0],pad_ends[1][1]),5,(0,0,255),-1)
        cv2.circle(cimg_ful,(lever[0][0],lever[0][1]),5,(0,255,0),-1)
        cv2.imshow("Specify pad and lever positions",cimg_ful)
        
        if cv2.waitKey(1)==27:
            break
        
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return pad_ends,lever

def write_to_csv(data,colnames, videoPath,pad_ends,lever):
    
    ''' get the numpy array with the column names and save it as a csv '''
    
    csv_path = videoPath.split('.')[0]+'_LED.csv'
    LED_data = pd.DataFrame(data = data, columns = colnames)
    pad_y = int((pad_ends[0,1]+pad_ends[1,1])/2)
    supp_df = pd.DataFrame([['pad_left', pad_ends[0,0], pad_y], ['pad_right', pad_ends[1,0], pad_y], 
                            ['lever', lever[0,0], lever[0,1]]],columns=['point','x', 'y'])
    result = pd.concat([LED_data, supp_df ], axis=1)
    result.to_csv(csv_path,index=False)
    
def build_videoPath_list(path, parent_folder_name = ""):
    
    # path = "/media/shiva/Seagate Expansion Drive/DeepLabCut_Analysis/Rat_14/2019-05-02_D-6"
    # parent_folder_name = "Left"
    
    '''go over the directory tree in path and find ll .avi files'''
    
    fname = []
    for (dirpath, dirnames, filenames) in walk(path):
    
        for f in filenames:
            
            folder_name = os.path.basename(os.path.normpath(dirpath))
            
            if parent_folder_name in folder_name:
                
                fname.append(os.path.join(dirpath, f))
                
    videofile_path = [ fi for fi in fname if (fi.endswith(".avi") or 
                                              fi.endswith(".mp4") or 
                                              fi.endswith(".mkv") or
                                              fi.endswith(".wmv") or
                                              fi.endswith(".mpeg"))]

    return videofile_path

def analyze_video(circles_cor, videoPath,on_thresh,pad_ends,lever,x0,x1,y0,y1):
    
    video=Video()
    video.capture = cv2.VideoCapture(videoPath)
    video.width = int(video.capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    video.height = int(video.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    #video.fps = video.capture.get(cv2.CAP_PROP_FPS)
    video.nbFrames=int(video.capture.get(7))

    
    frame = Frame()
    frame.set_properties(x0,x1,y0,y1)
    
    LED = LED_class()
    LED.on_thresh = on_thresh
    
    if video.nbFrames != 0: # if th evideo is readable
        LED.get_pix(circles_cor,frame) # set the masks for each LED in the LED class
        LED_on_off, columns,on_threshold,video_corrupted = analyze_frame(video,frame,LED, find_threshold = False)
        write_to_csv(LED_on_off, columns, videoPath, pad_ends, lever)
        
def find_on_threshold(circles_cor, videoPath_list,x0,x1,y0,y1,on_range):
    
    print("Wait while the LED on-thresholds are set. This will take a few seconds.. \n ")
    
    video_corrupted = True
    count = 0
    
    while (video_corrupted == True):
        
        videoPath = videoPath_list[count]
        video=Video()
        video.capture = cv2.VideoCapture(videoPath)
        video.width = int(video.capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        video.height = int(video.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        #video.fps = video.capture.get(cv2.CAP_PROP_FPS)
        video.nbFrames=int(video.capture.get(7))
        
        frame = Frame()
        frame.set_properties(x0,x1,y0,y1)
        
        LED = LED_class()
        LED.on_range = on_range # set on range by input from user
        
        if video.nbFrames != 0: # if the video is readable
        
            LED.get_pix(circles_cor,frame) # set the masks for each LED in the LED class
            LED_on_off, columns, on_threshold, video_corrupted = analyze_frame(video,frame,LED, find_threshold = True)
            
        count += 1
    
    print("Thresholds are as following = ", on_threshold)
    print("make sure all the relevant LEDs are turned on at least once in the reference video")
    
    return on_threshold, videoPath

def analyze_frame(video,frame,LED, find_threshold = False):
    
    global LED_intensity
    
    
    LED_on_off = np.zeros((video.nbFrames,len(LED.mask)),dtype= int) # stores the state of each LED for each frame
    LED_intensity = np.zeros((video.nbFrames,len(LED.mask)))
    start = timeit.default_timer()
    
    n = 0
    
    while ( video.capture.isOpened()):
        
        n = n + 1
#        print("remaining frame = ", video.nbFrames - n)
        ret, video.currentFrame = video.capture.read()  
        
        try:
            
            video_corrupted = False
            this_frame = video.currentFrame[frame.y_crop_lim[0]:frame.y_crop_lim[1] , frame.x_crop_lim[0]:frame.x_crop_lim[1]]
            frame.no = n
            frame.gray_cropped = cv2.cvtColor(this_frame, cv2.COLOR_BGR2GRAY)    
            img = cv2.medianBlur(frame.gray_cropped,5)
            LED.switch = np.zeros((len(LED.mask)))
            cir_count = 0
            global exp
            
            for i in LED.mask.keys():
                
                img_copy = img.copy()
                img_copy[LED.mask[i] == 0] = 0 # mask the complement of the circle
                
                if i=='R_pad': exp = img_copy
#                cv2.imshow('detected circles',img_copy)
#                cv2.waitKey(0)
                average_inten = np.sum(img_copy)/np.count_nonzero(img_copy)
                LED.intensity[cir_count] = average_inten
                cir_count = cir_count + 1

            ind = np.where((LED.intensity - LED.on_thresh) > 0)[0]
            # print("inten \n", LED.intensity)
            # print("thresh \n", LED.on_thresh)
            # print(ind)
            LED.switch[ind] = 1 # if the intensity psses the thresh consider as swithched ON
#            if sum(LED.intensity) == 0:
#                print(LED.intensity,LED.switch)
#                cv2.imshow('detected circles',img)
##                cv2.waitKey(0)
#                if cv2.waitKey(1)==27:
#                    cv2.destroyAllWindows()
#                    cv2.waitKey(1)
            LED_on_off[n-1,:] = LED.switch
            LED_intensity[n-1,:] = LED.intensity
            
        except TypeError: # TypeError ==> no more readable frames, AttributeError ==> couldn't open video at all
        
            print("No more readable frames",'\n',"number of frames read = ",frame.no)
            
            if n > 1500: # 1500 frames is enough for setting threshold
            
                break
            
            else:
                
                if video.nbFrames == 0: print("Couldn't open video")
                video_corrupted = True
                break
#            LED_on_off = np.delete(LED_on_off,np.arange(n,LED_on_off.shape[0]),axis = 0) # return the array for the detected frames
            
        if cv2.waitKey(1)==27 or n == video.nbFrames: video.capture.release()
    
    LED_intensity = np.delete(LED_intensity,np.arange(n-1,LED_intensity.shape[0]),axis = 0)
    LED_on_off = np.delete(LED_on_off,np.arange(n-1,LED_on_off.shape[0]),axis = 0)
    stop = timeit.default_timer()
    print('runtime = ', int(stop - start)," sec")
    
    # return the on threshold as the average of the max and min intensity
    max_intensity = np.amax(LED_intensity, axis = 0)
    min_intensity = np.amin(LED_intensity, axis = 0)    
    
    # LED.on_thresh = (max_intensity - min_intensity)*(LED.on_range/100) + min_intensity
    if find_threshold:
        on_thresh = (max_intensity - min_intensity)*(LED.on_range/100) + min_intensity
    
        # ind_bad, = np.where(max_intensity - min_intensity < 100) # the LED never turned on which happens for Cue
        # #but we don't wanna miss that in the next videos so we set an average threshold of others
        # if len(ind_bad) != 0:
        #     ind_good_to_go, = np.where(max_intensity - min_intensity > 20)
        #     # LED.on_thresh[ind_bad] = np.average(LED.on_thresh[ind_good_to_go]) 
        #     on_thresh[ind_bad] = np.average(on_thresh[ind_good_to_go]) 
        ########### Modification because LED was found always ON
        ind_bad = np.logical_or(max_intensity - min_intensity < 80, min_intensity > 100) # the LED never turned on which happens for Cue or the LED was always on
        #but we don't wanna miss that in the next videos so we set an average threshold of others
        if len(ind_bad) != 0:
            
            ind_good_to_go = ~ind_bad
            # LED.on_thresh[ind_bad] = np.average(LED.on_thresh[ind_good_to_go]) 
            on_thresh[ind_bad] = np.average(on_thresh[ind_good_to_go]) 
    else:
        
        on_thresh = np.nan
        
    print("max intensities = ", max_intensity.astype(int))
    print("min intensities = ", min_intensity.astype(int), "\n")
    
    return LED_on_off, LED.mask.keys(), on_thresh, video_corrupted


def set_LED_on_range(n_LED, default_on_range):
    
    while(True):
        
        try:
            
            on_range_str = []
            on_range_str = input("Above what percentage of max intensity should be considered switched on? (Either press enter to set the default (25%) or input the percentage value for each LED seperated by spaces.) \n").split(' ')
           
            if on_range_str[0] == '' or on_range_str[0] == 'exit':
                
                print("\n default value is set. LEDs with >25% intensity are considered as switched on. \n")
                on_range = np.full((n_LED), default_on_range)
                break
            
            else:
                
                on_range = np.array([int(x.strip()) for x in on_range_str])
                
                if len(on_range) != n_LED:
                    
                    raise NameError
                    
                else:
                    
                    break
                
        except NameError:
            
            print(("\n \n The number of inputs must be equal to the number of LEDs (n={})!. Try again! \n ".format(n_LED)))
            
        except ValueError:
            
            print("\n \n Invalid input. Try again. Make sure there are no extra spaces in your entry... \n ")
    
    return on_range

def set_LED_threshold(circles_cor, videoPath_list, path,x0,x1,y0,y1, bypass_user = False, path_to_ref = ""):
    
    while True:
        
        if not bypass_user:
            
            user_input = input("Do you want to derive thresholds (Press Enter)  or read from previously saved .csv file (Enter the full path to the file)? \n")
        
        else:
            
            user_input = path_to_ref
            
        if user_input == "" :
            
            on_range = set_LED_on_range(n_LED, default_on_range)
            on_thresh, videoPath = find_on_threshold(circles_cor, videoPath_list, x0,x1,y0,y1, on_range)
            save_thresh_based_on_user_command(on_thresh, user_input, videoPath, path)  
            
            break
        
        elif os.path.isfile(user_input): 
            
            on_thresh = read_csv(user_input)[0]
            # save_thresh_based_on_user_command(on_thresh, derive_thresh_input, videoPath, path)    
            break
        
    print("\n The LED thresholds are as following: {} \n".format((on_thresh).astype(int)))
    
    return on_thresh

def save_thresh_based_on_user_command(on_thresh, derive_thresh_input, videoPath, path):
    
    while True:
        
        if_save_thresh = input("Do you want to save the thresholds (y/[n])? \n")
        
        if if_save_thresh in ["y", "Y"]:
            
            save_on_thresh(on_thresh, videoPath, path, os.path.join(path, 'on_threshold.csv'))
            break 
        
        elif if_save_thresh in ["n", "N", ""]:
            
            break
        
        else:
            
            print("Invalid input. Try again. If you are entering a path make sure the path exists. \n")
            
def save_on_thresh(on_thresh, videoPath, path, csv_path):
    
    columns = ['Cue','R_pad','L_pad', 'laser', 'reward']
    df = pd.DataFrame(on_thresh.reshape(1,5), columns = columns)
    supp_df = pd.DataFrame([videoPath], columns = ['videofile_reference'])
    result = pd.concat([df, supp_df ], axis=1)
    result.to_csv(csv_path, index = False)
    
def read_csv(path):
    
    df = pd.read_csv(path)
    df = df.drop(['videofile_reference'], axis = 1)
    output = df.to_numpy()
    
    return output


def set_LED_circle_coords(cropped_image, videoPath, path, show_circles = True, bypass_user = False, path_to_ref = ""):
    
    remark = False
    
    while True:
        
        if not bypass_user:
            
            user_input = input("Do you want to mark LED circles (Press Enter)  or read from previously saved .csv file (Enter the full path to the file)? \n")
        
        else:
            
            user_input = path_to_ref
        
        if user_input == "" or remark == True: 
            
            circles_cor = mark_circles(cropped_image)
            break
    
        elif os.path.isfile(user_input): 
            
            circles_cor = read_csv(user_input)
            
        if show_circles:
            
            draw_circles_on_img(circles_cor, cropped_image)
            
        circles_acceptable = input("In case circles don't match, Do you want to mark them again (y/[n])?\n")
        
        if circles_acceptable in ['n', 'N', ""]:
            
            remark = False
            break
        
        else:
            
            remark = True
                
    print("\n The LED coordinates are as following: {} \n".format((circles_cor).astype(int)))
    
    if user_input == "":
        
        save_LED_circle_coords_based_on_user_command(circles_cor, videoPath, path)
        
    return circles_cor


def mark_circles(cropped_image):
    
    while True:
        
        circles_cor = get_circles(cropped_image) # specify circles on the frame
        inp = input("Do you want to try marking the circles again ([n]/y)? \n")
        
        if inp in ["n", "N", ""]:
            
            break
        
    return circles_cor

def save_LED_coords(LED_coords, videoPath, path, csv_path):
    
    columns = ['Center_X','Center_Y','Radius']
    df = pd.DataFrame(LED_coords, columns = columns)
    supp_df = pd.DataFrame([videoPath], columns = ['videofile_reference'])
    result = pd.concat([df, supp_df ], axis=1)
    result.to_csv(csv_path,index=False)
    
def save_LED_circle_coords_based_on_user_command(LED_coords, videoPath, path):
    
    while True:
        
        if_save = input("Do you want to save the coordinates of the LED circles to a file (y/[n])? \n")
        
        if if_save in ['y', 'Y']:
            
            save_LED_coords(LED_coords, videoPath, path, os.path.join(path, 'LED_cirlce_coords.csv'))
            break
        
        elif if_save in ['n', 'N', '']:
            
            print("Coords not saved. \n")
            break
        
        else:
            
            print("invalid input. try again (y/[n]) \n")
            
    
def analyze_list_of_videos(circles_cor, videoPath_list,on_thresh,pad_ends,lever, x0, x1,y0,y1):
    
    for c, videoPath in enumerate(videoPath_list):
        
        csv_path = videoPath.split('.')[0]+'_LED.csv'
        # if os.path.isfile(csv_path):
        #     print("File is already analyzed. Moving on ... \n")
        #     continue
        
        print("analyzing file {} of {} files".format(c+1, len(videoPath_list)))
        print("Filename = \n",videoPath)
        
        analyze_video(circles_cor, videoPath,on_thresh,pad_ends,lever, x0,x1,y0,y1)
        
def analyze_based_on_user_command(circles_cor, videoPath_list,on_thresh,pad_ends,lever, x0, x1,y0,y1, bypass_user = False):
    
    while True:
        
        if not bypass_user:
            
            if_analyse_all = input("Do you want to proceed with analysis of all video files (n = {}) ([y]/n)? \n".format(len(videoPath_list)))
       
        else: 
            
            if_analyse_all = ''
        
        if if_analyse_all in ['y', "Y", ""]:
                
            analyze_list_of_videos(circles_cor, videoPath_list,on_thresh,pad_ends,lever, x0, x1,y0,y1)
            break
        
        elif if_analyse_all in ["n", "N"]:
            
            print("No analysis performed.. \n")
            break
        
        else:
            
            print("\n \n Invalid input. Try again ([y]/n). \n")
            
def get_path_from_user():
    
    while True:
        
        path = input("Enter the full path under which the video file hierarchy resides: \n (There must not be any videos other than the ones you want analyzed in this directory tree.) \n")     
        
        if os.path.exists(path):
            
            break
        
        else:
            
            print("Path doesn't exist. Try entering the path again... \n")
    return path

def get_videofile_paths(path, bypass_user = False, parent_folder_name = ""):
    while True:
        if not bypass_user:
            parent_folder_name = input("If you want to look under specific subfolders (e.g. Left/Right) enter the subfolder name, otherwise press enter to get all the videos. \n")

        print("looking under :", path)
        videoPath_list = build_videoPath_list(path, parent_folder_name = parent_folder_name) #get list of video paths
        
        if len(videoPath_list) > 0:
            
            break
        
        else:
            
            if not bypass_user:
                
                print("No videos in this subfolder. Try entering another subfoler name. \n")
                
            else:
                
                break
            
    return videoPath_list

def save_list_to_txt(filepath, liste):

    with open(filepath, 'w') as f:
        
        for item in liste:
            
            f.write("%s\n" % item)
            
def read_list_of_files_from_txt(path_to_txt):
    
    file_list = open(path_to_txt, "r")
    
    return file_list

def get_list_of_marks_all_exp_folders(dir_path, save_pkl = True, info_dict_path = None):
    
    global info_dict
    
    folders  =  os.listdir(dir_path)
    folders_path_list = [os.path.join(dir_path, f) for f in folders if "." not in f]
    # parent_folder_name = input("If you want to look under specific subfolders (e.g. Left/Right) enter the subfolder name, otherwise press enter to get all the videos. \n")
    parent_folder_name = ""
    info_dict = {}
    df_pad = pd.DataFrame(columns = ['pad_left_x', 'pad_left_y', 'pad_right_x', 'pad_right_y'])
    df_lever = pd.DataFrame(columns = ['lever_x', 'lever_y'])
    df_crop = pd.DataFrame(columns = ['crop_left_x', 'crop_right_x', 'crop_left_y', 'crop_right_y'])
    # on_thresh_csv_path= input("If you want to use pre-derived LED thresholds enter the full path to the on_threshold reference file. Otherwise if you want it to be set for each folder separetly press Enter.")
    # circles_cor_csv_path= input("If you want to use pre-marked circle coordinates enter the full path to the LED circle coord reference file, otherwise press Enter.")
    on_thresh_csv_path = "/media/shiva/LaCie/Video_TestDLC_Shiva_Nico/Rat_12/2019-05-22_D-8/on_threshold.csv"
    circles_cor_csv_path= "/media/shiva/LaCie/Video_TestDLC_Shiva_Nico/Rat_12/2019-05-22_D-8/LED_cirlce_coords.csv"
    count = 0
    folders_path_list_analyzed = []
    
    for i, path in enumerate(folders_path_list[15:]):
        
        print("folder {0} from {1} folders".format(i, len(folders_path_list)))
        videoPath_list = get_videofile_paths(path, bypass_user=True, parent_folder_name = parent_folder_name)
        folders_path_list_analyzed.append(path)

        if len(videoPath_list) == 0:
            
            print("folder: ", path , 'contains no video files')
            folders_path_list.remove(path)
            continue
        
        image = get_one_frame_from_video(videoPath_list[0]) # get one frame to specify circle coordinates on
        pad_ends,lever = get_pad_and_lever_from_user(image)
        x0,x1,y0,y1 = crop_with_user_specified_corners(image)#, show_circles = True, circles_cor=circles_cor) # get coordinates for cropping
        cropped_image = image[y0:y1, x0:x1]#crop_frame(image, x0,x1,y0,y1)

        circles_cor = set_LED_circle_coords(cropped_image, videoPath_list[0], path, 
                                            show_circles = True, bypass_user=True, path_to_ref = circles_cor_csv_path)
        
        info_dict[path] = {'pad_ends':pad_ends, 'lever':lever, 'crop':[x0,x1,y0,y1]}
        df_pad.loc[i] = pad_ends.flatten()
        df_crop.loc[i] = [x0,x1,y0,y1]
        df_lever.loc[i] = lever.flatten()
        
    df_files = pd.DataFrame(folders_path_list_analyzed, columns= ['path'])
    df = pd.concat([df_crop, df_pad, df_lever, df_files], axis = 1)
    
    if save_pkl:
        
        pickle_obj(info_dict, info_dict_path)
        
    return  df, parent_folder_name, info_dict, on_thresh_csv_path, circles_cor

def pickle_obj(obj, filepath):

    with open(filepath, "wb") as file_:
        pickle.dump(obj, file_)  # , -1)


def load_pickle(filepath):

    return pickle.load(open(filepath, "rb"))  # , -1)

def main_with_pre_aqui_marks(parent_folder_name, info_dict, on_thresh_csv_path, circles_cor):

    for path in list(info_dict.keys()):
        
        videoPath_list = get_videofile_paths(path, bypass_user=True, parent_folder_name = parent_folder_name)
        print(videoPath_list)
        
        # image = get_one_frame_from_video(videoPath_list[0]) # get one frame to specify circle coordinates on
            
        #on_thresh = check_intensity_threshold(image,x0,x1,y0,y1) # to set the threshold manually
        
        on_thresh = set_LED_threshold(circles_cor, videoPath_list, path, *info_dict[path]['crop'], 
                                      bypass_user = True, path_to_ref=on_thresh_csv_path)
        
        analyze_based_on_user_command(circles_cor, videoPath_list, on_thresh,  info_dict[path]['pad_ends'],  
                                      info_dict[path]['lever'],  *info_dict[path]['crop'], bypass_user = True)
                
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        

if __name__ == '__main__':
    
    
    global n_LED
    
    n_LED = 5 #default number of LED lights
    default_on_range = 25 #by default if each LED is lit by 25% of max intensity, it's considered switched on.
    analyze_in_bulk = input("Do you want to mark the pad, lever, and crop corners for all experiment folders and use templates for LED coordinates and thresholds (y/[n])? This way all the analysis will automatically be performed after you mark all folders.")
    
    if analyze_in_bulk:
        
        # dir_path = input("Enter the full path for the directory that contains all the experiment folders")
        dir_path = "/media/shiva/LaCie/Video_TestDLC_Shiva_Nico/Rat_12"
        df, parent_folder_name, info_dict, on_thresh_csv_path, circles_cor = get_list_of_marks_all_exp_folders(dir_path,save_pkl = True, 
                                                                                                               info_dict_path=os.path.join(dir_path, 'dictionary_of_ent_marks.pkl'))
		
        df.to_csv(os.path.join(dir_path, os.path.basename(os.path.normpath(dir_path)) + '_summary_of_files_analyzed.csv'), index= False)
        main_with_pre_aqui_marks(parent_folder_name, info_dict,on_thresh_csv_path, circles_cor)
        
    else:
        
        path = "/media/shiva/LaCie/Video_TestDLC_Shiva_Nico/Rat_12"
        # path =  get_path_from_user()
        videoPath_list = get_videofile_paths(path)
        image = get_one_frame_from_video(videoPath_list[0]) # get one frame to specify circle coordinates on
        pad_ends,lever = get_pad_and_lever_from_user(image)
        x0,x1,y0,y1 = crop_with_user_specified_corners(image) # get coordinates for cropping
        cropped_image = crop_frame(image, x0,y0,x1,y1)
        circles_cor = set_LED_circle_coords(cropped_image, videoPath_list[0], path)
            
        #on_thresh = check_intensity_threshold(image,x0,x1,y0,y1) # to set the threshold manually
        
        on_thresh = set_LED_threshold(circles_cor, videoPath_list, path,x0,x1,y0,y1)
        
        analyze_based_on_user_command(circles_cor, videoPath_list,on_thresh,pad_ends,lever, x0, x1,y0,y1)
                
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
def check_intensity_set_threshold(image,x0,x1,y0,y1):
    
    ''' when inted to set manually show the derived average intensities of LEDs for one 
        frame and get the on-theshold from the user as input'''
        
    frame = Frame()
    frame.set_properties(x0,x1,y0,y1)
    LED = LED_class(on_thresh = 80)
    LED.get_pix(circles_cor,frame) # set the masks for each LED in the LED class

    img = image[y0:y1,x0:x1]
    img = cv2.medianBlur(img,5)
    cir_count = 0
    
    for i in LED.mask.keys():
        
        img_copy = img.copy()
        img_copy[LED.mask[i] == 0] = 0 # mask the complement of the circle
#                cv2.imshow("mask circles", img_copy)
#                cv2.waitKey(0)
        average_inten = np.sum(img_copy)/np.count_nonzero(img_copy)
        LED.intensity[cir_count] = average_inten
        cir_count = cir_count + 1

    print("average pix inensities for LEDs are: \n",np.around(LED.intensity))
    print("press esc then input the mean pix intensity for LED-on threshold: ")
    cv2.imshow('frame',image)
    cv2.waitKey(0)
    
    on_thresh = int(input()) # get the threshold as input
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return on_thresh


#%%         Find circles with OpenCV and show
#video=Video()
#videoPath='/home/shiva/Desktop/Sophie/Left/Rat2_ArchT3_20mW_20190523_141130_C001H002S0001.avi'
#videoPath='/media/shiva/LaCie/VideoRat_Sophie/videos_Rat12/14-06-19/Rat 12 head 1 6OHDA x2 14-06-19_20190614_080116_C001H001S0001.avi'
#
#video.capture = cv2.VideoCapture(videoPath)
#video.width = video.capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
#video.height = video.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
#video.fps = video.capture.get(cv2.CAP_PROP_FPS)
#video.nbFrames=int(video.capture.get(7)) #nombre de frames
#print("nb frames : ",video.nbFrames)
#
#while(video.capture.isOpened()):
#
#    ret, frame = video.capture.read()
#    
#    gray = cv2.cvtColor(frame[400:-1, 0:-1], cv2.COLOR_BGR2GRAY)
#    img = gray
#    img = cv2.medianBlur(img,5)
#    circle = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
#                                param1=50,param2=30,minRadius=5,maxRadius=30)
#    
#    if circle is None:
#        continue
#    circles = np.uint16(np.around(circle))
#    for i in circles[0,:]:
##         draw the outer circle
#        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
##         draw the center of the circle
#        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
#
#    cv2.imshow('frame',img)
#    
#    if cv2.waitKey(1)==27 :
#        break
#
#video.capture.release()
#cv2.destroyAllWindows()
#cv2.waitKey(1)
