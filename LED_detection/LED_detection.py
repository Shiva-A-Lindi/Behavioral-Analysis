# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
#from PyQt5 import QtWidgets,QtCore,QtGui
import csv
import numpy as np
import pandas as pd
from os import walk
import os
#--------------------------------------------------------------------------------------------------------------------PLAYER
import timeit
from threading import Thread

# import the Queue class from Python 3
import sys
if sys.version_info >= (3, 0):
	from queue import Queue
 
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue
    

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
        
        self.On_thresh = 50 # the threshold above which the LED is presumed to be on
        self.mask = dict([('Cue',None),('R_pad',None), ('L_pad',None), ('laser',None), ('reward', None)])
        self.intensity = np.zeros((len(self.mask)))
        self.switch = np.zeros((len(self.mask)))
        
    def get_pix(self,circles_cor,frame):
        
        count = 0
        for led in self.mask.keys():
            i = circles_cor[count,:]
            self.mask[led] = np.zeros((frame.height,frame.width))
            cv2.circle(self.mask[led],(i[0],i[1]),i[2],1,thickness = -1) #find the pixels inside the circle
            count = count +1 

def get_one_frame_from_video(videoPath):
    video=Video()
    video.capture = cv2.VideoCapture(videoPath)
    ret, video.currentFrame = video.capture.read()
    this_frame = cv2.cvtColor(video.currentFrame[400:-1 , 0:-1], cv2.COLOR_BGR2GRAY)   
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
        
    print("specify the center and one point on the edge for each circle")    
    cv2.namedWindow("specify circles")
    cv2.setMouseCallback("specify circles", get_circles_by_click)
    # keep looping until the 'q' key is pressed
    while True:
    	# display the image and wait for a keypress
        cv2.imshow("specify circles", img)
        if cv2.waitKey(1)==27 or len(inputs)==10:
            break
    
    circles_cor = np.zeros((n_circles,3))
    inputs = np.int16(np.array(inputs))
    
    for i in range (n_circles):
        center = inputs[i*2]
        edge = inputs[i*2+1]
        r = round(pow((pow((edge[0]-center[0]),2)+pow((edge[1]-center[1]),2)),0.5)).astype(int)
        circles_cor[i] = [center[0],center[1], r]
    circles_cor = np.uint16(np.around(circles_cor))
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    print("circle centers and radius:", circles_cor)
    for i in circles_cor:
    #         draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    print("press esc when you're done to close the windows")
    while True:
    	# display the image and wait for a keypress
        cv2.imshow("chosen circles", cimg)
        if cv2.waitKey(1)==27:
            break
    cv2.destroyAllWindows()
    return circles_cor

def write_to_csv(data,colnames, videoPath):
    ''' get the numpy array with the column naames and save it as a csv '''
    csv_path = videoPath[:-4]+'_LED.csv'
    result = pd.DataFrame(data = data, columns = colnames)
    result.to_csv(csv_path,index=False)
    
def build_videoPath_list(path):
    '''go over the directory tree in path and find ll .avi files'''
    fname = []
    for (dirpath, dirnames, filenames) in walk(path):
    
        for f in filenames:
    #         if dirpath[-4:]=='Left':
    #             fname.append(os.path.join(dirpath, f))
            fname.append(os.path.join(dirpath, f))
    videofile_path = [ fi for fi in fname if fi.endswith(".avi") ]
#    for i in videofile_path:
#        print(i)
    return videofile_path

def analyze_video(videoPath):
    video=Video()
    video.capture = cv2.VideoCapture(videoPath)
    video.width = int(video.capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    video.height = int(video.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    #video.fps = video.capture.get(cv2.CAP_PROP_FPS)
    video.nbFrames=int(video.capture.get(7))
    
    x0,x1 = [0 , video.width]
    y0,y1 = [400 , video.height]
    
    frame = Frame()
    frame.set_properties(x0,x1,y0,y1)
    
    LED = LED_class()
    
    if video.nbFrames != 0: # if th evideo is readable
        LED.get_pix(circles_cor,frame) # set the masks for each LED in the LED class
        LED_on_off, columns = analyze_frame(video,frame,LED)
        write_to_csv(LED_on_off, columns, videoPath)

def analyze_frame(video,frame,LED):
    n = 0
    LED_on_off = np.zeros((video.nbFrames,len(LED.mask)),dtype= int) # stores the state of each LED for each frame
    start = timeit.default_timer()
    while(video.capture.isOpened()):
        n = n +1
#        print("remaining frame = ", video.nbFrames - n)
        ret, video.currentFrame = video.capture.read()  
        try:
            this_frame = video.currentFrame[frame.y_crop_lim[0]:frame.y_crop_lim[1] , frame.x_crop_lim[0]:frame.x_crop_lim[1]]
            frame.no = n
            frame.gray_cropped = cv2.cvtColor(this_frame, cv2.COLOR_BGR2GRAY)    
            img = cv2.medianBlur(frame.gray_cropped,5)
            LED.switch = np.zeros((len(LED.mask)))
            cir_count = 0
            for i in LED.mask.keys():
        
                img_copy = img.copy()
                img_copy[LED.mask[i] == 0] = 0 # mask the complement of the circle
        #        cv2.imshow('detected circles',img_copy)
        #        cv2.waitKey(0)
                average_inten = np.sum(img_copy)/np.count_nonzero(img_copy)
                LED.intensity[cir_count] = average_inten
                cir_count = cir_count + 1

            ind, = np.where(LED.intensity > LED.On_thresh)
            LED.switch[ind] = 1 # if the intensity psses the thresh consider as swithched ON
            LED_on_off[n-1,:] = LED.switch

        except (AttributeError,TypeError):
            LED_on_off = np.delete(LED_on_off,np.arange(n,LED_on_off.shape[0]),axis = 0) # return the array for the detected frames
            break
        if cv2.waitKey(1)==27 or n == video.nbFrames: video.capture.release()
    stop = timeit.default_timer()
    print('runtime = ', int(stop - start)," sec")

    return LED_on_off,LED.mask.keys()

#%% 

# Rat_2
    
#path ="/media/shiva/LaCie/Nico_BackUp_Ordi-P1PNH-5/Données Valentin/videos/Rat_2"
#videoPath='/home/shiva/Desktop/Sophie/Left/Rat2_ArchT3_20mW_20190523_141130_C001H002S0001.avi'
#circles_cor = np.uint16(np.array([[34, 80, 14],[108, 90, 16],[184, 93, 18],[263, 93, 18],[343, 93, 18]]))

# Rat_12
    
path ="/media/shiva/LaCie/VideoRat_Sophie/videos_Rat12"
videoPath='/media/shiva/LaCie/VideoRat_Sophie/videos_Rat12/14-06-19/Rat 12 head 1 6OHDA x2 14-06-19_20190614_080116_C001H001S0001.avi'



videoPath_list = build_videoPath_list(path) #get list of video paths
image = get_one_frame_from_video(videoPath) # get one frame to specify circle coordinates on
circles_cor = get_circles(image) # specify circles on the frame

c = 0
for videoPath in videoPath_list:
    c += 1
    print("files left = ", len(videoPath_list)-c)
    analyze_video(videoPath)



#%%         Find circles with OpenCV and show
video=Video()
videoPath='/home/shiva/Desktop/Sophie/Left/Rat2_ArchT3_20mW_20190523_141130_C001H002S0001.avi'
videoPath='/media/shiva/LaCie/VideoRat_Sophie/videos_Rat12/14-06-19/Rat 12 head 1 6OHDA x2 14-06-19_20190614_080116_C001H001S0001.avi'

video.capture = cv2.VideoCapture(videoPath)
video.width = video.capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
video.height = video.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
video.fps = video.capture.get(cv2.CAP_PROP_FPS)
video.nbFrames=int(video.capture.get(7)) #nombre de frames
print("nb frames : ",video.nbFrames)

while(video.capture.isOpened()):

    ret, frame = video.capture.read()
    
    gray = cv2.cvtColor(frame[400:-1, 0:-1], cv2.COLOR_BGR2GRAY)
    img = gray
    img = cv2.medianBlur(img,5)
    circle = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=5,maxRadius=30)
    
    if circle is None:
        continue
    circles = np.uint16(np.around(circle))
    for i in circles[0,:]:
#         draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#         draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('frame',img)
    
    if cv2.waitKey(1)==27 :
        break

video.capture.release()
cv2.destroyAllWindows()
