# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:39:00 2019

@author: courtand

how to use (windows-python*****************)
first : create a list containing all videos paths

windows > dir/b/s *.avi > list.txt
linux > find "$PWD"/ -name "*.avi" > list.txt


then : - open anaconda prompt
       - launch >python OptoDetect-5b.py "path/to/video list file.txt"

*************************************************************************
version 5 : correction bug : réinitialisation des listes après sauvegarde
version 5b : correction décalage de l'index de détection des stimulations
    par rapport à ce que l'on observe



Distribution : pyinstaller
>conda install -c conda-forge pyinstaller 

how to use (windows exe)
- open cmd.exe
- launch : >"path\to\OptoDetect-5b.exe" "path\to\videos list file.txt"


"""





import cv2
#from PyQt5 import QtWidgets,QtCore,QtGui
import csv
#import numpy as np


#--------------------------------------------------------------------------------------------------------------------PLAYER
import time
from threading import Thread

# import the Queue class from Python 3
import sys
if sys.version_info >= (3, 0):
	from queue import Queue
 
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue
    


class Video:
    """classe rassemblant les différents paramètres video
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
        self.firstFrame=None
        self.LUT=None
        self.playing=False
        self.currentFrame=None
        
class StimAnalysis:
    def __init__(self):
        self.optoStimArea=[]
        self.optoStimFramesList=[]
        
class FileVideoStream_ToPlay:
    """
    lecture de la vidéo frame par frame et stockage des frames dans un buffer
    """
    def __init__(self, video, queueSize=300):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        
        self.stream = video
        self.stopped = False
         
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self,nframe):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        self.stream.set(1,nframe)
        t.start()
        return self

    def update(self):
        #boucler jusqu'à la fin de la video get(7)=nbre de frames cv2 videocapture
        endFrame=self.stream.get(7)
        #i=0
        #while True:
        while self.stream.get(1)<endFrame: 
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                #convertedFrame=convert_image(frame)
                i=self.stream.get(1)
                #print("load ",i,grabbed)
              
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.Q.put([i,frame])
                
#                now = ptime.time()
#                dt = now - varM.lastTime_live
#                varM.lastTime_live = now
#                if varM.measuredLivefps is None:
#                    varM.measuredLivefps = 1.0/dt
#                else:
#                    s = np.clip(dt*3., 0, 1)
#                    varM.measuredLivefps = varM.measuredLivefps * (1-s) + (1.0/dt) * s
#                #print("load time: ",dt)
               
            else:
                time.sleep(2.0)

    def read(self):
        # return next frame in the queue
        return self.Q.get()
        
    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0    
        
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True     





def playpause_video(videoPath):
    
    if video.playing==False:
        
        video.playing=True
        video.grabber.stopped = False
        
        #vider le buffer
        #video.stream.task_done()
        #video.grabber.Q.queue.clear()
                     
        video.grabber.start(0)#------------------------thread de chargement des images dans le buffer à partir du numéro de frame passé en argument
        print("grabber started")
        time.sleep(1.0)
        video_play_stream(video.grabber)#---------------------------------------boucle de lecture des images
        #lorsque la boucle est interrompue :
        print("streamplayer stopped")
        pause_video()
        
    else :
        pause_video()
                
def pause_video():
    video.playing=False
    video.grabber.stop()
    video.grabber.Q.queue.clear()
    #ui.progressBuffer.setValue(video.grabber.Q.qsize()/video.grabber.Q.maxsize*100)
    
    #effacer les zones tracées dans viewBoxMap et redéfinir la map
    #init_viewBox_map()    
    #update_fullPlot()
    
    

def video_play_stream(fvs):
    
    """
    lecture des frames accumulés dans le tampon file video stream
    """
    
    #if ui.track_checkBox.isChecked()==True :
    #    QtCore.QTimer.singleShot(50, lambda: update_plot())
    #QtCore.QTimer.singleShot(50, lambda: update_analysis())
    # loop over frames from the video file stream
    lightON=False
    stopAnalysis=0
    
    for i in range(video.nbFrames) :
    
        if fvs.more():
            stopAnalysis=0
            
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale (while still retaining 3
            # channels)
            #condition d'arrêt
            if video.playing==False:
                break
            #prendre une image dans le buffer
            frameplus=fvs.read()
            nframe=frameplus[0]
            #print("played frame : ", nframe)
            video.currentFrame=frameplus[1]
            #ui.progressBuffer.setValue(fvs.Q.qsize()/fvs.Q.maxsize*100)
            """                    
            # display the size of the queue on the frame
            cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
            	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
            """ 
            # convert frame
            convertedFrame=convert_image(video.currentFrame)
            #!!! redimmensionne l'image 1/2
    #        # show the frame 
    #        ui.img.setImage(tFrame,autoLevels=False)
    #        ui.timeLine_slider.setValue(ui.timeLine_slider.value()+1)
    
    #        cv2.imshow('im',video.currentFrame)
    #        key = cv2.waitKey(20)
    #        #if key!=-1 : print(key)
    #        if key == 27:  # exit on ESC
    #            break
                
            #-----------------------------------------------------------------------------------------------ANALYSIS
                                          
            lightON=analysis(nframe,convertedFrame,lightON)
                              
            #---------------------------------------------------------------------------------------------------        
    #        #calcul du temps d'analyse
    #        now = ptime.time()
    #        dt = now - varM.lastTime_play
    #        varM.lastTime_play = now
    #        if varM.measuredPlayfps is None:
    #            varM.measuredPlayfps = 1.0/dt
    #        else:
    #            s = np.clip(dt*3., 0, 1)
    #            varM.measuredPlayfps = varM.measuredPlayfps * (1-s) + (1.0/dt) * s
    #        #print("play : ",dt)
    #        #avec un framerate fps pour le maintenir on ajoute un timer pour ralentir l'affichage si besoin
    #        fpsTime=1/video.grabFrameRate
    #        timeToSleep=fpsTime-dt
    #        if timeToSleep>0:
    #            time.sleep(timeToSleep)
    #            #print(timeToSleep)                   
            #ui.measuredPlayfps_Label.setText('%0.2f fps' % varM.measuredPlayfps)
            #-------------------------------------------------------------------------------------------------------
    #        app.processEvents()  ## force complete redraw for every plot
    
        else :
            if stopAnalysis==0 :
                print("fvs.more=",fvs.more())
                stopAnalysis=1
                time.sleep(1.0)
            else :
                break
                
            
    cv2.destroyWindow('im')
        
#------------------------------ ---------------------------------------------------------------------------------------------END PLAYER        


def convert_image(im):
    """
    redimentionner l'image pour accélérer l'analyse
    """
    #w,h=im.shape
    #print("image size : ",im.shape)
    scaledim=cv2.resize(im, None, fx = 0.5, fy = 0.5,interpolation = cv2.INTER_CUBIC)
        
    return scaledim



def create_Result(file,path,fps,width,height):
   
    #TODO : test si le fichier est en cours d'utilisation
    #création d'un fichier résultat dont l'entête est spécifique à une manip
       
    metadatas=[[path],
                   ["framerate : "+str(fps)+' fps'],
                   ["video size : "+str(width)+" ; "+str(height)],
                   ["scale : per pixel"]
                   ]
    
    #if ui.manipType_comboBox.currentText()=="opto stim":
        
    csvResult=csv.writer(file,delimiter=',', lineterminator='\n')
    #TODO :utilisation d'un context manager pour gérer la fermeture du fichier en cas d'exception
    #with csv.writer(open(dialog1.lb_VideoName.text()+"_result.csv","wb")) as csvResult :
    
    csvResult.writerows(metadatas)
    print(metadatas)
    #step_Results.write(','.join(iheader for iheader in header_List))
    #step_Results.write('\n')
    header_List=["ON","OFF"]
    csvResult.writerow([*header_List]) #basically the * unpacks the list as arguments for the function
    
    return csvResult
    
   
def save_Results(result_f):
    
    #if ui.manipType_comboBox.currentText()=="opto stim":
                
        #récupérer les valeurs des positions pour chaque index dans chaque liste
#        for t in optoStimFramesList:
#            result_f.writerow([*t])
    liste=iter(stimAnalysis.optoStimFramesList)
    for on, off in zip(liste,liste):
        result_f.writerow([on[0],off[0]])
    #réinitialiser les listes :
    stimAnalysis.optoStimArea=[]
    stimAnalysis.optoStimFramesList=[]



    
def analysis(nframe,im,lightON):

    blueLower = (150, 50, 100)
    blueUpper = (190, 255, 255)
    greenLower = (50, 80, 60)
    greenUpper = (150, 255, 255)
    
    #lightON=False
             
    
    print(nframe)
    blurred = cv2.GaussianBlur(im, (3, 3), 0)
    #conversion HSV et seuillage couleur
    #hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    im_thresh = cv2.inRange(hsv, greenLower, greenUpper)
    #mask = cv2.erode(mask, None, iterations=1)
    #mask = cv2.dilate(mask, None, iterations=1)
    #cv2.morphologyEx(mask, cv2.MORPH_OPEN,analyze.kernel)
    
    #dénombrement des objets seuillés et mesure de leur aire
    (cnts,_) = cv2.findContours(im_thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(cnts)>0:
        #classement des contours par taille décroissante, ne conserver que le premier
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
        cntArea=cv2.contourArea(cnt[0])
        #M = cv2.moments(cnt)
        #area=M['m00']
        print("area : ",cntArea)
        #TODO : la surface ne suffit pas : 
        # x,y,w,h = cv2.boundingRect(cnt)
        # aspect_ratio = float(w)/h
        #enregistrer la liste des valeurs de surface
        stimAnalysis.optoStimArea.append(cntArea)
        if len(stimAnalysis.optoStimArea)>1 : 
            if stimAnalysis.optoStimArea[-1]>stimAnalysis.optoStimArea[-2]*1.5 and stimAnalysis.optoStimArea[-1]>40 and lightON==False:
                #!!! valeur limite modifiée 100==>25 image divisée par 2===>correction 40
                #add_light_event("light ON")
                lightON=True
                print(nframe," - light ON")
                stimAnalysis.optoStimFramesList.append((nframe,"ON"))
            elif stimAnalysis.optoStimArea[-1]<stimAnalysis.optoStimArea[-2]/1.5 and stimAnalysis.optoStimArea[-1]<40 and lightON==True: 
                #add_light_event("light OFF")
                lightON=False
                print(nframe," - light OFF")
                stimAnalysis.optoStimFramesList.append((nframe,"OFF"))
    else:
        if lightON==True :
            lightON=False
            print(nframe," - light OFF")
            stimAnalysis.optoStimFramesList.append((nframe,"OFF"))
    
    return lightON
    
    
def list_analyze(listPath):
    list_vidéos = open(listPath, "r")
    
    #parcourir une liste
    for line in list_vidéos :
        #nettoyer la ligne extraite des caractères invisibles
        videoPath=line.rstrip()
        video.capture = cv2.VideoCapture(videoPath)
        
        #intancier le FileVideoStream_ToPlay pour la mise en buffer
        video.grabber = FileVideoStream_ToPlay(video.capture)
        
        video.width = video.capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        video.height = video.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        video.fps = video.capture.get(cv2.CAP_PROP_FPS)
        video.nbFrames=int(video.capture.get(7)) #nombre de frames
        print("nb frames : ",video.nbFrames)
        
        # lancer le lecteur video
        playpause_video(videoPath)
        
             
        print("end video")
       
        resultFilePath = videoPath[:-4]+"stim2.csv"
        print("resultFile : ",resultFilePath)
        with open(resultFilePath, 'w') as resultfile:
            resultcsv=create_Result(resultfile,resultFilePath,video.fps,video.width,video.height)
            #resultFile=create_Result()
            #print("resultFile : ",resultFile)
            save_Results(resultcsv)
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
       
    stimAnalysis=StimAnalysis()
    video=Video()
#    list_analyze(r"D:\Travaux\greg\Asier\TESTS-optostim\listOptodetect-test118-3.txt")
#    print(sys.argv[1:])
#    list_analyze(sys.argv[1:][0])
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("listVideos", help="list of videos to analyze")
    args = parser.parse_args()
    print("liste des vidéos à analyser =", args.listVideos)
    list_analyze(args.listVideos)
            
    
   