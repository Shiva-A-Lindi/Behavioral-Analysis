# Behavioral Analysis


This repository is dedicated to the data analysis of behavioral experiments. These experiments include goal-directed lever-reaching tasks in rats as well as mice engaged in locomotion tasks during different experimental conditions such as *brain lesion* or *optogenetic manipulations*<sup id="a1">[1](#f1)</sup>. 

The metadata for this project is provided through different pipelines. Temporal position of body parts are tracked using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)<sup id="a2">[2](#f2)</sup>, a package for markerless pose estimation of user-defined features with deep learning for all animals. Whereas the information provided by the experiment chamber itself is processed with a stand-alone code developed using cv2 Python package. The step-by-step guide for different aspects of this project is provided [here](https://shiva-a-lindi.github.io/Behavioral-Analysis/build/html/index.html).


## Body-part tracking with DLC:


<p align="left">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/gifs/Locomotion_DLC_and_plot.gif width="50%" height="50%" hspace="50">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/gifs/Rat_lever_demo.gif width="22%" height="22%" hspace="40">
  <p align="left">Resulting body-part tracking with DLC. <strong>Left</strong>: Locomotion task. Plot shows the average of nose extremity and tail base. <strong>Right</strong>: Lever reaching task. Plot shows the traces of wrist base (light blue dot) <sup id="a1">[1]</sup>. (<p align="center">

</p>

### Mistracking handling:

Even when optimizing the trained network it so happens that there are mistakes in the trackings of body parts. It would not be as important in an experiment where you need the information about the average position of the animal (e.g. here during the locomotion task), but it would be a cause for gross error if you want to calculate the tortuosity and traveled distance of the animal's hand during a lever reaching task. Therefore, I implemented algorithms to pick up these jumps and correct them.


<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/mis_tracking_locomotion.png width="40%" height="40%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/mis_tracking_lever_reaching.png width="40%" height="40%">
  <p align="left">Mistrackings with DLC pose estimation. <strong>Left</strong>: Individual traces of left and right side cameras. Below: corrected, averaged with a moving window. <strong>Right</strong>: Traces of multiple trials correction shown bellow.<p align="center">
</p>

### Building metadata and measuring across trials:

The data from all sessions with multiple trials each are then organized and stored in an easy to access manner. Different parameters are explored for each experiment protocal.

<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Forpaw.png width="60%" height="60%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Lever_reaching_position.png width="30%" height="30%">
</p>

<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Locomotion_position.png width="40%" height="40%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Lever_reaching_velocity.png width="50%" height="50%">
  <p align="center">Examples of behavioral data analysis results <sup id="a1">[1]</sup>. <p align="center">
</p>


### Extracting data from experiment chamber:

<img align="right" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/LED_detection.png width="25%" height="25%" alt="">

In the lever-reaching task the experiment chamber is equipped with an LED panel with LED lights conveying information about the experiment e.g. whether or not the paw is on the pad, a cue is being presented or a reward is being delivered. [Here](https://github.com/Shiva-A-Lindi/Behavioral-Analysis/tree/master/LED_detection) you can find an Python script that targets the extraction of these information from the videos.

### Documentation:

You can find the necessary information about the developed codes [here](https://shiva-a-lindi.github.io/Behavioral-Analysis/build/html/index.html).

### References:

<b id="f1">1</b> Unpublishe data provided by Nicolas Mallet at Physiology and pathophysiology of executive functions Lab, Institut des Maladies Neurodégénératives (IMN), CNRS, Bordeaux). [↩](#a1)

<b id="f2">2</b> Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M (2018) DeepLabCut: mark- erless pose estimation of user-defined body parts with deep learning. Nature Neuroscience 21:1281–1289. [↩](#a2)
