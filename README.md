# Behavioral Analysis


This repository is dedicated to the analysis of animal behavioral experiments data including goal-directed lever-reaching tasks in rats, and mice engaged in locomotion tasks during different experimental conditions such as *brain lesion* or *optogenetic manipulations*<sup id="a1">[1](#f1)</sup>. 

The metadata for this project is provided through different pipelines, temporal position of body parts are tracked using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)<sup id="a2">[2](#f2)</sup>, a package for markerless pose estimation of user-defined features with deep learning for all animals. Whereas the information provided by the experiment chamber itself is processed with a stand-alone code developed using [opencv-python](https://pypi.org/project/opencv-python/) package. The step-by-step guide for different aspects of this project is provided in the [documentation](https://shiva-a-lindi.github.io/Behavioral-Analysis/build/html/index.html).


## Body-part tracking with DLC


<p align="left">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/gifs/Locomotion_DLC_and_plot.gif width="50%" height="50%" hspace="50">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/gifs/Rat_lever_demo.gif width="22%" height="22%" hspace="40">
  <p align="left">Resulting body-part tracking with DLC. <strong>Left</strong>: Locomotion task. The animation shows the average of nose extremity and tail base as a function of time. <strong>Right</strong>: Lever reaching task. The animation shows the traces of wrist base in time (light blue dot) <sup id="a1">[1]</sup>. <p align="center">

</p>

### Handling Mistracks

Even after the ideal optimization of the network rare instances of mislablings occur. This is not of great importance where the information about the average position of the animal is required, e.g. during the locomotion task, however it leads to significant errors when calculations of the tortuosity and traveled distance of the animal's hand during a lever reaching task. This repository correctly detects and handles such mislablings. For examples of such issues see the figure below.


<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/mis_tracking_locomotion.png width="40%" height="40%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/mis_tracking_lever_reaching.png width="40%" height="40%">
  <p align="left">Mistrackings with DLC pose estimation. <strong>Left</strong>: Individual traces of left and right side cameras. Below: corrected, averaged with a moving window. <strong>Right</strong>: Traces of multiple trials correction shown below.<p align="center">
</p>

### Building metadata and measuring across trials

The data from all sessions each containing multiple trials are then organized and stored to be easilly accessible. In addition, multiple sets of parameters are explored for each experimental protocol to assess the effectiveness of various protocols.

<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Forpaw.png width="60%" height="60%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Lever_reaching_position.png width="30%" height="30%">
</p>

<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Locomotion_position.png width="40%" height="40%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Lever_reaching_velocity.png width="50%" height="50%">
  <p align="center">Examples of behavioral data analysis results <sup id="a1">[1]</sup>. <p align="center">
</p>


## Extracting data from the experiment chamber

<img align="right" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/LED_detection.png width="25%" height="25%" alt="">

In the lever-reaching task the experiment chamber is equipped with an LED panel where LED lights signal particular events in the experiment, for example presence of the animal paw on the pad, cue presentation, or reward delivary. This [script](https://github.com/Shiva-A-Lindi/Behavioral-Analysis/tree/master/LED_detection) targets the extraction of these information from the videos.

## Documentation

The necessary information, code structure, file hierarchy about the developed codes are available in the [documentation](https://shiva-a-lindi.github.io/Behavioral-Analysis/build/html/index.html).

## References

<b id="f1">1</b> Unpublishe data provided by Nicolas Mallet at Physiology and pathophysiology of executive functions Lab (Institut des Maladies Neurodégénératives ([IMN](https://www.imn-bordeaux.org/en/teams/physiology-and-pathophysiology-of-executive-functions/)), CNRS, Bordeaux). [↩](#a1)

<b id="f2">2</b> Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M (2018) DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nature Neuroscience 21:1281–1289. [↩](#a2)
