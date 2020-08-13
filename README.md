# Behavioral Analysis


This repository is dedicated to the data analysis of behavioral experiments. These experiments include goal-directed lever-reaching tasks in rats as well as mice engaged in locomotion tasks during different experimental conditions such as *brain lesion* or *optogenetic manipulations*. 

The data for this project is provided through different pipelines. Temporal position of body parts are tracked using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), a package for markerless pose estimation of user-defined features with deep learning for all animals. Whereas the information provided by the experiment chamber itself is processed with this interactive [code](https://github.com/Shiva-A-Lindi/Behavioral-Analysis/LED_detection) developed using CV2 Python package. The step-by-step guide is provided [here]().


## Body-part tracking with DLC:


<p align="left">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Locomotion_DLC_and_plot.gif width="50%" height="50%">
</p>


<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Forpaw.png width="60%" height="60%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Lever_reaching_position.png width="30%" height="30%">

  <p align="center">This is a centered caption for the image<p align="center">
</p>


<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Locomotion_position.png width="40%" height="40%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Lever_reaching_velocity.png width="50%" height="50%">

  <p align="center">This is a centered caption for the image<p align="center">
</p>
### mis-tracking handling:

Even when optimizing the trained network it so happens that there are mistakes in the trackings of body parts. It would not be as important in an experiment where you need the information about the average position of the animal (e.g. here during the locomotion task), but it would be a cause for gross error if you want to calculate the tortuosity and traveled distance of the animal's hand during a lever reaching task. Therefore, I implemented algorithms to pick up these jumps and correct them.


<p align="center">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/mis_tracking_locomotion.png width="40%" height="40%">
  <img alt="img-name" src=https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/mis_tracking_lever_reaching.png width="40%" height="40%">
  <p align="center">This is a centered caption for the image<p align="center">
</p>
