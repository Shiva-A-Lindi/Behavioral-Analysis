# Behavioral-Analysis

This repository is dedicated to data analysis of behavioral experiments. So far these experiments include rats performing a **goal-directed lever-reaching task** and mice **running on a treadmill** during different experimental conditions such as *brain lesions* and *optogenetic manipulation*.

The data for this analysis are derived through different methods. The pose estimation of animal body parts are obtained using the [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), a package for Markerless pose estimation of user-defined features with deep learning for all animals. On the other hand, the information provided by the experiment chamber through LED lights is analyzed through an interactive code available [here](https://github.com/Shiva-A-Lindi/Behavioral-Analysis/tree/master/LED_detection) with a step-by-step user guide [here]().

![landmarks extracted with DeepLabCut](https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Locomotion_DLC.gif?raw=true)

![Locomotion DLC average plot](https://github.com/Shiva-A-Lindi/Behavioral-Analysis/blob/media/Locomotion_plot.gif?raw=true)