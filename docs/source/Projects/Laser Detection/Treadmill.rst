Treadmill
~~~~~~~~~


This pipline is developed to detect laser pulses received at the optic fibers on the animal's head when moving on a treadmill. 

It works in two separate ways:

#. You provide **Spike2** analog signals (with extensions ``.smr`` or ``.smrx``) of the session (understandbly with an unknow offset with the video recording).
	This file should include the same trial tag with a letter following a 2 digit number e.g. ``a01``, ``f25``, etc. Which is used to identify the video file in the **Video** folder to it's corresponding analog signal file under **Laser** folder. 
	Needless to say that providing the analog pulse yields a  much more accurate reading of the pulses.

#. If you don't provide the analog signals, the laser pulses are detected soley based on the video recording.


	.. note::
		You need to provide the information about the stimulation (pulse) duration in the config file.

