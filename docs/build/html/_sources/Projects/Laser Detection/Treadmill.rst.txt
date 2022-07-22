Treadmill
~~~~~~~~~


This pipline is developed to detect laser pulses received at the optic fibers on the animal's head when moving on a treadmill (as shown below).

.. figure:: /media/Laser_detection_loco_frame.png
  	:scale: 30 %
  	:alt: alternate text
  	:align: center


It works in two separate ways:

#. Either you provide **Spike2** analog signals (with extensions ``.smr`` or ``.smrx``) of the session (understandably with an unknown offset with respect to the video recording).

	This file should include the same trial tag with a letter following a 2-digit number e.g. ``a01``, ``f25``, etc. Which is used to identify the video file in the **Video** folder to its corresponding analog signal file under **Laser** folder. 
	Needless to say that providing the analog pulse will always result in a much more accurate reading of the pulses.

#. If you don't provide the analog signals, the laser pulses are detected solely based on the video recording.

	.. note::
		You need to provide the information about the stimulation (pulse) duration in the config file.

Here's an example of the beta pulse protocol:

.. image:: /media/Laser_detection_times_series_beta.jpg
	:width: 65%

.. image:: /media/Laser_detection_beta_pulse.jpg
	:width: 32%