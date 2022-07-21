Open Field
~~~~~~~~~~


This pipline is developed to detect laser pulses received at the optic fibers on the animal's head when moving freely in an open field. See example below.

.. figure:: /media/Laser_detection_frame.jpeg
  :scale: 30 %
  :alt: alternate text
  :align: center

It is necessary for to provide the ``stim_duration``, ``inter_stim_interval`` in the config file.
Here's an example for a 30-second ON followed by 30-second OFF protocol.

.. figure:: /media/Laser_detection_time_series_OF.jpg
  :scale: 30 %
  :alt: alternate text
  :align: center

.. figure:: /media/Laser_detection_time_series_OF_pulse.jpg
  :scale: 30 %
  :alt: alternate text
  :align: right

You can see that the 4th pulse is barely captured by the side camera. Therefore, to make consistent detections the detection of these pulses are extrapolated from the solid detection (with durations complying with the ``stim_duration`` in specified in the config file). That is why it is important to input the specifics of the protocol in the config file.

	.. note::
		note that if none of the detected pulses are complient with the protocol but the detected pulse durations have very low variability, they are reported as they are without any correction applied.

