Locomotion Task Analysis
========================

File hierarchy
--------------

The root directory to your project should have this hierarchy::

	root
	|-- Animal_Type_1 		<-- all data for one specific animal type e.g. 'FoxP2-Cre'
	|	|-- ChR2			<-- Optogenetic expression 
	|	|	|-- Animal_0		<-- data for one animal in that animal type.
	|	|	|	|-- Protocol_0	<-- folders separate protocols e.g. 'Square_1_mW'
	|	|	|	|	|-- DLC <-- contains body part trackings of all sessions derived from DLC
	|	|	|	|	|	|-- session_0_DLC.xlsx 
	|	|	|	|	|	.
	|	|	|	|	|	.
	|	|	|	|	|-- Laser <-- contains laser detection files
	|	|	|	|		|-- session_0_laser.xlsx
	|	|	|	|		|-- session_1_laser.xlsx
	|	|	|	|		.
	|	|	|	|		.
	|	|	|	.
	|	|	|	.
	|	|	|	|-- Protocol_n  
	|	|	|	|	
	|	|	|	`-- Spontaneous <-- Same everything just for Spontaneous sessions
	|	|	|		`DLC <-- only DLC folder since there's no laser
	|	|	|
	|	|	|-- Animal_1 <-- same for all other animals
	|	|	.
	|	|	.
	|	|	|
	|	|	`-- Animal_n
	|	`-- Control
	|		|-- Animal_0
	|		.
	|		`-- Animal_m
	|	
	|-- Animal_Type_1 
	|	
	|
	|-- Animal_Type_n <-- same for all other animal types
	|
	|-- data_npz
	|	|-- Protocol_0
	|	|	|-- ChR2
	|	.	`-- Control
	|	.
	|	|
	|	`-- Protocol_n
	|
	`-- Subplots

Jupyter notebook
----------------

Here_ you will find the jupyter notebook that contains different aspects of Locomotion task data analysis. In the same location you would also find the *Locomotion.py* moodule used for this analysis. The documentation for its functions can be found here_. 
In order to start work on data analysis, you have to first fix some parameters:


.. _Here: https://github.com/Shiva-A-Lindi/Behavioral-Analysis/tree/master/Locomotion
.. _here: https://shiva-a-lindi.github.io/Behavioral-Analysis/build/html/Projects/Locomotion/Code_doc_Locomotion.html
