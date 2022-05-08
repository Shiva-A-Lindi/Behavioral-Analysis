




import sys
import os
import re
import numpy as np
from File_hierarchy import *

move_if_not_all_files_found = True
path =  '/media/shiva/LaCie/Data_INCIA_Shiva/2021_02_19_newTrain2_Video_empty'
directory = Directory(path)

stim_type_list = ['Square', 'Beta', 'beta', 'square']
print('file extensions in this directory: ', directory.extensions)

directory.refine_file_names()
directory.add_stim_type_to_name(stim_type_list = stim_type_list,
                                extensions = ['.avi', '.h5', '.pickle', '.csv'])

videos = directory.filepath_list['.avi']

for f in videos:
    
    exp = Experiment(f)
    exp.extract_info_from_video_filename()
    exp.show_specs()
    exp.find_matching_smr_file( directory.smr_filepath_list)
    exp.find_matching_DLC_files(directory.DLC_filepath_list)
    exp.move_files_into_hirerarchy(new_path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Vglut2D2Cre/ChR2', 
                                   move_if_not_all_files_found = move_if_not_all_files_found)
        


  

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    