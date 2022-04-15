#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:53:24 2022

@author: shiva
"""

import sys
import os
import re
import numpy as np

class Directory :
    
    def __init__(self, path):
        
        self.path = path
        self.all_filepath_list = None
        self.filepath_list = {}
        self.extensions = None
        self.build_filePath_list()
        self.build_filepath_dict()
        self.DLC_files = self.filter_paths_based_on_existing_str( 'DLC')

    def build_filePath_list(self):
        
        '''go over the directory tree in path and find all files with extensions included in <extensions>'''
        
        self.all_filepath_list = []
        extensions = []
        
        for (dirpath, dirnames, filenames) in os.walk(self.path):
        
            for f in filenames:

                self.all_filepath_list.append(os.path.join(dirpath, f))
                extensions.append( os.path.splitext(f) [1])
                
        self.extensions = np.unique(extensions)
        
    def build_filepath_dict(self):
        
        for ext in self.extensions:
            
            self.filepath_list[ext] = [ fi for fi in self.all_filepath_list 
                                             if (fi.endswith( ext ))]

    def filter_paths_based_on_existing_str(self, substr):
        
        return  [ fi for fi in self.all_filepath_list 
                 if if_substr_in_str( substr, fi)]
                
    def refine_file_names(self):
        
        for path in self.all_filepath_list:
            
            file = File(path)
            file.remove_spaces_in_name()
            file.replace_txt('.', '-')
            file.replace_txt('alone', '')
            file.replace_txt('only', '')
            file.rename(file.name)
            
    def add_stim_type_to_name(self, stim_type_list = ['Square', 'square', 'beta', 'Beta'], 
                              extensions = ['.avi', '.h5', '.pickle', '.csv']):
        
        for ext, f_list in self.filepath_list.items() :
            if ext in extensions:
                for f in f_list:
                    
                    file = File(f)
                    
                    print(file.name)
                    if not if_any_substr_in_str( stim_type_list, file.name_elements[1]):
                    
                        file.add_to_name_elements('SquarePulse', 1)
            
        
class File :
    
    def __init__(self, path):
        
        self.path = path ## static doesn't change
        self.dirpath, self.name =  os.path.split(path)
        self.name_elements = None
        self.get_extension()
        self.get_name_elemets()
        
    def get_extension(self):
        
        self.name_base, self.extension = os.path.splitext( self.name )
        
    def enforce_underscore_bef_DLC_in_name(self):
        
        if 'DLC' in self.name_base:
            
            char_bef_DLC = self.name_base [ self.name_base.index( 'DLC' ) - 1]
            
            if char_bef_DLC != '_':
                
                new_name = self.name.split( 'DLC' )[0] + '_DLC' + \
                            self.name.split( 'DLC' )[1]
                                
                self.rename(new_name)
         
    def replace_txt_and_rename( self, text, with_text):
        
        new_name = self.name.replace(text, with_text)
                        
        self.rename(new_name)
        
    def rename(self, new_name):
        
        new_path =  os.path.join( self.dirpath, new_name) 
        os.rename( self.path, 
                   new_path )
        self.__init__(new_path) ## updates everything
        
    def build_name(self):
        self.name = self.name_base + self.extension

    def build_path(self):
        self.path = os.path.join(self.dirpath, self.name)

    def replace_txt(self, text, with_text):
        
        self.name_base = self.name_base.replace(text, with_text) 
        self.build_name()
        
    def remove_spaces_in_name(self):
        
        self.name_base = self.name_base.replace(' ', '') 
        self.build_name()
    
    def get_name_elemets(self):
        
        self.name_elements = self.name_base.split('_')
    
    def add_to_name_elements(self, txt_to_add, element_ind):
        
        new_name = ( '_'.join(self.name_elements[:element_ind] ) + 
                    '_' + 
                    txt_to_add + 
                    '_' +
                    '_'.join(self.name_elements[element_ind:]) )
        
        self.rename(new_name + self.extension)
        
class Experiment():
    
    def __init__(self, video_filepath):
        
        self.stim_type = None
        self.stim_power = None
        self.stim_location = None
        self.day_tag = None
        self.mouse_no = None
        self.mouse_line = None
        self.video = File(video_filepath)
        self.DLC = None
        self.smr = None
        self.laser = None
        self.treadmil_velocity = None
                
    def get_mouse_no_and_line(self, no_specifier = '#'):
        
        mouse =  self.video.name_base.split('_') [0]
        self.mouse_no = mouse.split( no_specifier ) [1]
        self.mouse_line = mouse.split( no_specifier ) [0]
        

    def extract_info_from_video_filename(self) :
        
        self.get_mouse_no_and_line( no_specifier = '#')

        self.stim_type = self.video.name_elements[1]
        self.stim_location = self.video.name_elements[2]  
        self.stim_power = self.video.name_elements[3]
        self.treadmil_velocity = self.video.name_elements[4]
        self.day_tag = self.video.name_elements[-1]
        
    def show_specs(self):
        
        print( 
              'video file :', self.video.name, '\n',
              'mouse no : ', self.mouse_no, '\n',
              'mouse line :', self.mouse_line, '\n',
              'stim type : ', self.stim_type, '\n',
              'stim_power : ', self.stim_power, '\n',
              'location : ', self.stim_location, '\n',
              'treadmil velocity :', self.treadmil_velocity, '\n',
              'day tag :', self.day_tag
              )
        
def if_substr_in_str( substr, string):
    
    return re.search( substr, string, re.IGNORECASE)

def if_any_substr_in_str( substr_list, string):
    
    return any(map(string.__contains__, substr_list))





path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted/Mouse_71/STN-STR'
stim_type_list = ['Square', 'Beta', 'beta', 'square']
directory = Directory(path)
print('file extensions in this directory: ', directory.extensions)

directory.refine_file_names()
directory.add_stim_type_to_name(stim_type_list = stim_type_list,
                                extensions = ['.avi', '.h5', '.pickle', '.csv'])

directory = Directory(path)
videos = directory.filepath_list['.avi']

for f in videos:
    
    exp = Experiment(f)
    exp.extract_info_from_video_filename()
    exp.show_specs()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    