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
        self.DLC_filepath_list = self.filter_paths_based_on_existing_str( 'DLC')
        self.smr_filepath_list = self.get_smr_files(extensions = ['.smrx', '.smr', '.s2rx'])
        
    def build_filePath_list(self):
        
        '''go over the directory tree in path and make a list of all files and extensions'''
        
        self.all_filepath_list = []
        extensions = []
        
        for (dirpath, dirnames, filenames) in os.walk(self.path):
        
            for f in filenames:

                self.all_filepath_list.append(os.path.join(dirpath, f))
                extensions.append( os.path.splitext(f) [1])
                
        self.extensions = np.unique(extensions)
        
    def get_smr_files(self, extensions = ['.smrx', '.smr', '.s2rx']):
        
        return [ fi for fi in self.all_filepath_list 
                if (fi.endswith( tuple(extensions) ))]
        
    def build_filepath_dict(self):
        
        for ext in self.extensions:
            
            self.filepath_list[ext] = [ fi for fi in self.all_filepath_list 
                                             if (fi.endswith( ext ))]

    def filter_paths_based_on_existing_str(self, substr):
        
        return  [ fi for fi in self.all_filepath_list 
                 if if_substr_in_str( substr, fi)]
                
    def refine_file_names(self, string = 'DLC'):
        
        for path in self.all_filepath_list:
            
            file = File(path)
            file.check_filename_type(no_specifier = '#')
            file.remove_spaces_in_name()
            file.replace_txt('.', '-')
            file.replace_txt('alone', '')
            file.replace_txt('only', '')
            file.enforce_underscore_bef_certain_string(string = string)
            file.rename(file.name)
            
        self. __init__( self.path)
       
    def add_stim_type_to_name(self, stim_type_list = ['Square', 'square', 'beta', 'Beta'], 
                              extensions = ['.avi', '.h5', '.pickle', '.csv']):
        
        for ext, f_list in self.filepath_list.items() :
            
            if ext in extensions:
                
                for f in f_list:
                    
                    file = File(f)
                    
                    if not if_any_substr_in_str( stim_type_list, file.name_elements[1]):
                    
                        file.add_to_name_elements('SquarePulse', 1)
                        
        self. __init__( self.path)   
        
    @staticmethod
    def enforce_dirname_if_already_exists(path, string_in_name = 'SquarePulse', 
                                          change_to = 'squarepulse'):
        
        for (dirpath, dirnames, filenames) in os.walk(path):
        
            for dirname in dirnames:
                
                if string_in_name in dirname:
                    
                    src =  os.path.join(dirpath, dirname)
                    dest = os.path.join(dirpath, dirname.replace(string_in_name, change_to))
                    print('renaming..', src, 'to', dest)
                    os.rename(src, dest)
    
    @staticmethod
    def create_dir_if_not_exist( directory):
        
        if not os.path.exists(directory):
            
            os.makedirs( directory)
            
    
class File :
    
    def __init__(self, path):
        
        self.path = path ## static doesn't change
        self.dirpath, self.name =  os.path.split(path)
        self.name_elements = None
        self.get_extension()
        self.get_name_elemets()
        
    def get_extension(self):
        
        self.name_base, self.extension = os.path.splitext( self.name )
        
    def enforce_underscore_bef_certain_string(self, string = 'DLC'):
        
        if string in self.name_base:
            
            char_bef_string = self.name_base [ self.name_base.index( string) - 1]
            
            if char_bef_string != '_':
                
                
                self.name = self.name.split( string )[0] + '_' + string + \
                            self.name.split( string )[1]
                
                
    def replace_txt_and_rename( self, text, with_text):
        
        new_name = self.name.replace(text, with_text)
                        
        self.rename(new_name)
        
    def rename(self, new_name):
        
        new_path =  os.path.join( self.dirpath, new_name) 
        os.rename( self.path, 
                   new_path )
        # print(self.path, 
        #            new_path)
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
        
    def check_filename_type(self, no_specifier = '#'):
        
        if self.extension == '.avi':
            mouse =  self.name_base.split('_') [0]
            
            if no_specifier not in mouse:
                
                print(self.path)
                raise("Attention file name is not stereosypical!")
    
                
    def move_file_if_not_already_in_dest(self, new_filepath):
        
        if not os.path.exists(new_filepath):
            
            os.rename( self.path, new_filepath)    
            
        else:
            
            print(new_filepath, 'already moved')
        
class Experiment():
    
    def __init__(self, video_filepath):
        
        self.stim_type = None
        self.stim_power = None
        self.stim_location = None
        self.day_tag = None
        self.mouse_no = None
        self.mouse_line = None
        self.video = File(video_filepath)
        self.DLC_files = []
        self.smr_files = []
        self.laser = None
        self.treadmil_velocity = None
                

            
    def get_mouse_no_and_line(self, no_specifier = '#'):
        
        mouse =  self.video.name_base.split('_') [0]

        self.mouse_no = mouse.split( no_specifier ) [1]
        self.mouse_line = mouse.split( no_specifier ) [0]
        
    def correct_stim_loc(self):
        
        if self.stim_location == 'STR-STN':
            self.stim_location = 'STN-STR'
    
    def get_day_tag(self):
        
        if self.video.name_elements[-1].lower() == 'stacked' :
            
            if len(self.video.name_elements[-2]) != 3 :
                
                raise ValueError ('day tag is not streorypical')
                
            else:
                
                self.day_tag = self.video.name_elements[-2]
                
        elif len(self.video.name_elements[-1]) != 3 :
            
            raise ValueError ('day tag is not streorypical')
            
        else:
            
            self.day_tag = self.video.name_elements[-1]
            
            
    def extract_info_from_video_filename(self) :
        
        self.get_mouse_no_and_line( no_specifier = '#')

        self.stim_type = self.video.name_elements[1]
        self.stim_location = self.video.name_elements[2].replace('+', '-')
        self.correct_stim_loc()
        self.stim_power = self.video.name_elements[3]
        self.treadmil_velocity = self.video.name_elements[4]
        self.get_day_tag()
        
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
        
    def find_matching_smr_file(self, smr_filepath_list):
        
        for f in smr_filepath_list:
            
            file = File(f)
            
            if self.day_tag in file.name_elements:
                
                self.smr_files.append(file)
                
                if file.extension == '.smr' or file.extension == '.smrx':
                    print('SMR = ', file.name)
                
    def find_matching_DLC_files(self, DLC_filepath_list):
        
        for f in DLC_filepath_list:
            
            file = File(f)
            
            if self.day_tag in file.name_elements:
                
                self.DLC_files.append(file)
                
                if file.extension == '.csv':
                    print('DLC = ', file.name)
       
    def move_video_file(self, path_exp):

        new_dir = os.path.join( path_exp, 'Video')
        
        Directory.create_dir_if_not_exist( new_dir )

        new_filepath = os.path.join( new_dir, self.video.name)
        
        self.video.move_file_if_not_already_in_dest(new_filepath)
        
    def move_smr_files(self, path_exp):

        new_dir = os.path.join( path_exp, 'Laser')
        
        Directory.create_dir_if_not_exist( new_dir )
            
        for file in self.smr_files:
            
            new_filepath = os.path.join( new_dir, file.name)
            
            file.move_file_if_not_already_in_dest(new_filepath)
                
    def move_DLC_files(self, path_exp):

        new_dir = os.path.join( path_exp, 'DLC')
        
        Directory.create_dir_if_not_exist( new_dir )
            
        for file in self.DLC_files:
            
            new_filepath = os.path.join( new_dir, file.name)
            
            file.move_file_if_not_already_in_dest(new_filepath)
                
    def move_files_into_hirerarchy(self, new_path = ''):


        path_exp = os.path.join ( new_path,
                                  self.mouse_line, 
                                  'Mouse_' + self.mouse_no, 
                                  self.stim_location,
                                  self.stim_type.lower() + '_' + 
                                  self.stim_power) 
        
        Directory.create_dir_if_not_exist( path_exp)
        
        self.move_video_file(path_exp)
        self.move_smr_files(path_exp)
        self.move_DLC_files(path_exp)
        
         
        
def if_substr_in_str( substr, string):
    
    return re.search( substr, string, re.IGNORECASE)

def if_any_substr_in_str( substr_list, string):
    
    return any(map(string.__contains__, substr_list))


root = '/media/shiva/LaCie/Data_INCIA_Shiva'

path_list = [os.path.join( root, name) for name in os.listdir(root) if os.path.isdir(os.path.join( root, name)) and name.startswith('2021')]

path = path_list[5]
print( os.path.basename(path))



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
    exp.move_files_into_hirerarchy(new_path = '/media/shiva/LaCie/Data_INCIA_Shiva_sorted')
        


  

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    