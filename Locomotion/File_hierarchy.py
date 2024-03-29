#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:00:55 2022

@author: shiva
"""

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
import shutil

class Directory :
    
    def __init__(self, path):
        
        self.path = path
        self.all_filepath_list = None
        self.filepath_list = {}
        self.extensions = None
        self.filenames = []
        self.build_filePath_list()
        self.build_filepath_dict()
        self.DLC_filepath_list = self.filter_paths_based_on_existing_str( 'DLC')
        self.smr_filepath_list = self.get_spec_files(extensions = ['.smrx', '.smr', '.s2rx'])
        
    def build_filePath_list(self):
        
        '''go over the directory tree in path and make a list of all files and extensions'''
        
        self.all_filepath_list = []
        extensions = []
        
        for (dirpath, dirnames, filenames) in os.walk(self.path):
        
            for f in filenames:
                
                if not f.startswith('.'):
                    self.all_filepath_list.append(os.path.join(dirpath, f))
                    extensions.append( os.path.splitext(f) [1])
                    
        self.extensions = np.unique(extensions)
        
    def get_filenames(self):
        
        self.filenames = [os.path.basename(f) for f in self.all_filepath_list]
        
    def get_spec_files(self, extensions = ['.smrx', '.smr', '.s2rx']):
        
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
       
    def remove_spaces_dots(self):
        
        for path in self.all_filepath_list:
            
            file = File(path)
            file.remove_spaces_in_name()
            file.replace_txt('.', '-')
            print(file.name)
            file.rename(file.name)
            
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
            
    def remove_files(filepath_list):
        
        for file in filepath_list:
            os.remove(file)

    @staticmethod
    def rename_folder(path, original_name, new_name):
        
        for (dirpath, dirnames, filenames) in os.walk(path):
        
            for d in dirnames:

                if d == original_name:
                    os.rename(os.path.join(dirpath, original_name),
                              os.path.join(dirpath, new_name))
                    
    @staticmethod      
    def create_folder_with_exising_folders(current_path, folder = 'STR'):
        
        for child in os.scandir(current_path):

            if child.is_dir():
                
                new_folder = os.path.join(child.path, folder)
                scanned = [i.path for i in os.scandir(child.path)]
                Directory.create_dir_if_not_exist(new_folder)

                for grandchild in scanned: 
                    
                    print(scanned, 'moving', grandchild, 'to',  os.path.join(new_folder,
                                  os.path.basename(grandchild)))
                    shutil.move(grandchild, 
                                os.path.join(new_folder,
                                             os.path.basename(grandchild))
                                )



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
        
    @staticmethod
    def rm_if_exist(parent_folder, filepath_to_del):
        
        d = Directory(parent_folder)
        
        for filepath in d.all_filepath_list:
            
            if os.path.basename(filepath) == filepath_to_del:

                try: 
                    os.remove(filepath)
    
                except OSError as e: # name the Exception `e`
                    print ("Failed with:", e) # look what it says
                    
                break
            
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
        self.treadmill_velocity = None
        self.file_found = {'smr' : False, 'DLC': False}

            
    def get_mouse_no_and_line(self, no_specifier = '#'):
        
        mouse =  self.video.name_base.split('_') [0]

        self.mouse_no = mouse.split( no_specifier ) [1]
        self.mouse_line = mouse.split( no_specifier ) [0]

    
    def get_day_tag(self):
        
        if self.video.name_elements[-1].lower() == 'stacked' : # if "Stacked" is added to the end of the filename
            
            if len(self.video.name_elements[-2]) != 3 :
                
                raise ValueError (self.video.name_elements[-2], 'detected as day tag which is not streorypical')
                
            else:
                
                self.day_tag = self.video.name_elements[-2]
                
        elif len(self.video.name_elements[-1]) != 3 :
            
            raise ValueError (self.video.name_elements[-1], 'detected as day tag which is not streorypical')
            
        else:
            
            self.day_tag = self.video.name_elements[-1]

    def get_treadmill_velocity(self, default_velocity = 15):
        
        velocity = [string for string in self.video.name_elements if 'cm-s' in string]
        
        if any(velocity):
            
            self.treadmill_velocity = int(velocity[0].replace('cm-s', ''))
            
        else: # if velocity not mentioned in name it must be the default of 15 cm/s
            
            self.treadmill_velocity = default_velocity

    def extract_info_from_video_filename(self) :
        
        self.get_mouse_no_and_line( no_specifier = '#')

        self.stim_type = self.video.name_elements[1]
        self.stim_location = self.video.name_elements[2].replace('+', '-')
        self.stim_power = self.video.name_elements[3]
        self.get_day_tag()
        self.get_treadmill_velocity()
        
    def show_specs(self):
        
        print( 
              'video file :', self.video.name, '\n',
              'mouse no : ', self.mouse_no, '\n',
              'mouse line :', self.mouse_line, '\n',
              'stim type : ', self.stim_type, '\n',
              'stim_power : ', self.stim_power, '\n',
              'location : ', self.stim_location, '\n',
              'treadmil velocity :', self.treadmill_velocity, '\n',
              'day tag :', self.day_tag
              )
        
    def find_matching_smr_file(self, smr_filepath_list):
        
        file_found = False
        
        for f in smr_filepath_list:
            
            file = File(f)
            
            if self.day_tag in file.name_elements:
                
                self.smr_files.append(file)
                
                if file.extension == '.smr' or file.extension == '.smrx':
                    print('SMR = ', file.name)
                    file_found = True
                    self.file_found['smr'] = True
                    
        Experiment.report_file_not_found(self.file_found['smr'], 'smr')
        
    def find_matching_DLC_files(self, DLC_filepath_list):
        
        

        for f in DLC_filepath_list:
            
            file = File(f)
            
            if self.day_tag in file.name_elements:
                
                self.DLC_files.append(file)
                
                if file.extension == '.csv' and 'modif' not in file.name_base.split('_')[-1]:
                    
                    print('DLC = ', file.name)
                    self.file_found['DLC'] = True
                    
        Experiment.report_file_not_found(self.file_found['DLC'], 'DLC')
       
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
                
    def move_files_into_hirerarchy(self, new_path = '', move_if_not_all_files_found = False):


        path_exp = os.path.join ( new_path, 
                                  'Mouse_' + self.mouse_no, 
                                  self.stim_location,
                                  self.stim_type.lower() + '_' + 
                                  self.stim_power) 
        
        Directory.create_dir_if_not_exist( path_exp)
        
        if sum(list(self.file_found.values())) == 2 or move_if_not_all_files_found:
            
            self.move_video_file(path_exp)
            self.move_smr_files(path_exp)
            self.move_DLC_files(path_exp)
    
    @staticmethod
    def report_file_not_found(file_found, dtype):
        
        if not file_found:
            
            print('Matching {} not found!'.format(dtype))
         
        
def if_substr_in_str( substr, string):
    
    return re.search( substr, string, re.IGNORECASE)

def if_any_substr_in_str( substr_list, string):
    
    return any(map(string.__contains__, substr_list))
