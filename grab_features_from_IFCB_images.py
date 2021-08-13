# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 2015
This script will grab the feature data from extracted feature files 
for all images in an automated class file.
Can bin data by category or leave each image separate.
This particular script was edited to use neural nets to estimate the 
number of cells in a diatom chain or return a 1 (if not a diatom chain) 
for each image.
@author: Darren Henrichs
"""
###old information about this script
# script to extract the biovolume estimates from IFCB V2 feature files
# and sum them per category for class files
# this will read a directory of class files and search the feature path for
# those files, pulling the biovolume from them

# 06/13/2017 DWH
# this script is a modified version of the biovolume grabber script
# this script will take the biovolume value for each cell, convert it to
# units of carbon following formulas from Menden_Deuer and Lessard 2000
# then sum the total carbon per category

####end old information


import os
import pandas as pd
__author__ = 'dhenrichs'

# path to the feature files
feature_paths = ['/home/transcriptome/Documents/darren/synology/IFCB/extracted_features/',
                 '/home/transcriptome/Documents/darren/synology/IFCB/extracted_features_surfside/',
                 '/home/transcriptome/Documents/darren/synology/IFCB/extracted_features_IFCB6/']

# path to the image files
folder_path = '/home/transcriptome/Documents/darren/synology/IFCB/classifiers/Training_sets_21Jun2019/Dinophysis/'

# path to where the outfiles with biovolume will be located
outpath = '/home/transcriptome/Documents/darren/synology/IFCB/'


def grab_features(in_feature, in_img):
    """Open the feature file and obtain the listed features for each image.
    """
    
    feature_data = load_feature_file(in_feature)
    outdata = pd.DataFrame(feature_data, index=feature_data.index, 
                               columns=['Biovolume', 'Area', 'MajorAxisLength', 'MinorAxisLength',
                                        'summedBiovolume', 'summedArea', 'summedMajorAxisLength', 
                                        'summedMinorAxisLength', 'EquivDiameter', 'Extent', 'H90',
                                        'H180', 'Orientation', 'Perimeter', 'summedPerimeter'])
                              
    
    #print(outdata)        
    return outdata.loc[int(in_img)]

def calculate_carbon_from_biovolume(invalue, category):
    """Calculate the cellular carbon from the given biovolume value based on  
       what category the image is assigned and how large it is. Conversion 
       formulas are from Table 4 in Menden-Deuer and Lessard (2000).
       
       inputs:
            invalue (float) = the biovolume value from the features file
            category (str) = the category to which the image was assigned 
    
       returns:
            carbon_value (float) = the carbon calculated from the formulas
    """
    diatoms = ['Asterionellopsis', 'Centric', 'Ch_simplex', 'Chaetoceros', 'Corethron', 'Cylindrotheca',
               'Cymatosira', 'DactFragCeratul', 'Ditlyum', 'Eucampia', 'Eucampiacornuta', 'Guinardia',
               'Hemiaulus', 'Leptocylindrus', 'Licmophora', 'Melosira', 'Odontella', 'Pleurosigma', 'Pseudonitzschia',
               'Rhizosolenia', 'Skeletonema', 'Thalassionema', 'Thalassiosira', 'centric10', 'pennate', ]

    if category in diatoms:
        if invalue > 3000.: # diatoms > 3000 cubic microns (um**3)
            carbon_value = (10**(-0.933)) * (invalue ** 0.881)
        else:
            carbon_value = (10**(-0.541)) * (invalue ** 0.811)
    else:
        if invalue < 3000.: # protist plankton < 3000 cubic microns (um**3)
            carbon_value = (10**(-0.583)) * (invalue ** 0.860)
        else:
            carbon_value = (10**(-0.665)) * (invalue ** 0.939)

    return carbon_value



def load_feature_file(in_feature):
    """Load the feature file into a pandas dataframe."""
    
    f = pd.read_csv(in_feature, index_col=0)
    return f

def parse_image_name(instr):
    "Parse the image filename"
    if instr[-4] == '.':
        instr = instr[:-4]
    if instr[0] == 'D':
        year = instr[1:5]
        name = instr[:24]
        img = instr.split('_')[-1]
    else:
        year = instr.split('_')[1]
        name = instr[:21]
        img = instr.split('_')[-1]
    return [year, name, img]

if __name__ == '__main__':
    #to do: load models, load the features and class files,
    # grab the list of files from each directory
    
    # start working through the images individually
    num_errors = 0
    all_img_features = []
    outdf = pd.DataFrame()
    for img_index, indiv_file in enumerate(os.listdir(folder_path)):
        features_found = False
        print(indiv_file)
        image = parse_image_name(indiv_file)
            
        for path in feature_paths:
            if not features_found:
                file_in_features = path + image[0] + '/'+image[1] + '_fea_v2.csv'
                try:
                    temp_biovolumes = grab_features(file_in_features, image[2])
                    all_img_features.append(temp_biovolumes)
                    outdf[indiv_file[:-4]] = temp_biovolumes
                    features_found = True
                except:
                    print(indiv_file, ' trying other feature path')
                    pass
    outdf.T.to_csv(outpath + 'temp.csv')
    #with open(outpath+'temp.csv') as f:
    
        