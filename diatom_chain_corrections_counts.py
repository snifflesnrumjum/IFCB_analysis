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


from scipy.io import loadmat
import os
import pandas as pd
import numpy as np
import pickle
__author__ = 'dhenrichs'


infile = 'chain_correction.cfg' # name of the config file to use

def grab_config_variables(infile):
    '''Will load the necessary variables from the config file'''
    if infile not in os.listdir():
        print('Config file NOT FOUND')
        print('Please provide the correct config file')
        raise Exception('Config file error')
    else:
        #load config file
        with open(infile,'r') as f:
            in_config = [line[:-1] for line in f]
        
    outdict = {}
    for line in in_config:
        if line[0] == '#':
            pass
        elif '=' in line:
            temp = line.split(' ', maxsplit=2)
            outdict[temp[0]] = temp[2]
    
    return outdict

#read in the config file
input_dict = grab_config_variables(infile)

#copy the config variables to individual variables
feature_path = input_dict['feature_path']
class_path = input_dict['class_path']
outpath = input_dict['outpath']
date_limiter = input_dict['date_limiter']
automated_or_manual = input_dict['automated_or_manual']
model_path = input_dict['model_path']
cnn = input_dict['cnn']
ifcb_style = input_dict['ifcb_style']


if ifcb_style == 'old':
    filename_length = 21
    filestart = 'I'
else:
    filename_length = 24
    filestart = 'D'

def grab_features(in_feature, in_class, automated, models=None):
    """Open the feature file and obtain the listed features for each image.
    """
    
    feature_data = load_feature_file(in_feature)
    outdata = pd.DataFrame(feature_data, index=feature_data.index, 
                               columns=['Biovolume', 'Area', 'MajorAxisLength', 'MinorAxisLength',
                                        'summedBiovolume', 'summedArea', 'summedMajorAxisLength', 
                                        'summedMinorAxisLength', 'EquivDiameter', 'Extent', 'H90',
                                        'H180', 'Orientation', 'Perimeter', 'summedPerimeter'])
                              
    if automated == 'automated':
        if cnn:
            category_list, class_data, roinums = load_class_file_automated_CNNstyle(in_class)
        else:
            category_list, class_data, roinums = load_class_file_automated(in_class)
        
    elif automated == 'manual':
        #print "loading class file..."
        category_list, class_data, roinums = load_class_file_manual(in_class)
        #outdata['class'] = class_data
        
    else:
        return None
    
    if ifcb_style == 'new':
        #create a temporary dataframe to hold the data; this should solve the issue of feature file sizes not matching up with
        #class file sizes due to MATLAB scripts randomly skipping images
        classified_data = pd.DataFrame(class_data, index=roinums, columns=['class'])
        print len(class_data), len(class_data[0]), outdata.shape
        #concat the two dataframes based on intersection of index
        outdata = pd.concat([outdata, classified_data], axis=1).dropna(subset=['Biovolume'])
       
    else:
        print len(class_data), len(class_data[0]), outdata.shape
        if len(class_data) == outdata.shape[0]:
            outdata['class'] = class_data
        else:
            classified_data = pd.DataFrame(class_data, index=roinums, columns=['class'])
            outdata = pd.concat([outdata, classified_data], axis=1).dropna(subset=['Biovolume'])
            
        
    outdata['num_cells'] = 1
    for index, image in outdata.iterrows():
        outdata.loc[index, ['num_cells']] = get_num_cells_in_chain(image['class'], image, models)
        
    return outdata

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
    # categories as of 8/1/2021 CNN classifier
    # may need to update these as needed
    diatoms = ['Asterionellopsis', 'Asterionellopsis_single', 'Centric', 'Chaetoceros', 'Chaetoceros_danicus', 'Chaetoceros_peruvianus', 
               'Chaetoceros_simplex', 'Chaetoceros_single', 'Chaetoceros_socialis', 'Corethron', 'Cylindrotheca',
               'Cymatosira', 'DactFragCeratul', 'Dactyliosolen_blavyanus', 'Ditylum', 'Ephemera', 'Eucampia', 'Eucampia_cornuta', 'Guinardia',
               'Hemiaulus_curved', 'Hemiaulus_straight', 'Leptocylindrus', 'Licmophora', 'Odontella', 'Paralia', 'Pleurosigma', 'Pseudonitzschia',
               'Rhizosolenia', 'Skeletonema', 'Thalassionema', 'Thalassiosira', 'centric10', 'pennate', 'pennate_rod', ]

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


def get_num_cells_in_chain(in_class, in_features, models):
    """Will use the feature data and a pre-made neural network to estimate the
       number of cells in the diatom chain based upon the classifier output for
       each image. Only diatoms will be counted in this way.
    """
    diatom_chains = ['Asterionellopsis', 'Chaetoceros', 'Cymatosira', 'DactFragCeratul', 'Eucampia', 'Eucampia_cornuta', 
                     'Guinardia', 'Hemiaulus_curved', 'Leptocylindrus', 'Pseudonitzschia', 'Skeletonema', 'Thalassionema', 
                     'Thalassiosira']
    #print in_class
    #print in_class[1]
    if in_class in diatom_chains:
        #print
        #print "in_class", in_class
        #print 'in_features', in_features.shape
        temp_features = in_features.copy()
        dump = temp_features.pop('class')
        dump = temp_features.pop('num_cells')
        #print 'temp_features', temp_features.shape, temp_features.values
        temp_counts = models['{0}_scaler'.format(in_class)].transform(temp_features.values.reshape(1, -1))
        temp_counts = models[in_class].predict(temp_counts)
        temp = models['{0}_scaler_targets'.format(in_class)].inverse_transform(temp_counts)
	if int(round(temp)) < 1:
	    temp = 1
        #print temp
    else:
        temp = 1
    return int(round(temp))
       
 
def load_models(indir):
    """Load the pre-trained models for the given classes."""
    
    spp = ['Asterionellopsis', 'Chaetoceros', 'Cymatosira', 'DactFragCeratul', 'Eucampia', 'Eucampiacornuta', 
                      'Guinardia', 'Hemiaulus', 'Leptocylindrus', 'Pseudonitzschia', 'Skeletonema', 'Thalassionema', 
                      'Thalassiosira']      
    models = {}
    for cat in spp:
        models[cat] = pickle.load(open("{0}{1}_net.pkl".format(indir, cat)))
        models['{0}_scaler'.format(cat)] = pickle.load(open("{0}{1}_scaler.pkl".format(indir, cat)))
        models['{0}_scaler_targets'.format(cat)] = pickle.load(open("{0}{1}_scaler_targets.pkl".format(indir, cat)))
        
    # update hemiaulus to hemiaulus_curved
    models['Hemiaulus_curved'] = models['Hemiaulus']
    models['Hemiaulus_curved_scaler'] = models['Hemiaulus_scaler']
    models['Hemiaulus_curved_scaler_targets'] = models['Hemiaulus_scaler_targets]
    
    return models


def load_class_file_automated(in_class):
    """Load the automated classifier results and list of class names.
    Returns:
            category_list = list of category names
            class_data = list classifications for each roi image
    """
    f = loadmat(class_path + in_class, verify_compressed_data_integrity=False)
    print f.keys()
    class_data = f['TBclass_above_threshold'] #use this line for automated classifier results; can be 'TBclass_above_optthresh' if available
    class_data = [category[0][0] for category in class_data] #un-nest the MATLAB stuff #use this line for automated classifier results
    category_list = f['class2useTB']
    category_list = [category[0][0] for category in category_list] #un-nest the MATLAB stuff
    roinum = f['roinum']
    return category_list, class_data, roinum

def load_class_file_automated_CNNstyle(in_class):
    """Load the automated classifier results and list of class names.
    Returns:
            category_list = list of category names
            class_data = list classifications for each roi image
    """
    f = loadmat(class_path + in_class, verify_compressed_data_integrity=False)
    class_data = f['TBclass_above_threshold'] #use this line for automated classifier results; can be 'TBclass_above_optthresh' if available
    class_data = [category[0] for category in class_data[0]] #un-nest the MATLAB stuff #use this line for automated classifier results
    category_list = f['class2useTB']
    category_list = [category[0][0] for category in category_list] #un-nest the MATLAB stuff
    roinum = f['roinum']
    roinum = [num[0] for num in roinum]
    return category_list, class_data, roinum
    

def load_class_file_manual(in_class):
    """Load the manual correction output from the classifier. This will provide the
       corrected information about what class each image belongs in.
    """
    
    #the structure of the mat file variable with the classes is slightly different in manual files
    #classlist is a table of shape (num_rois x 3) with the columns being: roinum, manual category, automated category
    #print "loading mat file..."
    f = loadmat(class_path + in_class)
    roinums = None
    class_data_manual = f['classlist']
    class_data = f['classlist'][:,2]
    roinums = f['classlist'][:,0]
    #print "starting for loop..."
    for index, value in enumerate(class_data):
        if not np.isnan(class_data_manual[index, 1]):
            class_data[index] = class_data_manual[index,1]
    
    roinums = [roinums[x] for x,y in enumerate(class_data) if not np.isnan(y)]
    class_data = [x for x in class_data if not np.isnan(x)]  #it appears this is working as intended but the roinums need to adjusted too
    #print "getting category list..."
    #print f['class2use_manual']
    #print class_data
    #print len(class_data)
    category_list = f['class2use_manual']
    #print len(category_list), max(class_data)
    #category_list = [category[0] for category in category_list[0] if len(category) > 0]
    category_list = [category[0] if len(category) > 0 else '' for category in category_list[0]]
    #print len(category_list), max(class_data)
    class_data = [category_list[int(x-1)] for x in class_data]
    #print "finished for loop..."
    return category_list, class_data, roinums


def load_feature_file(in_feature):
    """Load the feature file into a pandas dataframe."""
    
    f = pd.read_csv(feature_path + in_feature, index_col=0)
    return f


if __name__ == '__main__':
    #to do: load models, load the features and class files,
    # grab the list of files from each directory
    print "Loading models...",
    models = load_models(model_path) #load the models
    print "done."
    print "Getting features and class files..."
    list_of_feature_files = os.listdir(feature_path)
    list_of_class_files = os.listdir(class_path)
    list_of_class_files.sort()
    files_done = set([x[:filename_length] for x in os.listdir(outpath)])
    print "done."
    print "Feature files: {}".format(len(list_of_feature_files))
    print "Class files  : {}".format(len(list_of_class_files))
    print "Num files left to process: {}".format(len(list_of_class_files) - len(files_done))
    # start working through the class files individually
    num_errors = 0
    for class_index, indiv_file in enumerate(list_of_class_files):
        if indiv_file[-3:] == 'mat' and indiv_file[:filename_length] not in files_done and indiv_file[0] == filestart:
            if not date_limiter or date_limiter == indiv_file[:len(date_limiter)]:
                print "Processing {}...".format(indiv_file),
                features_found = True
	        try:
                if 1:
	            feature_index = 0
	            while list_of_feature_files[feature_index][:filename_length] != indiv_file[:filename_length]:
	                feature_index += 1
	                if feature_index >= len(list_of_feature_files)-1:
	                    print "feature file not found (first try)."
	                
	                    #raise ValueError("The feature file was not found") #this will error out and stop the program
	                    #print "feature file not found (first try)."
	                    features_found = False
                        #print feature_index

	            if features_found:
	                #print "grabbing features..."
	                temp_biovolumes = grab_features(list_of_feature_files.pop(feature_index), list_of_class_files[class_index], automated_or_manual, models)
	                temp_biovolumes.to_csv(outpath + indiv_file[:-3] + 'csv')
	                print "done!"
	       	except:
                
    		    num_errors += 1
    	            print "something went wrong. Number of errors = {0}".format(num_errors)
    		    if num_errors > 20:
    			    raise ValueError('Number of errors exceeded 20')
    
        else:
            continue
