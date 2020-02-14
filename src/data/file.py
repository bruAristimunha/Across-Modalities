"""
Created on Fri Fev 13 16:26:00 2020
@author: Bruno Aristimunha
"""
import sys
import scipy.io as sio
from pandas import DataFrame
from numpy import array
from os import listdir
from os.path import isfile, join

from myMNE import makeMNE


def files_in_path(path):
    return [path+"/"+f for f in listdir(path) if isfile(join(path, f))]

# Getting data from MNE strutuct
def data_from_mne(files):
    return array([file.get_data().T for file in files])

def read_file(PATH_AUD, PATH_VIS):
    path_file_aud = files_in_path(PATH_AUD)
    path_file_vis = files_in_path(PATH_VIS)

    # Reading files with the MNE library

    files_aud = list(map(makeMNE, path_file_aud))
    files_vis = list(map(makeMNE, path_file_vis))

    # Getting data in numpy format
    data_aud = data_from_mne(files_aud)
    data_vis = data_from_mne(files_vis)
    
    return data_aud,data_vis, files_aud[0].ch_names


def _bad_trials(file_name):

    file = sio.loadmat(file_name,squeeze_me=True, struct_as_record=False)

    badtrials = lambda df : df['ft_data_auditory'].badtrials
        
    return badtrials(file)

def get_bad_trials(PATH_AUD, PATH_VIS):

    path_file_aud = files_in_path(PATH_AUD)
    path_file_vis = files_in_path(PATH_VIS)

    
    bad_trials_aud = list(map(_bad_trials, path_file_aud))
    bad_trials_vis = list(map(_bad_trials, path_file_vis))
    
    df_bad_trials_aud = DataFrame([bad_trials_aud],index=['Aud'])
    df_bad_trials_vis = DataFrame([bad_trials_vis],index=['Vis'])

    return df_bad_trials_aud, df_bad_trials_vis


def get_bad_trials_comportamental(modality: str, N_TRIALS = 120, PATH_INFO = '../data/raw/info_'):
    """Read function to get the time (or the indice) when occurs S2.

    Parameters
    ----------
    modality: str 
       It will only work if the modality equals to 'aud' or 'vis'.

    export_as_indice: bool
        Control option for export type (indice or time) 

    Returns
    -------
    agg_by_person: np.array

        TO-DO: text.

    """

    # Concatenating with 'aud' or 'vis'
    info_path = PATH_INFO+modality

    # Mapping the files listed in the folder for reading function.
    # Each file contains information about a single individual experiment.
    delays_people = list(map(sio.loadmat, files_in_path(info_path)))

    # Accumulator variable
    agg_by_person = []

    for delay_by_person in delays_people:
        #import pdb; pdb.set_trace()
        # Accessing value in struct from matlab
        time_delay_by_person = delay_by_person['report']['all_trials_delay'][0][0][0]
        # Values in second
        time_reproduce_by_person = delay_by_person['report']['time_action'][0][0][0]

        agg_by_person.append(time_reproduce_by_person/time_delay_by_person)
    # Export as numpy for simplicity

    bad_comport = [DataFrame(array(agg_by_person))[i].apply(lambda x: (
        False if ((x >= 2.0) | (x < 0.5)) else True)) for i in range(N_TRIALS)]

    df_bad_comport = DataFrame(bad_comport).stack(-1).reset_index()

    df_clean_bad_comport = df_bad_comport[df_bad_comport[0] != True].reset_index(drop=True)
    
    df_clean_bad_comport.columns = ['trial', 'people','bad_flag']
    
    df_clean_bad_comport['bad_flag'] = ~df_clean_bad_comport['bad_flag']
    
    return df_clean_bad_comport[['people','trial','bad_flag']]


def get_time_delay(modality: str, export_as_indice=True, PATH_INFO = '../data/raw/info_'):
    
    """Read function to get the time (or the indice) when occurs S2.

    Parameters
    ----------
    modality: str 
       It will only work if the modality equals to 'aud' or 'vis'.

    export_as_indice: bool
        Control option for export type (indice or time) 
        
    Returns
    -------
    agg_by_person: np.array
        
        TO-DO: text.
        
    """
    
    # Concatenating with 'aud' or 'vis'
    info_path = PATH_INFO+modality

    # Mapping the files listed in the folder for reading function.
    # Each file contains information about a single individual experiment.
    delays_people = list(map(sio.loadmat, files_in_path(info_path)))

    # Accumulator variable
    agg_by_person = []

    for delay_by_person in delays_people:

        # Accessing value in struct from matlab
        time_delay_by_person = delay_by_person['report']['all_trials_delay'][0][0][0]
        # Values in second
        #import pdb; pdb.set_trace()
        
        if(export_as_indice):
            # second to milisecond, 4 from 250 Hz
            agg_by_person.append((time_delay_by_person*250).astype(int))
        else:
            agg_by_person.append(time_delay_by_person)

    # Export as numpy for simplicity
    return array(agg_by_person)

    