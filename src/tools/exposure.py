"""
Created on Fri Fev 13 23:12:00 2020
@author: Bruno Aristimunha
"""
from pandas import DataFrame
from numpy import array, divide, average
from os import listdir
from os.path import isfile, join

N_TRIALS = 120

def fixing_bad_trials(bad_aud_1, bad_aud_2, bad_vis_1, bad_vis_2):
    bad_aud_1 = bad_aud_1.T.reset_index()
    bad_aud_2 = bad_aud_2.T.reset_index()
    bad_vis_1 = bad_vis_1.T.reset_index()
    bad_vis_2 = bad_vis_2.T.reset_index()

    clean_aud_1 = bad_aud_1.explode(column='Aud')
    clean_aud_1.columns = ['people', 'trial']
    clean_aud_1 = clean_aud_1.dropna()
    clean_aud_1 = clean_aud_1.reset_index(drop=True)
    clean_aud_1['bad_flag'] = True

    clean_aud_2 = bad_aud_2.explode(column='Aud')
    clean_aud_2.columns = ['people', 'trial']
    clean_aud_2 = clean_aud_2.dropna()
    clean_aud_2 = clean_aud_2.reset_index(drop=True)
    clean_aud_2['bad_flag'] = True

    clean_vis_1 = bad_vis_1.explode(column='Vis')
    clean_vis_1.columns = ['people', 'trial']
    clean_vis_1 = clean_vis_1.dropna()
    clean_vis_1 = clean_vis_1.reset_index(drop=True)
    clean_vis_1['bad_flag'] = True

    clean_vis_2 = bad_vis_2.explode(column='Vis')
    clean_vis_2.columns = ['people', 'trial']
    clean_vis_2 = clean_vis_2.dropna()
    clean_vis_2 = clean_vis_2.reset_index(drop=True)
    clean_vis_2['bad_flag'] = True
    
    return clean_aud_1,clean_aud_2,clean_vis_1,clean_vis_2
    
 # Function to exposure separation:

def _exposure_1(data): 
    # Getting the first exposure.
    # [individual, raw, channels, trial]
    return data[::, ::, ::, 0:: 2]

def _exposure_2(data): 
    # Getting the second exposure.
    # [individual, raw, channels, trial]
    return data[::, ::, ::, 1:: 2]

# Function to exposure separation:

def _exposure_1_bad(df_bad_trials,type_name): 
    exposure_1_list = []
    for ind, person in df_bad_trials.T.iteritems():
        by_person = []
        for ind_b, bad in person.iteritems():
            if(not(isinstance(bad, int))):
                # import pdb;pdb.set_trace()
                by_person.append(divide((bad[ 0:: 2]),2.).astype(int))
            else:
                if(isinstance(bad, int)):
                    if(bad %2 == 0):
                        # print(bad)
                        by_person.append([int(bad/2)])
                    else:
                        by_person.append([])
                else:
                    if(bad == []):
                        by_person.append([])
        exposure_1_list.append(by_person)
    
    
    return DataFrame(exposure_1_list, index=[type_name])

def _exposure_2_bad(df_bad_trials, type_name): 
    exposure_2_list = []
    for ind, person in df_bad_trials.T.iteritems():
        by_person = []
        for ind_b, bad in person.iteritems():
            if(not(isinstance(bad, int))):
                by_person.append(divide((bad[ 1:: 2]),2.).astype(int))
            else:
                if(isinstance(bad, int)):
                    if(bad %2 != 0):
                        # print(bad)
                        by_person.append([int(bad/2)])
                    else:
                        by_person.append([])
                else:
                    if(bad == []):
                        by_person.append([])
        exposure_2_list.append(by_person)

  
    return DataFrame(exposure_2_list, index=[type_name])

def get_last_125ms(exposure, indice_exposure, use_average=True):
    """Splitting array in bins for the first exposure,
    and compute the average (or not)

    Parameters
    ----------
    exposure: numpy.array (3-dimensions) [raw, channels, trial]
        Array containing data from an exposure.

    indice_exposure: numpy.array (1-dimension)
        Array containing indice when occurs the S2

    average: bool 
        Control option to future analyses without average
    Returns
    -------
    agg_per_trial: np.array
    
    """
    # Cutting values before the 0.
    zero = 39

    # Getting 2000 ms.
    exposure_without_zero = exposure[zero:]

    # Accumulator variable
    agg_per_trial = []

    for trial, ind_s2 in list(zip(exposure_without_zero.T, indice_exposure)):

        last_32_points = trial.T[int(ind_s2-32):int(ind_s2)]

        if(use_average):

            agg_per_trial.append(average(last_32_points, axis=0).reshape(-1))

        else:

            agg_per_trial.append(last_32_points)
    # ---------------------------------------------------------------------

    return array(agg_per_trial).T


def _categorize(second_time):
    '''
    This is a test:
    >>> categorize(0.750)
    1
    >>> categorize(0.8749)
    1
    >>> categorize(0.875)
    2
    >>> categorize(0.900)
    2
    >>> categorize(0.9899)
    2
    >>> categorize(1.000)
    3
    >>> categorize(1.001)
    3
    >>> categorize(1.124)
    3
    >>> categorize(1.125)
    4
    >>> categorize(1.249)
    4
    >>> categorize(1.250)
    5
    >>> categorize(1.3748)
    5
    >>> categorize(1.3759)
    6
    >>> categorize(1.375)
    6
    >>> categorize(1.520)
    6
    '''
    
    #Convert to miliseconds
    millisecond_time = int(second_time * 1000)  
    #André tirou o piso
    if((millisecond_time >= 750) and (millisecond_time < 875)):
        return (1)
    else:
        if ((millisecond_time >= 875) and (millisecond_time < 1000)):
            return (2)
        else:
            if ((millisecond_time >= 1000) and (millisecond_time < 1125)):
                return (3)
            else:
                if ((millisecond_time >= 1125) and (millisecond_time < 1250)):
                    return (4)
                else:
                    if ((millisecond_time >= 1250) and (millisecond_time < 1375)):
                        return (5)
                    else:#André tirou o topo.
                        if ((millisecond_time >= 1375) and (millisecond_time <= 1520)):
                            return (6)
    #Error escape
    return -1

def get_group_time(time_s2_aud, time_s2_vis):
    classes_vis = [DataFrame(time_s2_vis)[i].apply(_categorize)
               for i in range(N_TRIALS)]
    classes_aud = [DataFrame(time_s2_aud)[i].apply(_categorize)
                   for i in range(N_TRIALS)]

    classes_vis = DataFrame(classes_vis).T
    classes_aud = DataFrame(classes_aud).T
    
    return classes_aud, classes_vis

