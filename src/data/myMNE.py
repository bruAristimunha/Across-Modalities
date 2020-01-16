import mne
import scipy.io as sio
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from mne import create_info, EpochsArray

def makeMNE(file_name)-> mne.epochs.EpochsArray:
    trial     = lambda df : df['ft_data_auditory'].trial.reshape((240, 64, 539)) #n_epochs, n_channels, n_times
    ch_names  = lambda df : df['ft_data_auditory'].label
    badtrials = lambda df : df['ft_data_auditory'].badtrials
    time      = lambda df : df['ft_data_auditory'].time
    baseline     = lambda df : df['ft_data_auditory'].cfg.baseline
    import warnings
    
    
    warnings.filterwarnings("ignore")
    
    
    
    file = sio.loadmat(file_name,squeeze_me=True, struct_as_record=False)

    data = trial (file)

    info = mne.create_info(
              ch_names = list(ch_names (file)),
              ch_types='eeg',
              sfreq = 539)
    
    base = tuple(baseline (file))
    tmin = min(time (file))

    epochs =  EpochsArray(data,info,tmin=tmin,baseline=base,proj=True,verbose=False)
    
    epochs.set_montage(mne.channels.read_montage(kind='biosemi64'))

    
    return epochs