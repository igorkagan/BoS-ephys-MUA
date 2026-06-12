import os
import re
import pickle
import numpy as np
import xarray as xr
from scipy.io import loadmat
from frites.io import logger

# Set path
session_type = 'Curius_BLOCKED'
data_dir = f'/envau/work/comco/bardanikas.g/KaganLab/MUA_curated_sessions/{session_type}'
align = 'A_InitialFixationReleaseTime_ms'
session_list = os.listdir(data_dir)

for session in session_list:
    logger.info(f'Load session: {session}')

    # Load the trial information
    trials_file = f'{session}.trialinfo.4python'
    trial_info = loadmat(f'{data_dir}/{session}/{trials_file}')['cur_raster_labels']
    go_seq = trial_info['go_seq_500_list'][0][0].squeeze()
    go_seq = np.array([item[0] for item in go_seq])
    direction_A_lr = trial_info['A_LR_pos_list'][0][0].squeeze()
    direction_A_lr = np.array([item[0][1] for item in direction_A_lr])
    trial_type = trial_info['TrialSubType_list'][0][0].squeeze()
    trial_type = np.array([item[0] for item in trial_type])
    rewards = trial_info['A_Reward_list'][0][0].squeeze()
    rewards = np.array([item[0] for item in rewards])
    aborted_trials_ind = np.where(rewards=='RA0')[0]
    condition = trial_info['conf_predictability_list'][0][0].squeeze()
    condition = np.array([item[0] for item in condition])

    # Load the times
    times_file = f'{session}.{align}.MUA.pre1000ms.post1000ms.x_vector_ms'
    time_vec = loadmat(f'{data_dir}/{session}/{align}/{times_file}')
    times = time_vec['x_vector_ms'].squeeze()

    # Find the channel files names
    channel_dir_files = os.listdir(f'{data_dir}/{session}/{align}')
    chInds = []
    for f,file in enumerate(channel_dir_files):
        if re.search('data', file): chInds.append(f)
    channel_filenames = np.array(channel_dir_files)[chInds]

    # Load the channel data
    ntrials, nchannels, ntimes = go_seq.size, len(channel_filenames), times.size
    mua = np.zeros((ntrials,nchannels,ntimes))
    channels = np.zeros(nchannels, dtype=object)
    index = session.find('A_')
    split_ch_id = 64 if session[index+2] == 'C' else 32

    for ii,ch_file in enumerate(channel_filenames):
        # Unpack the MUA
        data = loadmat(f'{data_dir}/{session}/{align}/{ch_file}')
        mua[:,ii,:] = data['cur_output_data']

        # Create the channel names vector
        ind = ch_file.find('ch')
        ch_id = ch_file[ind+2:ind+5]
        if int(ch_id) <= split_ch_id: 
            channels[ii] = 'PMv' + ch_id
        else: channels[ii] = 'PMd' + ch_id

    # Delete aborted trials
    ntrials = go_seq.size
    mask = np.ones(ntrials, dtype=bool)
    mask[aborted_trials_ind] = False
    mua = mua[mask]
    go_seq, condition = go_seq[mask], condition[mask]
    direction_A_lr, trial_type = direction_A_lr[mask], trial_type[mask]

    # Z-score across all trials & times for each channel
    zmean = mua.mean(axis=(0,2), keepdims=True)
    zstd = mua.std(axis=(0,2), keepdims=True)
    muaNormalized = (mua-zmean)/zstd
    xr_mua = xr.DataArray(muaNormalized, dims=['trials','channels','times'], 
                          coords=[direction_A_lr, channels, times])
    attrs = {'Trial Type':trial_type, 'Go Sequence':go_seq, 'Condition':condition}

    # Save the MUA
    MUA_epochs_path = f'{data_dir}/MUA_Epochs'
    try: os.mkdir(MUA_epochs_path)
    except FileExistsError: pass

    mua_filename = f'{session}.{align}_MUA_epochs.nc'
    xr_mua.to_netcdf(f'{MUA_epochs_path}/{mua_filename}', engine='h5netcdf')

    # Save attributes with trial info
    attrs_filename = f'{session}.{align}_attrs.pkl'
    with open(f'{MUA_epochs_path}/{attrs_filename}','wb') as handle:
        pickle.dump(attrs, handle)
    logger.info(f'DataArray object is saved in the file: "{mua_filename}".')