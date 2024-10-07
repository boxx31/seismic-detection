# For NASA Space Apps Challenge
# When run: the output is in output/catalog.csv. There may be repeated seismic events because multiple files share the same data.
# This program accesses other data through a catalog rather than directly

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.core.utcdatetime import UTCDateTime

cat_directory = './data/lunar/training/catalogs/'
data_directory = './data/lunar/training/data/S12_GradeA/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
cat = pd.read_csv(cat_file)

file_names = []
quake_times = []

# all variables - must be manually adjusted for non-lunar seismic activity
min_freq = 0.5
max_freq = 1.0
sta_noise = 80
lta_noise = 600
trigger_noise = 3
minimum_deviation = 0.25
sta_quake = 500
lta_quake = 3000
trigger_quake = 3
quake_length = 500
min_intensity = 0.00000000037

for row in cat.iloc:
    arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
    arrival_time_rel = row['time_rel(sec)']
    test_filename = row.filename

    mseed_file = f'{data_directory}{test_filename}.mseed'
    if (not os.path.exists(mseed_file)):
        continue
    st = read(mseed_file)
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data
    starttime = tr.stats.starttime.datetime
    arrival = (arrival_time - starttime).total_seconds()

    # bandpass filter
    st_filt = st.copy()
    st_filt.filter('bandpass',freqmin=min_freq,freqmax=max_freq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    # STA/LTA filter 1: check for noise
    sampling_rate = tr.stats.sampling_rate
    condition = classic_sta_lta(tr_data, int(sta_noise*sampling_rate), int(lta_noise*sampling_rate))
    noise_detected = np.array(trigger_onset(condition, trigger_noise, trigger_noise-0.1))
    noise_times = []
    for i in noise_detected:
        utc_start = UTCDateTime(starttime + timedelta(seconds = tr_times[i[0]]))
        utc_end = UTCDateTime(starttime + timedelta(seconds = tr_times[i[1]]))
        #print(tr_times[i[0]], tr_times[i[1]])
        tr_filt_slice = tr_filt.slice(utc_start, utc_end)
        if abs(tr_filt_slice.std()/tr_filt_slice.max()) < minimum_deviation:
            noise_times.append((utc_start, utc_end))

    # Remove spikes
    for i in noise_times:
        tr_filt.data[int((i[0] - tr.stats.starttime) * sampling_rate):int((i[1] - tr.stats.starttime) * sampling_rate)] = 0

    # Use sta/lta filter again
    sampling_rate = tr_filt.stats.sampling_rate
    condition = classic_sta_lta(tr_filt.data, int(sta_quake*sampling_rate), int(lta_quake*sampling_rate))
    quake_detected = np.array(trigger_onset(condition, trigger_quake, trigger_quake-0.1))
    for i in quake_detected:
        start = starttime + timedelta(seconds = tr_times[i[0]])
        end = starttime + timedelta(seconds = tr_times[i[1]])
        if np.mean(np.abs(tr_filt.slice(UTCDateTime(start), UTCDateTime(start + timedelta(seconds = quake_length))).data)) > min_intensity:
            file_names.append(test_filename)
            quake_times.append(datetime.strftime(start,'%Y-%m-%dT%H:%M:%S.%f'))

# Compile dataframe of detections
result = pd.DataFrame(data = {'filename':file_names, 'time_abs(%Y-%m-%dT%H:%M:%S.%f)':quake_times})

result.to_csv('./output/catalog.csv', index=False)