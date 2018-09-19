# coding: utf-8
import numpy as np
import pandas as pd
import gc
import os
os.environ['OMP_NUM_THREADS'] = '24'

# Set the path of the data
PATH = "~/.kaggle/competitions/talkingdata-adtracking-fraud-detection"
SCRATCH_PATH = "/scratch/brown/g1082124/talkingdata"

dtypes = {
	'ip'		: 'uint32',
	'app'		: 'uint16',
	'device'	: 'uint16',
	'os'		: 'uint16',
	'channel'	: 'uint16',
	'is_attributed'	: 'uint8',
	'click_id'	: 'uint32',
	}

# Import dataset
print('Importing data...')
#train = pd.read_csv(PATH + "/train.csv", dtype=dtypes)
train = pd.read_csv(PATH +"/train.csv", dtype=dtypes, skiprows = range(1, 131886954))
test = pd.read_csv(PATH + "/test.csv", dtype=dtypes)

len_train = len(train)
print('Concatting datasets...')
train = train.append(test)
del test; gc.collect()

cat_vars = ["app", "device", "os", "channel", "hour", "wday"]

print('Extracting date info...')
train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
train['wday'] = pd.to_datetime(train.click_time).dt.dayofweek.astype('uint8')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

train['in_test_hour'] = (3
			 - 2 * train['hour'].isin( most_freq_hours_in_test_data )
			 - 1 * train['hour'].isin( most_freq_hours_in_test_data )).astype('uint8')
print('Group by ip-day-test-hour...')
gp = train[['ip', 'day', 'in_test_hour', 'channel']].groupby(by=['ip', 'day', 'in_test_hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hour'})
train = train.merge(gp, on=['ip', 'day', 'in_test_hour'], how='left')
train.drop(['in_test_hour'], axis=1, inplace=True)
train['nip_day_test_hour'] = train['nip_day_test_hour'].astype('uint32')

print('Group by ip-app-os combinations...')
gp = train[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train = train.merge(gp, on=['ip', 'app', 'os'], how='left')

print('Group by ip-day-hour combinations...')
gp = train[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train = train.merge(gp, on=['ip', 'day', 'hour'], how='left')

print('Group by ip-day-hour-os...')
gp = train[['ip', 'day', 'hour', 'os', 'channel']].groupby(by=['ip', 'day', 'os', 'hour']).count().reset_index().rename(index=str, columns={'channel': 'nip_day_hour_os'})
train = train.merge(gp, on=['ip', 'day', 'os', 'hour'], how='left')
train['nip_day_hour_os'] = train['nip_day_hour_os'].astype('uint16')

print('Group by ip-app combinations...')
gp = train[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train = train.merge(gp, on=['ip', 'app'], how='left')

print('Group by ip-day-hour-device...')
gp = train[['ip', 'day', 'hour', 'device', 'channel']].groupby(by=['ip', 'day', 'hour', 'device'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hour_device'})
train = train.merge(gp, on=['ip', 'day', 'hour', 'device'], how='left')

from sklearn.preprocessing import LabelEncoder
print('Label encoding...')
train[cat_vars] = train[cat_vars].apply(LabelEncoder().fit_transform)

print('Resplitting datasets...')
test = train[len_train:]
train = train[:len_train]

train.drop(['ip', 'click_id', 'click_time', 'attributed_time', 'day'], 1, inplace=True)

def getKerasData(dataset):
    X = {
        'app': np.array(dataset.app),
        'ip': np.array(dataset.ip),
        'device': np.array(dataset.device),
        'os': np.array(dataset.os),
        'channel': np.array(dataset.channel),
    }
    return X

print('Finding emb szs...')

# Get Nums of Unique Values (max value + 1)
max_app = np.max([train['app'].max(), test['app'].max()]) + 1
#max_ip = np.max([train['ip'].max(), test['ip'].max()]) + 1
max_device = np.max([train['device'].max(), test['device'].max()]) + 1
max_os = np.max([train['os'].max(), test['os'].max()]) + 1
max_channel = np.max([train['channel'].max(), test['channel'].max()]) + 1
max_hour = np.max([train['hour'].max(), test['hour'].max()]) + 1
#max_day = np.max([train['day'].max(), test['day'].max()]) + 1
max_wday = np.max([train['wday'].max(), test['wday'].max()]) + 1
max_qty = np.max([train['qty'].max(), test['qty'].max()]) + 1
max_c1 = np.max([train['ip_app_count'].max(), test['ip_app_count'].max()]) + 1
max_c2 = np.max([train['ip_app_os_count'].max(), test['ip_app_os_count'].max()]) + 1
max_c3 = np.max([train['nip_day_test_hour'].max(), test['nip_day_test_hour'].max()]) + 1
max_c4 = np.max([train['nip_day_hour_os'].max(), test['nip_day_hour_os'].max()]) + 1
max_c5 = np.max([train['nip_hour_device'].max(), test['nip_hour_device'].max()]) + 1

print('max_app:', max_app)
#print('max_ip:', max_ip)
print('max_device:', max_device)
print('max_os:', max_os)
print('max_channel:', max_channel)
print('max_hour:', max_hour)
#print('max_day:', max_day)
print('max_wday:', max_wday)
print('max_qty:', max_qty)
print('max_c1:', max_c1)
print('max_c2:', max_c2)
print('max_c3:', max_c3)
print('max_c4:', max_c4)
print('max_c5:', max_c5, '\n')

print('Splitting and saving the train set...')
idx_1_4 = int(len_train / 4)
idx_2_4 = int(len_train / 2)
idx_3_4 = int(len_train - len_train / 4)

#print('Saving train_1_4.csv...')
#train.iloc[:idx_1_4].to_csv(SCRATCH_PATH + '/train_1_4.csv', index=False)
#print('Saving train_2_4.csv...')
#train.iloc[idx_1_4:idx_2_4].to_csv(SCRATCH_PATH + '/train_2_4.csv', index=False)
#print('Saving train_3_4.csv...')
#train.iloc[idx_2_4:idx_3_4].to_csv(SCRATCH_PATH + '/train_3_4.csv', index=False)
#print('Saving train_4_4.csv...')
#train.iloc[idx_3_4:].to_csv(SCRATCH_PATH + '/train_4_4.csv', index=False)

print('Saving test_proc.csv...')
test.to_csv(SCRATCH_PATH + '/test_proc.csv', index=False)

print('Saving train_50m_proc.csv...')
train.to_csv(SCRATCH_PATH + '/train_50m_proc.csv', index=False)

#print('Saving train_1_4.pkl...')
#train.iloc[:idx_1_4].to_pickle(SCRATCH_PATH + '/train_1_4.pkl')
#print('Saving train_2_4.pkl...')
#train.iloc[idx_1_4:idx_2_4].to_pickle(SCRATCH_PATH + '/train_2_4.pkl')
#print('Saving train_3_4.pkl...')
#train.iloc[idx_2_4:idx_3_4].to_pickle(SCRATCH_PATH + '/train_3_4.pkl')
#print('Saving train_4_4.pkl...')
#train.iloc[idx_3_4:].to_pickle(SCRATCH_PATH + '/train_4_4.pkl')
#
#print('Saving test_proc.pkl...')
#test.to_pickle(SCRATCH_PATH + '/test_proc.pkl')


