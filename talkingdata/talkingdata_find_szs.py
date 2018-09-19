# coding: utf-8
import numpy as np
import pandas as pd
import os
import gc
os.environ['OMP_NUM_THREADS'] = '24'

# Set the path of the data
PATH = "~/.kaggle/competitions/talkingdata-adtracking-fraud-detection"
SCRATCH_PATH = "/scratch/brown/g1082124/talkingdata"

# Import dataset
print('Importing data...')
train = pd.read_csv(PATH + "/train.csv")
test = pd.read_csv(PATH + "/test.csv")

len_train = len(train)
print('Concatting datasets...')
train = train.append(test)
del test; gc.collect()

cat_vars = ["app", "device", "os", "channel", "hour", "day", "wday"]

print('Extracting date info...')
train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
train['wday'] = pd.to_datetime(train.click_time).dt.dayofweek.astype('uint8')

print('Group by ip-app-os combinations...')
#gp = train[["ip", "app", "os", "channel"]].groupby(by=["ip", "app", "os"])
#[['channel']].count().reset_index.rename(index=str, columns={'channel': 'ip_app_os_count'})
gp = train[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train = train.merge(gp, on=['ip', 'app', 'os'], how='left')

print('Group by ip-day-hour combinations...')
gp = train[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train = train.merge(gp, on=['ip', 'day', 'hour'], how='left')

print('Group by ip-app combinations...')
gp = train[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train = train.merge(gp, on=['ip', 'app'], how='left')

from sklearn.preprocessing import LabelEncoder
print('Label encoding...')
train[cat_vars] = train[cat_vars].apply(LabelEncoder().fit_transform)

print('Resplitting datasets...')
test = train[len_train:]
train = train[:len_train]

train.drop(['ip', 'click_id', 'click_time', 'attributed_time'], 1, inplace=True)

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
max_day = np.max([train['day'].max(), test['day'].max()]) + 1
max_wday = np.max([train['wday'].max(), test['wday'].max()]) + 1
max_qty = np.max([train['qty'].max(), test['qty'].max()]) + 1
max_c1 = np.max([train['ip_app_count'].max(), test['ip_app_count'].max()]) + 1
max_c2 = np.max([train['ip_app_os_count'].max(), test['ip_app_os_count'].max()]) + 1

print('max_app:', max_app)
#print('max_ip:', max_ip)
print('max_device:', max_device)
print('max_os:', max_os)
print('max_channel:', max_channel)
print('max_hour:', max_hour)
print('max_day:', max_day)
print('max_wday:', max_wday)
print('max_qty:', max_qty)
print('max_c1:', max_c1)
print('max_c2:', max_c2, '\n')
