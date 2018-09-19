import pandas as pd
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '24'

path = "~/.kaggle/competitions/talkingdata-adtracking-fraud-detection/"

dtypes = {
            'ip' :'uint32',
                'app' :'uint16',
                    'device': 'uint16',
                        'os' :'uint16',
                            'channel': 'uint16',
                                #'is_attributed': 'uint8',
                                    #'click_id': 'uint32',
                                        'click_time': 'object'
                                        }

test_cols = ['ip', 'app', 'device', 'os', 'click_time', 'channel', 'click_id']
test_supp_cols = ['ip', 'app', 'device', 'os', 'click_time', 'channel']
test = pd.read_csv(path+'/test.csv', usecols=test_cols, dtype=dtypes)
test_supp_cols = ['ip', 'app', 'device', 'os', 'click_time', 'channel']
test_supp = pd.read_csv(path+'test_supplement.csv', usecols=test_supp_cols)
print(test_supp.loc[:, test_supp.isna().any()].head())
test_supp = test_supp.dropna()
test_supp = test_supp.astype(dtypes)
test['click_time'] = pd.to_datetime(test.click_time)
test_supp['click_time'] = pd.to_datetime(test_supp.click_time)
#test_supp = pd.read_csv(path+'test_supplement.csv', usecols=test_cols)
#test_supp = test_supp.dropna()
#test_supp = test_supp.astype(dtypes)
#test['click_time'] = pd.to_datetime(test.click_time)
#test_supp['click_time'] = pd.to_datetime(test_supp.click_time)
test_supp['is_attributed'] = .5
join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
#join_cols = ['click_time']
all_cols = join_cols + ['is_attributed']

print('Test:')
print(test.click_time.dtype, test_supp.click_time.dtype)

test = test.merge(test_supp[all_cols], how='left', on=join_cols)
test = test.drop_duplicates(subset=['click_id'])

print('Supp:\n', test_supp[all_cols].dropna().head())
print('Submit\n', test[all_cols].dropna().head())
print('Saving')

output_file = 'lgbm_submit.csv'

#test['is_attributed'] = test['is_attributed'].fillna(0.0)

#test[['click_id', 'is_attributed']].to_csv(output_file, index=False, float_format='%.9f')
