import numpy as np
import pandas as pd

print('Import csvs...\n')

df1 = pd.read_csv('submits/lgbm0_sub.csv')
df2 = pd.read_csv('submits/lgbm1_sub.csv')
df3 = pd.read_csv('submits/lgbm2_sub.csv')
df4 = pd.read_csv('submits/lgbm3_sub.csv')
df5 = pd.read_csv('submits/lgbm4_sub.csv')
#df6 = pd.read_csv('submits/panji_lgbm.csv')
#df7 = pd.read_csv('submits/simple_averaging.csv')
#df8 = pd.read_csv('submits/lgbm_50m_submit.csv')

models = { 'df1': {
                   'name': 'mylgbm',
                   'score': 97.98,
                   'df':df1},
           'df2': {
                   'name': 'mylgbm_w_nowday',
                   'score': 97.98,
                   'df': df2},
           'df3': {
                   'name': 'asrafuls_lgbm',
                   'score': 97.8,
                   'df': df3},
           'df4': {
                   'name': 'fm_ftrl',
                   'score': 97.72,
                   'df': df4},
           'df5': {
                   'name': 'baris',
                   'score': 97.98,
                   'df': df5},
#           'df6': {
#                   'name': 'panji',
#                   'score': 97.98,
#                   'df': df6},
#           'df7': {
#                   'name': 'averging',
#                   'score': 97.98,
#                   'df': df7},
#           'df8': {
#                   'name': 'mylgbm_50m',
#                   'score': 97.98,
#                   'df': df8},
           }
isa_lg = 0
isa_hm = 0
isa_am = 0

for df in models.keys():
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
    isa_am += models[df]['df'].is_attributed

isa_lg = np.exp(isa_lg/5)
isa_hm = 5/isa_hm
isa_am = isa_am/5

sub_am = pd.DataFrame()
sub_am['click_id'] = df1['click_id']
sub_am['is_attributed'] = isa_am
print(df1.head(), df2.head(), sub_am.head())

isa_fin = (isa_am + isa_hm + isa_lg)/3

sub_fin = pd.DataFrame()
sub_fin['click_id'] = df1['click_id']
sub_fin['is_attributed'] = isa_fin

print('Saving...')
sub_fin.to_csv('submits/blended.csv', index=False, float_format='%.9f')
