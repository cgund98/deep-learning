# %% Imports
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

# %% Load data
kaggle_path = '~/.kaggle/competitions/avito-demand-prediction/'
output_file = 'submits/lgb_avito.csv'

train = pd.read_csv(kaggle_path + 'train.csv', parse_dates=['activation_date'])
test = pd.read_csv(kaggle_path + 'test.csv', parse_dates=['activation_date'])
submit = pd.read_csv(kaggle_path + 'sample_submission.csv')
train_len = len(train)
data = pd.concat([train, test], axis=0)

print('Train Length: {} \nTest Length: {} \n'.format(train_len, len(test)))
# %% Info
print('Columns:\n', data.columns.values)
# %% Add time features
data['day_of_week'] = data.activation_date.dt.dayofweek
data['day_of_month'] = data.activation_date.dt.day
data['weekday'] = data.activation_date.dt.weekday
data['weekdofyear'] = data.activation_date.dt.week

data['price'] = np.log(data['price'] + .001)
data['price'].fillna(-999, inplace=True)
data['image_top_1'].fillna(-999, inplace=True)

data.drop(['image', 'activation_date'], axis=1, inplace=True)

# %% Encoding
cat_cols = ["user_id", "region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type", "image_top_1"]
data[cat_cols] = data[cat_cols].apply(LabelEncoder().fit_transform).astype(np.int32)
print(data[cat_cols].head())

# %% Text features
data['text_feat'] = data.apply(lambda row: ' '.join([
    str(row['param_1']),
    str(row['param_2']),
    str(row['param_3']),
]), axis=1)

text_features = ['description', 'text_feat', 'title']
for cols in text_features:
    data[cols] = data[cols].astype(str)
    data[cols] = data[cols].astype(str).fillna('nicaptota')
    data[cols] = data[cols].str.lower()
    data[cols + '_num_chars'] = data[cols].apply(len)
    data[cols + '_num_words'] = data[cols].apply(lambda comment: len(comment.split()))
    data[cols + '_num_unique_words'] = data[cols].apply(lambda comment: len(set(w for w in comment.split())))
    data[cols + '_words_vs_unique'] = data[cols + '_num_unique_words'] / data[cols + '_num_words'] * 100
data.drop(['param_1', 'param_2', 'param_3'],axis=1, inplace=True)

# %% Split datasets
#train_idx = data.loc[data.activation_date<=pd.to_datetime('2017-04-07')].index
#valid_idx = data.loc[data.activation_date>=pd.to_datetime('2017-04-08')].index
train = data.iloc[:train_len].copy()
y_train = train.deal_probability.values
test = data.iloc[train_len:].copy()
X_train, X_valid, y_train, y_valid = train_test_split(train, y_train, test_size=.10, random_state=42)

#%% Add means
# for c in agg_cols:
#     gp = X_train.groupby(c)['deal_probability']
#     mean = gp.mean()
#     std  = gp.std()
#     X_train[c + '_deal_probability_avg'] = X_train[c].map(mean)
#     X_train[c + '_deal_probability_std'] = X_train[c].map(std)
#     X_valid[c + '_deal_probability_avg'] = X_valid[c].map(mean)
#     X_valid[c + '_deal_probability_std'] = X_valid[c].map(std)
#     test[c + '_deal_probability_avg'] = test[c].map(mean)
#     test[c + '_deal_probability_std'] = test[c].map(std)
#
# for c in agg_cols:
#     gp = X_train.groupby(c)['price']
#     mean = gp.mean()
#     X_train[c + '_price_avg'] = X_train[c].map(mean)
#     X_valid[c + '_price_avg'] = X_valid[c].map(mean)
#     test[c + '_price_avg'] = test[c].map(mean)

# %% Get data ready for training
X_train.drop(text_features, inplace=True, axis=1)
X_valid.drop(text_features + ['item_id'], inplace=True, axis=1)
test.drop(text_features + ['item_id'], inplace=True, axis=1)
X_train.drop(['deal_probability', 'item_id'], inplace=True, axis=1); X_valid.drop(['deal_probability',], inplace=True, axis=1)

# %% Create lgbm datasets
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=lgb_train)

# %% Create model
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    #'num_leaves': 31,
    'max_depth': 15,
    'learning_rate': .025,
    'feature_fraction': .9,
    'bagging_fraction': .8,
    'bagging_freq': 5,
    'verbose': 0,
}
model = lgb.train(params, lgb_train, valid_sets=[lgb_valid], num_boost_round=5000, early_stopping_rounds=120, verbose_eval=100)
# .228957
# %% Predict
submit['deal_probability'] = model.predict(test, num_iteration=model.best_iteration)
submit['deal_probability'].clip(0.0, 1.0, inplace=True)
print(submit.head())
submit.to_csv(output_file, index=False)

# %% Plot Importances
%matplotlib inline
lgb.plot_importance(model, importance_type='gain', figsize=(10, 10))
