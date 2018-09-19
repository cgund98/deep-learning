
# coding: utf-8

# ## Imports

# In[1]:


from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Merge
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
import numpy as np
import pandas as pd


# In[2]:


# Set the path of the data
PATH = "~/.kaggle/competitions/talkingdata-adtracking-fraud-detection"
SCRATCH_PATH = "/scratch/brown/g1082124/talkingdata"


# In[3]:


# Import dataset
print('Loading test set...')
test = pd.read_pickle(SCRATCH_PATH + "/test_proc.pkl")




cat_vars = ["ip", "app", "device", "os", "channel"]

def getKerasData(dataset):
    X = {
        'app': np.array(dataset.app),
        'ip': np.array(dataset.ip),
        'device': np.array(dataset.device),
        'os': np.array(dataset.os),
        'channel': np.array(dataset.channel),
    }
    return X



# Get Nums of Unique Values (max value + 1)
max_app = 730
max_ip = 333168
max_device = 3799
max_os = 856
max_channel = 202


# In[27]:

print('Setting up model...')
# Define the inputs and embedding layers
emb_szs = emb_n = 50
dense_szs = 512

in_app = Input(shape=[1], name="app")
emb_app = Embedding(max_app, emb_szs)(in_app)
in_ip = Input(shape=[1], name="ip")
emb_ip = Embedding(max_ip, emb_szs)(in_ip)
in_device = Input(shape=[1], name="device")
emb_device = Embedding(max_device, emb_szs)(in_device)
in_os = Input(shape=[1], name="os")
emb_os = Embedding(max_os, emb_szs)(in_os)
in_channel = Input(shape=[1], name="channel")
emb_channel = Embedding(max_channel, emb_szs)(in_channel)

embs = concatenate([(emb_app), (emb_ip), (emb_device), (emb_os), (emb_channel)])

# Now turn those into dense layers
s_dout = SpatialDropout1D(0.5)(embs)
x = Flatten()(s_dout)
x = Dropout(0.5)(Dense(dense_szs, activation="relu")(x))
x = BatchNormalization()(x)
x = Dropout(0.5)(Dense(dense_szs, activation="relu")(x))
x = BatchNormalization()(x)
output = Dense(1, activation="sigmoid")(x)

#model = Model(inputs=[in_app, in_ip, in_device, in_os, in_os, in_channel], outputs=output)
model = Model(inputs=[in_app,in_channel,in_device,in_os,in_ip], outputs=output)
optimizer_adam = Adam(lr=0.001)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])


# In[28]:


#model.summary()
model.load_weights('../models/talkingdata_sample.h5')


# In[ ]:


class_weight = {0:.01,1:.99}


# In[20]:

print('Making predictions...')
X_test = getKerasData(test)
preds = model.predict(X_test)


# In[21]:


data = []
for i in range(0, len(preds)):
    entry = { 'click_id': i, 'is_attributed': preds[i][0] }
    data.append(entry)
submit = pd.DataFrame(data=data)


# In[22]:




# In[23]:

print('Creating .csv file')
submit.to_csv('submits/talkingdata_sample_submit.csv', index=False)

