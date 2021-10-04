#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TTMaskTrainingSetCreated'
while os.path.exists(TriggerName) == False :
    time.sleep(60*5)


# In[2]:


from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history

from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[3]:


BaseDir = '/run/media/sancere/DATA/Lucas_NextonCreated_npz/'
NPZdata = 'Masks_Generator_Mix2_onlyequalized'+ '.npz'

ModelDir ='/home/sancere/NextonDisk_1/Lucas_Model_to_use/Mask_Generator/'
ModelName = 'Masks_Generator_Mix2_onlyequalized'
TransferModelName = 'Masks_Generator_Mix_onlyequalized'


load_path = BaseDir + NPZdata 


# In[4]:



(X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


# In[5]:


plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)');


# In[6]:


config = Config(axes, n_channel_in, n_channel_out, probabilistic = False, unet_n_depth=5,unet_kern_size=7,train_epochs= 70, train_batch_size = 64, train_reduce_lr={'patience': 5, 'factor': 0.5})
print(config)
vars(config)


# In[7]:


model = CARE(config = config, name = ModelName, basedir = ModelDir)
model.load_weights(ModelDir + TransferModelName + '/' + 'weights_best.h5')


# In[8]:


history = model.train(X,Y, validation_data=(X_val,Y_val))


# In[9]:


print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);


# In[10]:


plt.figure(figsize=(12,7))
_P = model.keras_model.predict(X_val[:5])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');


# In[11]:


model.export_TF()

