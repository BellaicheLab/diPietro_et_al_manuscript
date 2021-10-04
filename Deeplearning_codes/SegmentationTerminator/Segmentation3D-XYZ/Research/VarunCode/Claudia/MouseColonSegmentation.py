#!/usr/bin/env python
# coding: utf-8

# # First we load the Data science bowl dataset of Kaggle and train it using N ray startdist suitable for small size florescent cells

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import glob
import sys
sys.path.append('../../Terminator/')
import glob
from TerminatorUtils.LoadKaggle import MembraneTraining, TwoDMembraneTraining
from tifffile import imread
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.utils import axes_dict, plot_some, plot_history
# In[ ]:





# In[2]:

MouseColon_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/MouseColon/'
NPZ_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/'
NPZ_filename = 'MouseColon.npz'
TwoDNPZ_filename = 'TwoDMouseColon.npz'
MouseColon_Model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/KaggleDSB/'
MouseColon_Model_name = 'FakeMembraneSmartSeeds'
UNETMouseColon_Model_name = 'FakeMembraneUNET'

TwoDMouseColon_Model_name = 'TwoDFakeMembraneSmartSeeds'
TwoDUNETMouseColon_Model_name = 'TwoDFakeMembraneUNET'




# In[3]:

#Network training parameters
Rays = 64
NetworkDepth = 3
Epochs = 100
LearningRate = 1.0E-4
batch_size = 4
PatchX = 512
PatchY = 512
PatchZ = 16
Kernel = 7


# # Do stardist training for the DSB dataset

# In[ ]:



TwoDMembraneTraining(MouseColon_dir, NPZ_dir, TwoDNPZ_filename, TwoDMouseColon_Model_name, MouseColon_Model_dir, PatchX=PatchX, PatchY=PatchY, use_gpu = False, batch_size = batch_size, depth = NetworkDepth, kern_size = Kernel, n_rays = Rays, epochs = Epochs, learning_rate = LearningRate)




load_path = NPZ_dir + TwoDNPZ_filename

(X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= NetworkDepth,train_epochs= Epochs, train_batch_size = batch_size, unet_kern_size = Kernel, train_learning_rate = LearningRate, train_reduce_lr={'patience': 5, 'factor': 0.5})
print(config)
vars(config)


# In[ ]:
model = CARE(config, name = TwoDUNETMouseColon_Model_name, basedir = MouseColon_Model_dir)
history = model.train(X,Y, validation_data=(X_val,Y_val))
model.export_TF()





MembraneTraining(MouseColon_dir, NPZ_dir, NPZ_filename, MouseColon_Model_name, MouseColon_Model_dir, PatchX=PatchX, PatchY=PatchY, PatchZ = PatchZ, use_gpu = True, batch_size = batch_size, depth = NetworkDepth, kern_size = Kernel, n_rays = Rays, epochs = Epochs, learning_rate = LearningRate)



# In[ ]:
load_path = NPZ_dir + NPZ_filename

(X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= NetworkDepth,train_epochs= Epochs, train_batch_size = batch_size, unet_kern_size = Kernel, train_learning_rate = LearningRate, train_reduce_lr={'patience': 5, 'factor': 0.5})
print(config)
vars(config)


# In[ ]:
model = CARE(config, name = UNETMouseColon_Model_name, basedir = MouseColon_Model_dir)
history = model.train(X,Y, validation_data=(X_val,Y_val))
model.export_TF()




# In[ ]:




