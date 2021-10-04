#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:40:47 2019

@author: aimachine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:38:04 2019

@author: aimachine
"""

import numpy as np
from TerminatorUtils import helpers
from keras import callbacks
from keras.layers import Flatten
import os
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from stardist.models import Config3D, StarDist3D
from sklearn.utils import class_weight
from stardist import calculate_extents, Rays_GoldenSpiral
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory

except (ImportError,AttributeError):
    from backports import tempfile
    
    
    
    
class StarDistDetection3D(object):






     def __init__(self, NpzDirectory, TrainModelName, model_dir, model_name,  unet_n_depth = 3, n_rays = 96, epochs = 100, learning_rate = 0.0001):
         
         
         
         self.NpzDirectory = NpzDirectory
         self.TrainModelName = TrainModelName
         self.model_dir = model_dir
         self.model_name = model_name
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.unet_n_depth = unet_n_depth
         self.train_patch_size = None
         #Attributes to be filled later
         
         self.X = None
         self.Y = None
         self.axes = None
         self.X_val = None
         self.Y_val = None
         self.Trainingmodel = None
         self._rays = n_rays
         
         #Load training and validation data
         self.loadData()
         self.TrainModel()
         
         
     def loadData(self):
         
         
         load_path = self.NpzDirectory +  self.TrainModelName
         (X,Y), (X_val,Y_val), axes = helpers.load_training_data(load_path, validation_split=0.1, verbose=True)
         
         self.train_patch_size = (X[0].shape[0], X[0].shape[1], X[0].shape[2])
         self.X = X
         self.Y = Y
         self.X_val = X_val
         self.Y_val = Y_val
         self.axes = axes
         
         
         
     def TrainModel(self):
         
         print(Config3D.__doc__)
         extents = calculate_extents(self.Y)
         anisotropy = tuple(np.max(extents) / extents)
         rays = Rays_GoldenSpiral(self.n_rays, anisotropy=anisotropy)
         conf = Config3D (
              n_rays       = rays,
              train_epochs = self.epochs,
              train_learning_rate = self.learning_rate,
              unet_n_depth = self.unet_n_depth,
              train_checkpoint = self.model_dir + self.model_name +'.h5',
              train_patch_size = self.train_patch_size,
              grid         = tuple(1 if a > 1.5 else 2 for a in anisotropy),
              use_gpu      = True,
              n_channel_in = 1,
              )
         print(conf)
         vars(conf)
         lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
         hrate = callbacks.History()
         srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        
         print('empirical anisotropy of labeled objects = %s' % str(anisotropy))
        
         Starmodel = StarDist3D(conf, name=self.model_name, basedir=self.model_dir)
         Starmodel.optimize_thresholds(self.X_val, self.Y_val)
         
         Starmodel.fit(self.X, self.Y, validation_data=(self.X_val,self.Y_val), shuffle = True, callbacks = [lrate, hrate, srate])
         
       
         
         
         
         
         
         
         
         
         
         
         