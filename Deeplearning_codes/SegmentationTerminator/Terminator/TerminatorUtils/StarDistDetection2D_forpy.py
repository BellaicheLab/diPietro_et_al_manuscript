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
from glob import glob
from tifffile import imread
from csbdeep.utils import axes_dict
from keras import backend as K
# import matplotlib.pyplot as plt
#from IPython.display import clear_output
from keras import optimizers
from stardist.models import Config2D, StarDist2D, StarDistData2D
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from csbdeep.io import load_training_data
from tqdm import tqdm
from csbdeep.utils import Path, normalize
import sys



from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
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
    
    
    
    
class StarDistDetection2D(object):






     def __init__(self, ImageDirectory, LabelDirectory, model_dir, model_name, copy_model_name = None, patch_size = (256, 256), use_gpu = True, unet_n_depth = 5, n_rays = 256, epochs = 100, learning_rate = 0.0001):
         
         
         
         self.ImageDirectory = ImageDirectory
         self.LabelDirectory = LabelDirectory
         self.model_dir = model_dir
         self.model_name = model_name
         
         self.n_rays = n_rays
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.unet_n_depth = unet_n_depth
         self.train_patch_size = patch_size
         #Attributes to be filled later
         self.use_gpu = use_gpu
         self.n_channel_in = 1
         self.X = sorted(glob(ImageDirectory))
         self.Y = sorted(glob(LabelDirectory))
         self.copy_model_name = copy_model_name   
         self.axes = None
         self.X_val = None
         self.Y_val = None
         self.X_trn = None
         self.Y_trn = None
         self.Trainingmodel = None
            
         
         #Load training and validation data
         self.loadData()
         self.TrainModel()
         
         
     def loadData(self):
         
         assert all(Path(x).name==Path(y).name for x,y in zip(self.X,self.Y))
         
         self.X = list(map(imread,self.X))
         self.Y  = list(map(imread,self.Y))
         
         axis_norm = (0,1)   # normalize channels independently


         self.X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(self.X)]
         self.Y = [y for y in tqdm(self.Y)]
         
         data = StarDistData2D(self.X,self.Y,batch_size=1,n_rays=self.n_rays,patch_size=self.train_patch_size,shape_completion=False)
         
         (img,dist_mask), (prob,dist) = data[0]

#          fig, ax = plt.subplots(2,2, figsize=(12,12))
#          for a,d,cm,s in zip(ax.flat, [img,prob,dist_mask,dist], ['gray','magma','bone','viridis'],
#                     ['Input image','Object probability','Distance mask','Distance (0°)']):
#             a.imshow(d[0,...,0],cmap=cm)
#             a.set_title(s)
#          plt.tight_layout()
#          plt.show()
         assert len(self.X) > 1, "not enough training data"
         rng = np.random.RandomState(42)
         ind = rng.permutation(len(self.X))
         n_val = max(1, int(round(0.15 * len(ind))))
         ind_train, ind_val = ind[:-n_val], ind[-n_val:]
         self.X_val, self.Y_val = [self.X[i] for i in ind_val]  , [self.Y[i] for i in ind_val]
         self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [self.Y[i] for i in ind_train] 
         print('number of images: %3d' % len(self.X))
         print('- training:       %3d' % len(self.X_trn))
         print('- validation:     %3d' % len(self.X_val))
         
     def TrainModel(self):
         
         conf = Config2D (
              n_rays       = self.n_rays,
              train_epochs = self.epochs,
              train_learning_rate = self.learning_rate,
              unet_n_depth = self.unet_n_depth,
              train_patch_size = self.train_patch_size,
              n_channel_in = self.n_channel_in,
              grid         = (2,2),
              train_loss_weights =  (1, 0.05),
              use_gpu      = self.use_gpu,
              train_tensorboard = True
              )
         print(conf)
         vars(conf)
         
         
         Starmodel = StarDist2D(conf, name=self.model_name, basedir=self.model_dir)
            
         if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_best.h5'):        
             Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_best.h5') 
             print('Loading Existing Model') 
            
         elif os.path.exists(self.model_dir + self.copy_model_name + '/' + 'weights_best.h5'):
             print('Loading Old Model')
             Starmodel.load_weights(self.model_dir + self.copy_model_name + '/' + 'weights_best.h5')
         
                
                
         Starmodel.train(self.X_trn, (self.Y_trn), validation_data=(self.X_val,(self.Y_val)), epochs = self.epochs)
         Starmodel.optimize_thresholds(self.X_val, self.Y_val)
         
         
         
         
         
         
         
         
         
         
         
         
         
