#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:21:31 2019

@author: aimachine
"""



from TerminatorUtils.UNET import UNet2D
from TerminatorUtils import helpers
from keras import callbacks
import os
#from IPython.display import clear_output
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
    
    
    
    
class UNetDetection2D(object):






     def __init__(self, NpzDirectory, TrainModelName, model_dir, model_name, model_weights,  epochs = 100, learning_rate = 0.0001):
         
         
         
         self.NpzDirectory = NpzDirectory
         self.TrainModelName = TrainModelName
         self.ValidationModelName = ValidationModelName
         self.model_dir = model_dir
         self.model_name = model_name
         self.epochs = epochs
         self.model_weights = model_weights
         self.learning_rate = learning_rate
         #Attributes to be filled later
         
         self.X = None
         self.Y = None
         self.axes = None
        
         self.Trainingmodel = None
         
         #Load training and validation data
         self.loadData()
         self.TrainModel()
         
         
     def loadData(self):
         
         
         (X,Y), axes = helpers.load_full_training_data(self.NpzDirectory, self.TrainModelName, verbose = True)
         
         
         
         self.X = X
         self.Y = Y
         self.axes = axes
         
         
         
     def TrainModel(self):
         
         
         lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
         hrate = callbacks.History()
         srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        
        
         Binarymodel = UNet2D(1,1, model_weights=self.model_weights)
         
         Binarymodel.fit(self.X, self.Y,validation_split=0.1, shuffle = True, callbacks = [lrate, hrate, srate])
         
         # Removes the old model to be replaced with new model, if old one exists
         if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
         self.Trainingmodel.save(self.model_dir + self.model_name )
         
         
         
         
         
         
         
         
         
         
         
         