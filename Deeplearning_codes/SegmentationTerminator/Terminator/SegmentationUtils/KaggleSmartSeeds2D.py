#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:38:04 2019

@author: aimachine
"""

import numpy as np
from SegmentationUtils import helpers
from keras import callbacks
from keras.layers import Flatten
import os
import imageio
from glob import glob
from tifffile import imread
from csbdeep.utils import axes_dict
from scipy.ndimage.morphology import  binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import find_objects
from keras import backend as K
from skimage.transform import resize
import matplotlib.pyplot as plt
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.utils import Path, normalize
#from IPython.display import clear_output
from keras import optimizers
from stardist.models import Config2D, StarDist2D, StarDistData2D
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import sys
from skimage.measure import label
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
    
    
def _raise(e):
    raise e

def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled
def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_dilation(mask,iterations = iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled    
    
class KaggleSmartSeeds2D(object):






     def __init__(self, BaseDir, NPZfilename, model_name, model_dir, GenerateNPZ = True, PatchX=128, PatchY=128,  use_gpu = True,  batch_size = 4, depth = 3, kern_size = 7, n_rays = 16, epochs = 400, learning_rate = 0.0001):
         
         
         
         
         self.NPZfilename = NPZfilename
         self.BaseDir = BaseDir
         self.model_dir = model_dir
        
         self.model_name = model_name
         self.GenerateNPZ = GenerateNPZ
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.depth = depth
         self.n_rays = n_rays
         self.kern_size = kern_size
         self.PatchX = PatchX
         self.PatchY = PatchY
         self.batch_size = batch_size
         self.use_gpu = use_gpu
    
        
         
         
         #Load training and validation data
         self.Train()
        
         
         
     def Train(self):
         
                   # Get train and test IDs
                    train_ids = next(os.walk(self.BaseDir))[1]
                    
                    
                    X_train = np.zeros((len(train_ids), self.PatchY, self.PatchX, 1), dtype=np.uint8)
                    
                    Binary_Y_train = np.zeros((len(train_ids), self.PatchY, self.PatchX, 1), dtype=np.bool)
                    
                
                    print('Loading images ')
                    sys.stdout.flush()
                
                    
                    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
                        path = self.BaseDir + id_
                        img = imageio.imread(path + '/images/' + id_ + '.png')[:,:,0]
                        img = resize(img, (self.PatchY, self.PatchX, 1), mode='constant', preserve_range=True)
                        X_train[n] = img
                        mask = np.zeros((self.PatchY, self.PatchX, 1), dtype=np.bool)
                        for mask_file in next(os.walk(path + '/masks/'))[2]:
                            if(mask_file.startswith('.') == False):
                              mask_ = imageio.imread(path + '/masks/' + mask_file)
                              mask_ = np.expand_dims(resize(mask_, (self.PatchY, self.PatchX), mode='constant', 
                                                        preserve_range=True), axis=-1)
                              mask = np.maximum(mask, mask_)
                        
                        Binary_Y_train[n] = mask
                        
                    axes = 'SXYC'
                    len(axes) == X_train.ndim or _raise(ValueError())  
                    if self.GenerateNPZ:
                      np.savez(self.BaseDir + self.NPZfilename, X = X_train, Y = Binary_Y_train, axes = axes) 
                    
                    
                    # Training UNET model
                    
                    print('Training UNET model')
                    load_path = self.BaseDir + self.NPZfilename + '.npz'

                    (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
                    c = axes_dict(axes)['C']
                    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
                    
                    config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= self.depth,train_epochs= self.epochs, train_batch_size = self.batch_size, unet_kern_size = self.kern_size, train_learning_rate = self.learning_rate, train_reduce_lr={'patience': 5, 'factor': 0.5})
                    print(config)
                    vars(config)
                    
                    model = CARE(config , name = 'UNET' + self.model_name, basedir = self.model_dir)
                    
                 
                    
                    if os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5'):
                        print('Loading checkpoint model')
                        model.load_weights(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5')
                    
                    model.train(X,Y, validation_data=(X_val,Y_val))
                    
                    axis_norm = (0,1,2)
                    print('Training StarDistModel model with unet backbone')
                    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_train)]
                    Y = [label(y) for y in tqdm(Binary_Y_train)]
                    
                    
                    assert len(X) > 1, "not enough training data"
                    rng = np.random.RandomState(42)
                    ind = rng.permutation(len(X))
                    n_val = max(1, int(round(0.15 * len(ind))))
                    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
                    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
                    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
                    print('number of images: %3d' % len(X))
                    print('- training:       %3d' % len(X_trn))
                    print('- validation:     %3d' % len(X_val))     
                 
                 
                    print(Config2D.__doc__)
                    conf = Config2D (
                      n_rays       = self.n_rays,
                      train_epochs = self.epochs,
                      train_learning_rate = self.learning_rate,
                      unet_n_depth = self.depth ,
                      train_patch_size = (self.PatchY,self.PatchX),
                      n_channel_in = 1,
                      train_checkpoint= self.model_dir + self.model_name +'.h5',
                      grid         = (2,2),
                      train_loss_weights=(1, 0.05),
                      use_gpu      = self.use_gpu
                      
                      )
                    print(conf)
                    vars(conf)
                     
                    
                    Starmodel = StarDist2D(conf, name=self.model_name, basedir=self.model_dir)
                    if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5'):
                      Starmodel.load_weights(  (self.model_dir + self.model_name + '/' + 'weights_now.h5'  ))
                     
                    Starmodel.train(X_trn, (Y_trn), validation_data=(X_val,(Y_val)), epochs = epochs)
                    Starmodel.optimize_thresholds(X_val, Y_val)

                 
                 
                 
         
         
        
         
         
         
         
         
         