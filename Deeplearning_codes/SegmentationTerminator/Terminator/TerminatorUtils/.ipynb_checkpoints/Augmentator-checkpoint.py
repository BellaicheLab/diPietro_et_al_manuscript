#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:44:06 2019

@author: aimachine
"""

from __future__ import print_function, unicode_literals, absolute_import, division
from tqdm import tqdm
from glob import glob
from tifffile import imread
import numpy as np
import cv2
import random
from TerminatorUtils.helpers import save_tiff_imagej_compatible
import scipy

from scipy import ndimage

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
    
    
    
class Augmentation(object):



      def __init__(self, inputdir, maskdir, outputdir, outputmaskdir,resizeX, resizeY,   putNoise = False, rotate = False, AppendName = "_"   ):    
        Path(outputdir).mkdir(exist_ok=True)
        Path(outputmaskdir).mkdir(exist_ok=True)
        self.inputdir = inputdir
        self.maskdir = maskdir
        self.outputdir = outputdir
        self.resizeX = resizeX
        self.resizeY = resizeY
        self.outputmaskdir = outputmaskdir
        self.putNoise = putNoise
        self.AppendName = AppendName
        self.rotate = rotate
        #Perform tasks
        self.do_augmentation()
        
        
           
      def do_augmentation(self):
         """
         Performs data augmentation on directory of images and stores result with appropriate name in target directory images
         
         """
    
         axes = 'XY'
         HEIGHT = self.resizeX
         WIDTH = self.resizeY
         #Read in all images
         Images = sorted(glob(self.inputdir + '/' +'*.tif')) 
         Masks = sorted(glob(self.maskdir + '/' +'*.tif')) 
         X = list(map(imread, Images))
         Y = list(map(imread, Masks))
        
         noisecount = 0
         rotatecount = 0
         origcount = 0     
         
         for i in range(0, len(X)):
             
             #Resize movies
             image = X[i]
             mask = Y[i]
             mImage = cv2.resize(image, (HEIGHT, WIDTH), interpolation = cv2.INTER_LANCZOS4)
             mMask = cv2.resize(mask, (HEIGHT, WIDTH), interpolation = cv2.INTER_LANCZOS4)
             
         
             origcount = origcount + 1 
         
             save_tiff_imagej_compatible((self.outputdir  + str(origcount) + self.AppendName + '.tif'  ) , mImage, axes)  
             save_tiff_imagej_compatible((self.outputmaskdir  + str(origcount) + self.AppendName + '.tif'  ) , mMask, axes)  

             
                 
                 

            
             if(self.putNoise):
                 
                 noisecount = noisecount + 1
                 noiseA, noiseB, noiseC = random_noise(mImage)
                 
               
                 save_tiff_imagej_compatible((self.outputdir + str(noisecount) + self.AppendName + 'NoiseA' + '.tif'  ) , noiseA, axes)
                 save_tiff_imagej_compatible((self.outputmaskdir + str(noisecount) + self.AppendName + 'NoiseA' +  '.tif'  ) , mMask, axes)

                
                 save_tiff_imagej_compatible((self.outputdir + str(noisecount) + self.AppendName + 'NoiseB' + '.tif'  ) , noiseB, axes)
                 save_tiff_imagej_compatible((self.outputmaskdir  + str(noisecount) + self.AppendName + 'NoiseB' + '.tif'  ) , mMask, axes)

              
                 save_tiff_imagej_compatible((self.outputdir  + str(noisecount) + self.AppendName + 'NoiseC' + '.tif'  ) , noiseC, axes)
                 save_tiff_imagej_compatible((self.outputmaskdir  + str(noisecount) + self.AppendName + 'NoiseC' +  '.tif'  ) , mMask, axes)

         
         
             if(self.rotate):
                angle = random.uniform(-90, 90)
                #Make rotations on original and three noisy movies  
                rotate_orig = random_rotation(mImage, angle)
                mask_rotate_orig = random_rotation(mMask, angle)
                rotatecount = rotatecount + 1
                
           
                save_tiff_imagej_compatible((self.outputdir  + str(rotatecount)+ self.AppendName + 'RotationOrig' + '.tif'  ) , rotate_orig, axes)
                save_tiff_imagej_compatible((self.outputmaskdir  + str(rotatecount) + self.AppendName + 'RotationOrig'  + '.tif'  ) , mask_rotate_orig, axes)

                if(self.putNoise):
                  rotate_noiseA = random_rotation(noiseA, angle)
            
                  
                  save_tiff_imagej_compatible((self.outputdir + str(rotatecount) + self.AppendName + 'RotationSigmaA' + '.tif'  ) , rotate_noiseA, axes)
                  save_tiff_imagej_compatible((self.outputmaskdir + str(rotatecount) + self.AppendName + 'RotationSigmaA' + '.tif'  ) , mask_rotate_orig, axes)

                  rotate_noiseB = random_rotation(noiseB, angle)
            
                 
                  save_tiff_imagej_compatible((self.outputdir +  str(rotatecount) + self.AppendName + 'RotationSigmaB' + '.tif'  ) , rotate_noiseB, axes)
                  save_tiff_imagej_compatible((self.outputmaskdir + str(rotatecount) + self.AppendName + 'RotationSigmaB' + '.tif'  ) , mask_rotate_orig, axes)
 
                  rotate_noiseC = random_rotation(noiseC, angle)
            
                 
                  save_tiff_imagej_compatible((self.outputdir + str(rotatecount) + self.AppendName + 'RotationSigmaC' + '.tif'  ) , rotate_noiseC, axes)
                  save_tiff_imagej_compatible((self.outputmaskdir  + str(rotatecount) + self.AppendName + 'RotationSigmaC' + '.tif'  ) , mask_rotate_orig, axes)
 



            

    
def random_rotation(image, angle):
      
      rotatedimage = scipy.ndimage.interpolation.rotate(image, angle, mode = 'nearest', reshape = False)
      return rotatedimage



def random_noise(image, sigmaA = 0.2, sigmaB = 1, sigmaC = 2):
        
    
        noisyimageA = image

        noisyimageA = ndimage.gaussian_filter(image,sigmaA, mode = 'nearest')
        noisyimageB = ndimage.gaussian_filter(image,sigmaB, mode = 'nearest')
        noisyimageC = ndimage.gaussian_filter(image,sigmaC, mode = 'nearest')
              
        return noisyimageA, noisyimageB, noisyimageC
         
         
         
         
         
         
         
         
         
         
         