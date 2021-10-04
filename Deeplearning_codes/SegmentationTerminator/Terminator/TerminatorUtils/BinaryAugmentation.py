#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:57:42 2019

@author: aimachine
"""


from __future__ import print_function, unicode_literals, absolute_import, division
from tqdm import tqdm
from glob import glob
from tifffile import imread
import elasticdeform
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
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
    
    
    
class BinaryAugmentation(object):



      def __init__(self, inputdir, maskdir, outputdir, outputmaskdir,  putNoise = False, rotate = False, deform = False, AppendName = "_"   ):    
        Path(outputdir).mkdir(exist_ok=True)
        Path(outputmaskdir).mkdir(exist_ok=True)
        self.inputdir = inputdir
        self.maskdir = maskdir
        self.outputdir = outputdir
        self.outputmaskdir = outputmaskdir
        self.putNoise = putNoise
        self.AppendName = AppendName
        self.rotate = rotate
        self.deform = deform
        #Perform tasks
        self.do_augmentation()
        
        
           
      def do_augmentation(self):
         """
         Performs data augmentation on directory of images and stores result with appropriate name in target directory images
         
         """
    
         axes = 'XY'
        
         #Read in all images
         Images = sorted(glob(self.inputdir + '/' +'*.tif')) 
         Masks = sorted(glob(self.maskdir + '/' +'*.tif')) 
         X = list(map(imread, Images))
         Y = list(map(imread, Masks))
        
         noisecount = 0
         rotatecount = 0
         origcount = 0     
         deformcount = 0
         
         for i in range(0, len(X)):
             
             #Resize movies
             
             mImage = X[i]
             mMask = Y[i]
             
         
             origcount = origcount + 1 
         
             save_tiff_imagej_compatible((self.outputdir  + str(origcount) + self.AppendName + '.tif'  ) , mImage, axes)  
             save_tiff_imagej_compatible((self.outputmaskdir  + str(origcount) + self.AppendName + '.tif'  ) , mMask, axes)  


             if(self.deform):
                 
                 deformcount = deformcount + 1
                 im_merge = np.concatenate((mImage[...,None], mMask[...,None]), axis=2)
                 deformCombo = random_deform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                 deformImage = deformCombo[...,0]
                 deformMask = deformCombo[...,1]
                 
                 save_tiff_imagej_compatible((self.outputdir + str(deformcount) + self.AppendName + 'Deform' + '.tif'  ) , deformImage, axes)
                 save_tiff_imagej_compatible((self.outputmaskdir + str(deformcount) + self.AppendName + 'Deform' +  '.tif'  ) , deformMask, axes)
            
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
                if(self.deform):
                    
                    im_merge = np.concatenate((mImage[...,None], mMask[...,None]), axis=2)
                    deformCombo = random_deform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                    deformRotated = deformCombo[...,0]
                    deformRotatedMask = deformCombo[...,1]
                
                    save_tiff_imagej_compatible((self.outputdir  + str(rotatecount)+ self.AppendName + 'RotationDeform' + '.tif'  ) , deformRotated, axes)
                    save_tiff_imagej_compatible((self.outputmaskdir  + str(rotatecount) + self.AppendName + 'RotationDeform'  + '.tif'  ) , deformRotatedMask, axes)
                
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
 



def random_deform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
 
               

    
def random_rotation(image, angle):
      
      rotatedimage = scipy.ndimage.interpolation.rotate(image, angle, mode = 'nearest', reshape = False)
      return rotatedimage

def random_noise(image, sigmaA = 0.2, sigmaB = 1, sigmaC = 2):
        
    
        noisyimageA = image

        noisyimageA = ndimage.gaussian_filter(image,sigmaA, mode = 'nearest')
        noisyimageB = ndimage.gaussian_filter(image,sigmaB, mode = 'nearest')
        noisyimageC = ndimage.gaussian_filter(image,sigmaC, mode = 'nearest')
              
        return noisyimageA, noisyimageB, noisyimageC
         
         
         
         
         
         
         
         
         
         
         