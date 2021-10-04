#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:07:18 2019

@author: aimachine
"""

import numpy as np
from .helpers import normalizeFloat
from glob import glob
from tifffile import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
from stardist import random_label_cmap
np.random.seed(42)
lbl_cmap = random_label_cmap()
import collections
from itertools import chain
from collections import namedtuple
from skimage.transform import downscale_local_mean
from stardist import fill_label_holes
from sklearn.model_selection import train_test_split
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
from csbdeep.data import  create_patches, RawData,no_background_patches,norm_percentiles,shuffle_inplace,sample_patches_from_multiple_stacks
from csbdeep.data.transform import Transform, permute_axes
from csbdeep.utils import _raise, consume, compose, axes_dict, axes_check_and_normalize
import sys, warnings
def create_downsample_patches(
        raw_data,
        patch_size,
        n_patches_per_image,
        patch_axes    = None,
        save_file     = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        normalization = norm_percentiles(),
        shuffle       = True,
        verbose       = True,
        downsample_factor = 2
    ):
    """Create normalized training data to be used for neural network training.
    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.
    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images, `n_channels` will be 1.
    Raises
    ------
    ValueError
        Various reasons.
    Example
    -------
    >>> raw_data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> X, Y, XY_axes = create_patches(raw_data, patch_size=(32,128,128), n_patches_per_image=16)
    Todo
    ----
    - Save created patches directly to disk using :class:`numpy.memmap` or similar?
      Would allow to work with large data that doesn't fit in memory.
    """
    ## images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())


    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches*np.prod(patch_size) * 4

    ## memory check
    _memory_check(n_required_memory_bytes)

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)

    sys.stdout.flush()

    ## sample patches from each pair of transformed raw images
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)

    for i, (x,y,_axes,mask) in tqdm(enumerate(image_pairs),total=n_images,disable=(not verbose)):
        if i >= n_images:
            warnings.warn('more raw images (or transformations thereof) than expected, skipping excess images.')
            break
        if i==0:
            axes = axes_check_and_normalize(_axes,len(patch_size))
            channel = axes_dict(axes)['C']
        # checks
        # len(axes) >= x.ndim or _raise(ValueError())
        axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))

        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_patches_per_image, mask, patch_filter)

        s = slice(i*n_patches_per_image,(i+1)*n_patches_per_image)
        X[s], Y[s] = normalization(_X,_Y, x,y,mask,channel)

    if shuffle:
        shuffle_inplace(X,Y)

    axes = 'SC'+axes.replace('C','')
    X = downscale_local_mean(X, (downsample_factor, downsample_factor))
    Y = downscale_local_mean(Y, (downsample_factor, downsample_factor))
    
    print('Downsample Images',X.shape, Y.shape)
    if channel is None:
        X = np.expand_dims(X,1)
        Y = np.expand_dims(Y,1)
    else:
        X = np.moveaxis(X, 1+channel, 1)
        Y = np.moveaxis(Y, 1+channel, 1)
  
    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X,Y,axes
class LocalRawData(namedtuple('RawData' ,('generator' ,'size' ,'description'))):
    """:func:`collections.namedtuple` with three fields: `generator`, `size`, and `description`.
    Parameters
    ----------
    generator : function
        Function without arguments that returns a generator that yields tuples `(x,y,axes,mask)`,
        where `x` is a source image (e.g., with low SNR) with `y` being the corresponding target image
        (e.g., with high SNR); `mask` can either be `None` or a boolean array that denotes which
        pixels are eligible to extracted in :func:`create_patches`. Note that `x`, `y`, and `mask`
        must all be of type :class:`numpy.ndarray` and are assumed to have the same shape, where the
        string `axes` indicates the order and presence of axes of all three arrays.
    size : int
        Number of tuples that the `generator` will yield.
    description : str
        Textual description of the raw data.
    """

    @staticmethod
    def from_folder(basepath, source_dirs, target_dir, axes='CZYX',  downsample_factor = 2, pattern='*.tif*'):
        """Get pairs of corresponding TIFF images read from folders.
        Two images correspond to each other if they have the same file name, but are located in different folders.
        Parameters
        ----------
        basepath : str
            Base folder that contains sub-folders with images.
        source_dirs : list or tuple
            List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
        target_dir : str
            Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
        axes : str
            Semantics of axes of loaded images (assumed to be the same for all images).
        pattern : str
            Glob-style pattern to match the desired TIFF images.
        Returns
        -------
        RawData
            :obj:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
            The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
            `source_dirs` and `y` is the corresponding image from the `target_dir`;
            `mask` is set to `None`.
        Raises
        ------
        FileNotFoundError
            If an image found in a `source_dir` does not exist in `target_dir`.
        Example
        --------
        >>> !tree data
        data
        ├── GT
        │   ├── imageA.tif
        │   ├── imageB.tif
        │   └── imageC.tif
        ├── source1
        │   ├── imageA.tif
        │   └── imageB.tif
        └── source2
            ├── imageA.tif
            └── imageC.tif
        >>> data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='YX')
        >>> n_images = data.size
        >>> for source_x, target_y, axes, mask in data.generator():
        ...     pass
        """
        p = Path(basepath)
        pairs = [(f, p/target_dir/f.name) for f in chain(*((p/source_dir).glob(pattern) for source_dir in source_dirs))]
        len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any images."))
        consume(t.exists() or _raise(FileNotFoundError(t)) for s,t in pairs)
        axes = axes_check_and_normalize(axes)
        n_images = len(pairs)
        description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=basepath, s=list(source_dirs),
                                                                                          o=target_dir, a=axes, pt=pattern)

        def _gen():
            for fx, fy in pairs:
                x, y = imread(str(fx)), imread(str(fy))
                x = downscale_local_mean(x, (downsample_factor, downsample_factor))
                y = downscale_local_mean(y, (downsample_factor, downsample_factor))
                len(axes) >= x.ndim or _raise(ValueError())
                yield x, y, axes[-x.ndim:], None

        return RawData(_gen, n_images, description)



    @staticmethod
    def from_arrays(X, Y, axes='CZYX'):
        """Get pairs of corresponding images from numpy arrays."""

        def _gen():
            for x, y in zip(X ,Y):
                len(axes) >= x.ndim or _raise(ValueError())
                yield x, y, axes[-x.ndim:], None

        return RawData(_gen, len(X), "numpy array")
def generate_2D_patch_training_data(BaseDirectory, SaveNpzDirectory, SaveName, patch_size = (512,512), n_patches_per_image = 64, transforms = None):

    
    raw_data = RawData.from_folder (
    basepath    = BaseDirectory,
    source_dirs = ['AllRaws'],
    target_dir  = 'AllMasks',
    axes        = 'YX',
    )
    
    X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = patch_size,
    n_patches_per_image = n_patches_per_image,
    transforms = transforms,
    save_file   = SaveNpzDirectory + SaveName,
    )
def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024*1024**2):
    try:
        # raise ImportError
        import psutil
        mem = psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.available
        if mem_frac > 1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac > thresh_free_frac:
            print('Warning: will use at least %.0f MB (%.1f%%) of available memory.\n' % (n_required_memory_bytes/1024**2,100*mem_frac), file=sys.stderr)
            sys.stderr.flush()
    except ImportError:
        if n_required_memory_bytes > thresh_abs_bytes:
            print('Warning: will use at least %.0f MB of memory.\n' % (n_required_memory_bytes/1024**2), file=sys.stderr)
            sys.stderr.flush()   
def generate_downsample_2D_patch_training_data(BaseDirectory, SaveNpzDirectory, SaveName, patch_size = (512,512), n_patches_per_image = 64, downsample_factor = 2, transforms = None):

    
    raw_data = LocalRawData.from_folder (
    basepath    = BaseDirectory,
    source_dirs = ['MoreOriginal'],
    target_dir  = 'MoreBinary',
    axes        = 'YX',
    downsample_factor = downsample_factor
    )
    
    X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = patch_size,
    n_patches_per_image = n_patches_per_image,
    transforms = transforms,
    save_file           = SaveNpzDirectory + SaveName
    )

def generate_2D_training_data(Imagedir, Labeldir, SaveNpzDirectory, SaveName, SaveNameVal,shapeX, shapeY, display = 0):
    
    
    
    
    axes = 'SXYC'
    save_data(axes, Imagedir, Labeldir, SaveNpzDirectory, SaveName, SaveNameVal,shapeX, shapeY, display, None)
    
    
def generate_3D_training_data(Imagedir, Labeldir, SaveNpzDirectory, SaveName, SaveNameVal,shapeX, shapeY, display = 0, displayZ = 0):
    
    
    assert len(Imagedir) == len(Labeldir)
    
    axes = 'SZXYC'
    save_data(axes, Imagedir, Labeldir, SaveNpzDirectory, SaveName, SaveNameVal,shapeX, shapeY, display, displayZ)

 
def save_data(axes, Imagedir, Labeldir, SaveNpzDirectory, SaveName, SaveNameVal,shapeX, shapeY, display, displayZ):
 

    data = []
    masks = []
    

                 
    Y = sorted(glob(Labeldir + '/' + '*.tif'))
    print(Y)
    LabelImages = list(map(imread, Y))
    FilledLabelImages = [y for y in tqdm(LabelImages)]
    
    X = sorted(glob(Imagedir + '/'  + '*.tif'))
    Images = list(map(imread, X))
    NormalizeImages = [normalizeFloat(image,1,99.8) for image in tqdm(Images)]
    


    assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
    
    for i in range(0, len(NormalizeImages)):
        
       X = NormalizeImages[i]
       Y = FilledLabelImages[i]
       Xbig = np.zeros([shapeX, shapeY]) 
       Xbig[0:shapeX, 0:shapeY] = X
        
       Ybig = np.zeros([shapeX, shapeY]) 
       Ybig[0:shapeX, 0:shapeY] = Y
        
       Xbig = np.expand_dims(Xbig, -1)
       Ybig = np.expand_dims(Ybig, -1)
       data.append(Xbig)
       masks.append(Ybig)
       
       
    data = np.array(data)
    masks = np.array(masks)
    
    if display is not None and display < len(NormalizeImages):
        if displayZ == None:
           plt.figure(figsize=(16,10))
           plt.subplot(121); plt.imshow(NormalizeImages[display],cmap='gray');   plt.axis('off'); plt.title('Raw image')
           plt.subplot(122); plt.imshow(FilledLabelImages[display],cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
           None;
        else:
           plt.figure(figsize=(16,10))
           plt.subplot(121); plt.imshow(NormalizeImages[display][displayZ,:,:],cmap='gray');   plt.axis('off'); plt.title('Raw image')
           plt.subplot(122); plt.imshow(FilledLabelImages[display][displayZ,:,:],cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
           None;
            
    
    print(data.shape, masks.shape)
    
    traindata, validdata, trainlabel, validlabel = train_test_split(data, masks, train_size = 0.95, test_size = 0.05, shuffle = True)
    
    save_full_training_data(SaveNpzDirectory, SaveName, traindata, trainlabel, axes)
    save_full_training_data(SaveNpzDirectory, SaveNameVal, validdata, validlabel, axes)

                
                
def _raise(e):
    raise e

               
def save_training_data(directory, filename, data, label, sublabel, axes):
    """Save training data in ``.npz`` format."""
  
    
  
    len(axes) == data.ndim or _raise(ValueError())
    np.savez(directory + filename, data = data, label = label, label2 = sublabel, axes = axes)
    
    
def save_full_training_data(directory, filename, data, label, axes):
    """Save training data in ``.npz`` format."""
  

    len(axes) == data.ndim or _raise(ValueError())
    np.savez(directory + filename, data = data, label = label, axes = axes)     
    
    
# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)



           
    