#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, unicode_literals, absolute_import, division


# In[0]:

import os
import time

TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TTT'
TimeCount = 0
TimeThreshold = 3600*0
while os.path.exists(TriggerName) == False and TimeCount < TimeThreshold :
   time.sleep(60*5)
   TimeCount = TimeCount + 60*5
    

# In[1]:



import sys
import os
import cv2
#To run the prediction on the GPU, else comment out this line to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from csbdeep.utils.tf import limit_gpu_memory

import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sys.path.append('../../Terminator/')
import glob
from tifffile import imread
from PIL import Image

from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons,dist_to_coord
from stardist.models import StarDist2D
from TerminatorUtils.helpers import normalizeFloat, OtsuThreshold2D, save_8bit_tiff_imagej_compatible,Integer_to_border,SeedStarDistWatershed
from TerminatorUtils.helpers import Prob_to_Binary
from TerminatorUtils.helpers import save_tiff_imagej_compatible
from csbdeep.models import Config, CARE
np.random.seed(6)
lbl_cmap = random_label_cmap()

import png


# **Movie 1**
        
# In[2]:


basedir = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/m_gracia/20210130_compression/pupe1/Projected/Rotated'
basedirMaskResults = basedir + '/Masks/'



# In[3]:


Model_Dir = '/run/media/sancere/DATA/Lucas_Model_to_use/Mask_Generator/'
ModelName = 'Masks_Generator_Mix_onlyequalized'

model = CARE(config = None, name = ModelName, basedir = Model_Dir)


# In[4]:


Path(basedirMaskResults).mkdir(exist_ok = True)


# In[5]:


Raw_path = os.path.join(basedir, '*tif') #be careful tif TIF 
axes = 'YX'
filesRaw = glob.glob(Raw_path)
filesRaw.sort

for fname in filesRaw:
        x = imread(fname)
        print('Saving file' +  basedirMaskResults + '%s_' + os.path.basename(fname))
        mask = model.predict(x, axes, n_tiles = (1, 2)) 
        Name = os.path.basename(os.path.splitext(fname)[0])
        #png.from_array(mask, mode="L").save(basedirMaskResults + Name +'.png') 
        save_tiff_imagej_compatible((basedirMaskResults + Name), mask, axes)
        
from csbdeep.utils import Path

TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TTMaskMelanie1'
Path(TriggerName).mkdir(exist_ok = True)

