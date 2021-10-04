

#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, unicode_literals, absolute_import, division

#In[0]:

#import os
#import time

#TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TT_TrainingWideBin1'
#TimeCount = 0
#TimeThreshold = 3600*3
#while os.path.exists(TriggerName) == False and TimeCount < TimeThreshold :
#   time.sleep(60*5)
#   TimeCount = TimeCount + 60*5


# In[1]:


import sys
import os
sys.path.append('/home/sancere/anaconda3/envs/tensorflowGPU/lib/python3.6/site-packages/')
sys.path.append('../../Terminator/')


#To run the prediction on the GPU, else comment out this line to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import csbdeep
from csbdeep.utils.tf import limit_gpu_memory
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
#from IPython.display import clear_output
import glob
import cv2
from tifffile import imread
import tqdm
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons,dist_to_coord
from stardist.models import StarDist2D
from TerminatorUtils.helpers import normalizeFloat, OtsuThreshold2D, save_8bit_tiff_imagej_compatible, save_16bit_tiff_imagej_compatible,Integer_to_border,SeedStarDistWatershed
from TerminatorUtils.helpers import Prob_to_Binary, zero_pad, WatershedwithMask, MaxProjectDist, WatershedSmartCorrection,fill_label_holes, WatershedwithoutMask
from TerminatorUtils.helpers import save_tiff_imagej_compatible
from csbdeep.models import Config, CARE
np.random.seed(6)
lbl_cmap = random_label_cmap()

from skimage.morphology import remove_small_objects, thin
from skimage.morphology import skeletonize


import time 


# **Movie 1 + model importation**

# In[2]:


basedir = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/m_balakireva/Maria_Movie2Lucas/Nts4aCrop'

basedir_StardistWater_FijiMask = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/m_balakireva/Maria_Movie2Lucas/Nts4aCrop/StarWatDistance/'
basedir_StardistWater_FijiMask_RTGPipeline = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/m_balakireva/Maria_Movie2Lucas/Nts4aCrop/StarWatDistance_RTGPipeline/'

Model_Dir = '/run/media/sancere/DATA1/Lucas_Model_to_use/Segmentation/'    


# In[3]:


StardistModelName = 'DrosophilaSegmentationSmartSeedsMatureMacroMiniElongatedLargePatch'

modelstar = StarDist2D(config = None, name = StardistModelName, basedir = Model_Dir)
Model_is_for_Bin2 = False

# In[4]:


Path(basedir_StardistWater_FijiMask).mkdir(exist_ok = True)
Raw_path = os.path.join(basedir, '*tif') #tif or TIF be careful
axis_norm = (0,1)   # normalize channels independently
axes = 'XY'
filesRaw = glob.glob(Raw_path)
filesRaw.sort


# In[5]:


for fname in filesRaw:
 
    if os.path.exists((basedir_StardistWater_FijiMask + 'Markers_' + os.path.basename(os.path.splitext(fname)[0]))) == False :
        
            # Name = os.path.basename(os.path.splitext(fname)[0])
            x = imread(fname)
            y = normalizeFloat(x,1,99.8, axis = axis_norm)
            originalX = x.shape[0]
            originalY = x.shape[1]
        
            #zero pad aand 8bit conversion after normalization
            y = zero_pad(y, 32, 32)
            print('Processing file', fname)              
           
            #Stardist, label image, details, probability map, distance map 
            segmentationImage, details = modelstar.predict_instances(y)
            segmentationImage = segmentationImage[:originalX, :originalY]
            prob, dist = modelstar.predict(y)
            grid = modelstar.config.grid
            prob = cv2.resize(prob,  dsize=(prob.shape[1] * grid[1],prob.shape[0] * grid[0]))
            dist = cv2.resize(dist, dsize=(dist.shape[1] * grid[1] , dist.shape[0] * grid[0] ))
            maxdist = MaxProjectDist(dist)  
            maxdist = maxdist[:originalX, :originalY]  

            
            #Generation of masks    
            #Seeds from Stardist + Whatershed, segmentation on probability map 
            HomeMadeMask = cv2.imread(fname.replace('.tif','_Mask.png'))
            HomeMadeMask = HomeMadeMask[:,:,0]
            HomeMadeMask = zero_pad(HomeMadeMask, 32, 32)
            HomeMadeMask = HomeMadeMask[:originalX, :originalY]
            Watershed, markers = WatershedwithoutMask(maxdist, segmentationImage.astype('uint16'), modelstar.config.grid)

    
            #Save different method segmentations
            #save_8bit_tiff_imagej_compatible((basedir_StardistWater_FijiMask + 'StarWatDistance' + os.path.basename(os.path.splitext(fname)[0]) ) , StardistWater_FijiMask, axes)
            # save_tiff_imagej_compatible((basedir_StardistWater_FijiMask + 'DistMap_' + os.path.basename(os.path.splitext(fname)[0]) ) , maxdist, axes)
            save_tiff_imagej_compatible((basedir_StardistWater_FijiMask + 'Markers_' + os.path.basename(os.path.splitext(fname)[0]) ) , markers, axes)


    





