#!/usr/bin/env python
# coding: utf-8

# In[17]:


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import cv2

#To run the prediction on the GPU, else comment out this line to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from csbdeep.utils.tf import limit_gpu_memory
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sys.path.append('../../Terminator/')

#from IPython.display import clear_output
import glob
from tifffile import imread
import tqdm
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons,dist_to_coord
from stardist.models import StarDist2D
from TerminatorUtils.helpers import normalizeFloat, OtsuThreshold2D, save_8bit_tiff_imagej_compatible, save_16bit_tiff_imagej_compatible,Integer_to_border,SeedStarDistWatershed
from TerminatorUtils.helpers import Prob_to_Binary, zero_pad
from TerminatorUtils.helpers import save_tiff_imagej_compatible
from csbdeep.models import Config, CARE
np.random.seed(6)
lbl_cmap = random_label_cmap()

from skimage.morphology import remove_small_objects


# In[18]:


# basedir = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/Tests_on_CycE04'

# basedir_StardistWater_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/Tests_on_CycE04/StarWat_StarMask/'
# basedir_StardistWater_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/Tests_on_CycE04/StarWat_UnetMask/'
# basedir_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/Tests_on_CycE04/StarMask/'
# basedir_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/Tests_on_CycE04/UnetMask/'

Model_Dir = '/media/sancere/Newton_Volume_1/CurieDeepLearningModels/DrosophilaSegmentation/'    #temporary path 


# In[19]:


StardistModelName = 'DrosophilaSegmentationSmartSeedsMatureElongated'
UnetMaskModelName = 'DrosophilaMaskSegmentationCARE'

modelstar = StarDist2D(config = None, name = StardistModelName, basedir = Model_Dir)
modelmask = CARE(config = None, name = UnetMaskModelName, basedir = Model_Dir)


# In[20]:


# Path(basedir_StardistWater_StardistMask).mkdir(exist_ok = True)
# Path(basedir_StardistWater_UnetMask).mkdir(exist_ok = True)
# Path(basedir_StardistMask).mkdir(exist_ok = True)
# Path(basedir_UnetMask).mkdir(exist_ok = True)



# Raw_path = os.path.join(basedir, '*tif') #tif or TIF be careful

# axis_norm = (0,1)   # normalize channels independently
# axes = 'XY'

# filesRaw = glob.glob(Raw_path)
# # filesRaw.sort


# # In[ ]:



# count = 0
# for fname in filesRaw:
 
#     if os.path.exists(basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) == False     or os.path.exists((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False :
#             count = count + 1
#             #if count%20 == 0:
#                 #clear_output()
           
#             # Name = os.path.basename(os.path.splitext(fname)[0])
#             x = imread(fname)
#             y = normalizeFloat(x,1,99.8, axis = axis_norm)
#             originalX = x.shape[0]
#             originalY = x.shape[1]
        
#             #zero pad after normalization
#             y = zero_pad(y, 32, 32)
            
#             print('Processing file', fname)              
           
#             #Stardist, label image, details, probability map, distance map 
#             segmentationImage, details = modelstar.predict_instances(y)
#             prob, dist = modelstar.predict(y)
#             grid = modelstar.config.grid
#             prob = cv2.resize(prob,  dsize=(prob.shape[1] * grid[1],prob.shape[0] * grid[0]))
#             print('Processing file 2', fname) 
            
#             #Seeds from Stardist + Whatershed, segmentation on probability map 
#             Watershed, markers = SeedStarDistWatershed(prob,details['points'],modelstar.config.grid )
#             print('Processing file 2b', fname) 
            
#             #Convert Integer image to binary
#             Binary_Watershed = Integer_to_border(Watershed[:originalX, :originalY])
                          
#             #Generation of masks               
#             StardistMask = Prob_to_Binary(prob[:originalX, :originalY], segmentationImage[:originalX, :originalY])    
#             UnetMask = modelmask.predict(x, axes, n_tiles = (1, 2))
#             UnetMask = UnetMask.astype('uint8')
#             thresh = threshold_otsu(UnetMask)
#             UnetMask = UnetMask > thresh
#             UnetMask = remove_small_objects(UnetMask, 5000)
#             UnetMask = UnetMask.astype('uint8')
#             print('Processing file 3', fname) 
                                                                   
#             #Create StardistWater_StardistMask and StardistWater_UnetMask with logical and operation            
#             StardistWater_StardistMask = np.logical_and(StardistMask, Binary_Watershed)
#             StardistWater_UnetMask = np.logical_and(UnetMask, Binary_Watershed)
#             print('Processing file 4', fname) 
                                
#             #Save different method segmentations
#             save_8bit_tiff_imagej_compatible((basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) ,  StardistWater_StardistMask, axes)
#             save_8bit_tiff_imagej_compatible((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , StardistWater_UnetMask, axes)
#             print('Processing file 5', fname) 
#             save_16bit_tiff_imagej_compatible((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0])) , StardistMask, axes)
#             save_8bit_tiff_imagej_compatible((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , UnetMask, axes)


# # In[ ]:


# basedir = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE03'

# basedir_StardistWater_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE03/StarWat_StarMask/'
# basedir_StardistWater_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE03/StarWat_UnetMask/'
# basedir_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE03/StarMask/'
# basedir_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE03/UnetMask/'

# Model_Dir = '/media/sancere/Newton_Volume_1/CurieDeepLearningModelsCopies_forparaTrainPred'    #temporary path 


# # In[ ]:


# Path(basedir_StardistWater_StardistMask).mkdir(exist_ok = True)
# Path(basedir_StardistWater_UnetMask).mkdir(exist_ok = True)
# Path(basedir_StardistMask).mkdir(exist_ok = True)
# Path(basedir_UnetMask).mkdir(exist_ok = True)



# Raw_path = os.path.join(basedir, '*tif') #tif or TIF be careful

# axis_norm = (0,1)   # normalize channels independently
# axes = 'XY'

# filesRaw = glob.glob(Raw_path)
# # filesRaw.sort


# # In[ ]:



# count = 0
# for fname in filesRaw:
 
#     if os.path.exists(basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) == False     or os.path.exists((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False :
#             count = count + 1
#             #if count%20 == 0:
#                 #clear_output()
           
#             # Name = os.path.basename(os.path.splitext(fname)[0])
#             x = imread(fname)
#             y = normalizeFloat(x,1,99.8, axis = axis_norm)
#             originalX = x.shape[0]
#             originalY = x.shape[1]
        
#             #zero pad after normalization
#             y = zero_pad(y, 32, 32)      
            
#             print('Processing file', fname)              
           
#             #Stardist, label image, details, probability map, distance map 
#             segmentationImage, details = modelstar.predict_instances(y)
#             prob, dist = modelstar.predict(y)
#             grid = modelstar.config.grid
#             prob = cv2.resize(prob,  dsize=(prob.shape[1] * grid[1],prob.shape[0] * grid[0]))
#             print('Processing file 2', fname) 
            
#             #Seeds from Stardist + Whatershed, segmentation on probability map 
#             Watershed, markers = SeedStarDistWatershed(prob,details['points'],modelstar.config.grid )
#             print('Processing file 2b', fname) 
            
#             #Convert Integer image to binary
#             Binary_Watershed = Integer_to_border(Watershed[:originalX, :originalY])
                          
#             #Generation of masks               
#             StardistMask = Prob_to_Binary(prob[:originalX, :originalY], segmentationImage[:originalX, :originalY])    
#             UnetMask = modelmask.predict(x, axes, n_tiles = (1, 2))
#             UnetMask = UnetMask.astype('uint8')
#             thresh = threshold_otsu(UnetMask)
#             UnetMask = UnetMask > thresh
#             UnetMask = remove_small_objects(UnetMask, 5000)
#             UnetMask = UnetMask.astype('uint8')
#             print('Processing file 3', fname) 
                                                                   
#             #Create StardistWater_StardistMask and StardistWater_UnetMask with logical and operation            
#             StardistWater_StardistMask = np.logical_and(StardistMask, Binary_Watershed)
#             StardistWater_UnetMask = np.logical_and(UnetMask, Binary_Watershed)
#             print('Processing file 4', fname) 
                                
#             #Save different method segmentations
#             save_8bit_tiff_imagej_compatible((basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) ,  StardistWater_StardistMask, axes)
#             save_8bit_tiff_imagej_compatible((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , StardistWater_UnetMask, axes)
#             print('Processing file 5', fname) 
#             save_16bit_tiff_imagej_compatible((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0])) , StardistMask, axes)
#             save_8bit_tiff_imagej_compatible((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , UnetMask, axes)


# In[ ]:


basedir = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE04'

basedir_StardistWater_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE04/StarWat_StarMask/'
basedir_StardistWater_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE04/StarWat_UnetMask/'
basedir_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE04/StarMask/'
basedir_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE04/UnetMask/'

Model_Dir = '/media/sancere/Newton_Volume_1/CurieDeepLearningModelsCopies_forparaTrainPred'    #temporary path 


# In[ ]:


Path(basedir_StardistWater_StardistMask).mkdir(exist_ok = True)
Path(basedir_StardistWater_UnetMask).mkdir(exist_ok = True)
Path(basedir_StardistMask).mkdir(exist_ok = True)
Path(basedir_UnetMask).mkdir(exist_ok = True)



Raw_path = os.path.join(basedir, '*tif') #tif or TIF be careful

axis_norm = (0,1)   # normalize channels independently
axes = 'XY'

filesRaw = glob.glob(Raw_path)
# filesRaw.sort


# In[ ]:



count = 0
for fname in filesRaw:
 
    if os.path.exists(basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) == False     or os.path.exists((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False :
            count = count + 1
            #if count%20 == 0:
                #clear_output()
           
            # Name = os.path.basename(os.path.splitext(fname)[0])
            x = imread(fname)
            y = normalizeFloat(x,1,99.8, axis = axis_norm)
            originalX = x.shape[0]
            originalY = x.shape[1]
        
            #zero pad after normalization
            y = zero_pad(y, 32, 32)  
            
            print('Processing file', fname)              
           
            #Stardist, label image, details, probability map, distance map 
            segmentationImage, details = modelstar.predict_instances(y)
            prob, dist = modelstar.predict(y)
            grid = modelstar.config.grid
            prob = cv2.resize(prob,  dsize=(prob.shape[1] * grid[1],prob.shape[0] * grid[0]))
            print('Processing file 2', fname) 
            
            #Seeds from Stardist + Whatershed, segmentation on probability map 
            Watershed, markers = SeedStarDistWatershed(prob,details['points'],modelstar.config.grid )
            print('Processing file 2b', fname) 
            
            #Convert Integer image to binary
            Binary_Watershed = Integer_to_border(Watershed[:originalX, :originalY])
                          
            #Generation of masks               
            StardistMask = Prob_to_Binary(prob[:originalX, :originalY], segmentationImage[:originalX, :originalY])    
            UnetMask = modelmask.predict(x, axes, n_tiles = (1, 2))
            UnetMask = UnetMask.astype('uint8')
            thresh = threshold_otsu(UnetMask)
            UnetMask = UnetMask > thresh
            UnetMask = remove_small_objects(UnetMask, 5000)
            UnetMask = UnetMask.astype('uint8')
            print('Processing file 3', fname) 
                                                                   
            #Create StardistWater_StardistMask and StardistWater_UnetMask with logical and operation            
            StardistWater_StardistMask = np.logical_and(StardistMask, Binary_Watershed)
            StardistWater_UnetMask = np.logical_and(UnetMask, Binary_Watershed)
            print('Processing file 4', fname) 
                                
            #Save different method segmentations
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) ,  StardistWater_StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , StardistWater_UnetMask, axes)
            print('Processing file 5', fname) 
            save_16bit_tiff_imagej_compatible((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0])) , StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , UnetMask, axes)


# In[ ]:


basedir = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE05'

basedir_StardistWater_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE05/StarWat_StarMask/'
basedir_StardistWater_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE05/StarWat_UnetMask/'
basedir_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE05/StarMask/'
basedir_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromVictoire2Lucas/CycE05/UnetMask/'

Model_Dir = '/media/sancere/Newton_Volume_1/CurieDeepLearningModelsCopies_forparaTrainPred'    #temporary path 


# In[ ]:


Path(basedir_StardistWater_StardistMask).mkdir(exist_ok = True)
Path(basedir_StardistWater_UnetMask).mkdir(exist_ok = True)
Path(basedir_StardistMask).mkdir(exist_ok = True)
Path(basedir_UnetMask).mkdir(exist_ok = True)



Raw_path = os.path.join(basedir, '*tif') #tif or TIF be careful

axis_norm = (0,1)   # normalize channels independently
axes = 'XY'

filesRaw = glob.glob(Raw_path)
# filesRaw.sort


# In[ ]:


count = 0
for fname in filesRaw:
 
    if os.path.exists(basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) == False     or os.path.exists((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False :
            count = count + 1
            #if count%20 == 0:
                #clear_output()
           
            # Name = os.path.basename(os.path.splitext(fname)[0])
            x = imread(fname)
            y = normalizeFloat(x,1,99.8, axis = axis_norm)
            originalX = x.shape[0]
            originalY = x.shape[1]
        
            #zero pad after normalization
            y = zero_pad(y, 32, 32)  
            
            print('Processing file', fname)              
           
            #Stardist, label image, details, probability map, distance map 
            segmentationImage, details = modelstar.predict_instances(y)
            prob, dist = modelstar.predict(y)
            grid = modelstar.config.grid
            prob = cv2.resize(prob,  dsize=(prob.shape[1] * grid[1],prob.shape[0] * grid[0]))
            print('Processing file 2', fname) 
            
            #Seeds from Stardist + Whatershed, segmentation on probability map 
            Watershed, markers = SeedStarDistWatershed(prob,details['points'],modelstar.config.grid )
            print('Processing file 2b', fname) 
            
            #Convert Integer image to binary
            Binary_Watershed = Integer_to_border(Watershed[:originalX, :originalY])
                          
            #Generation of masks               
            StardistMask = Prob_to_Binary(prob[:originalX, :originalY], segmentationImage[:originalX, :originalY])    
            UnetMask = modelmask.predict(x, axes, n_tiles = (1, 2))
            UnetMask = UnetMask.astype('uint8')
            thresh = threshold_otsu(UnetMask)
            UnetMask = UnetMask > thresh
            UnetMask = remove_small_objects(UnetMask, 5000)
            UnetMask = UnetMask.astype('uint8')
            print('Processing file 3', fname) 
                                                                   
            #Create StardistWater_StardistMask and StardistWater_UnetMask with logical and operation            
            StardistWater_StardistMask = np.logical_and(StardistMask, Binary_Watershed)
            StardistWater_UnetMask = np.logical_and(UnetMask, Binary_Watershed)
            print('Processing file 4', fname) 
                                
            #Save different method segmentations
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) ,  StardistWater_StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , StardistWater_UnetMask, axes)
            print('Processing file 5', fname) 
            save_16bit_tiff_imagej_compatible((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0])) , StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , UnetMask, axes)


# In[ ]:


basedir = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromAude2Lucas/Movies_Analyses'

basedir_StardistWater_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromAude2Lucas/Movies_Analyses/StarWat_StarMask/'
basedir_StardistWater_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromAude2Lucas/Movies_Analyses/StarWat_UnetMask/'
basedir_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromAude2Lucas/Movies_Analyses/StarMask/'
basedir_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/FromAude2Lucas/Movies_Analyses/UnetMask/'

Model_Dir = '/media/sancere/Newton_Volume_1/CurieDeepLearningModelsCopies_forparaTrainPred'    #temporary path 


# In[ ]:


Path(basedir_StardistWater_StardistMask).mkdir(exist_ok = True)
Path(basedir_StardistWater_UnetMask).mkdir(exist_ok = True)
Path(basedir_StardistMask).mkdir(exist_ok = True)
Path(basedir_UnetMask).mkdir(exist_ok = True)



Raw_path = os.path.join(basedir, '*tif') #tif or TIF be careful

axis_norm = (0,1)   # normalize channels independently
axes = 'XY'

filesRaw = glob.glob(Raw_path)
# filesRaw.sort


# In[ ]:


count = 0
for fname in filesRaw:
 
    if os.path.exists(basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) == False     or os.path.exists((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False :
            count = count + 1
            #if count%20 == 0:
                #clear_output()
           
            # Name = os.path.basename(os.path.splitext(fname)[0])
            x = imread(fname)
            y = normalizeFloat(x,1,99.8, axis = axis_norm)
            originalX = x.shape[0]
            originalY = x.shape[1]
        
            #zero pad after normalization
            y = zero_pad(y, 32, 32)  
            
            print('Processing file', fname)              
           
            #Stardist, label image, details, probability map, distance map 
            segmentationImage, details = modelstar.predict_instances(y)
            prob, dist = modelstar.predict(y)
            grid = modelstar.config.grid
            prob = cv2.resize(prob,  dsize=(prob.shape[1] * grid[1],prob.shape[0] * grid[0]))
            print('Processing file 2', fname) 
            
            #Seeds from Stardist + Whatershed, segmentation on probability map 
            Watershed, markers = SeedStarDistWatershed(prob,details['points'],modelstar.config.grid )
            print('Processing file 2b', fname) 
            
            #Convert Integer image to binary
            Binary_Watershed = Integer_to_border(Watershed[:originalX, :originalY])
                          
            #Generation of masks               
            StardistMask = Prob_to_Binary(prob[:originalX, :originalY], segmentationImage[:originalX, :originalY])    
            UnetMask = modelmask.predict(x, axes, n_tiles = (1, 2))
            UnetMask = UnetMask.astype('uint8')
            thresh = threshold_otsu(UnetMask)
            UnetMask = UnetMask > thresh
            UnetMask = remove_small_objects(UnetMask, 5000)
            UnetMask = UnetMask.astype('uint8')
            print('Processing file 3', fname) 
                                                                   
            #Create StardistWater_StardistMask and StardistWater_UnetMask with logical and operation            
            StardistWater_StardistMask = np.logical_and(StardistMask, Binary_Watershed)
            StardistWater_UnetMask = np.logical_and(UnetMask, Binary_Watershed)
            print('Processing file 4', fname) 
                                
            #Save different method segmentations
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) ,  StardistWater_StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , StardistWater_UnetMask, axes)
            print('Processing file 5', fname) 
            save_16bit_tiff_imagej_compatible((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0])) , StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , UnetMask, axes)


# In[ ]:


basedir = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/v_cachoux/DIAP_c1_mov5'

basedir_StardistWater_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/v_cachoux/DIAP_c1_mov5/StarWat_StarMask/'
basedir_StardistWater_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/v_cachoux/DIAP_c1_mov5/StarWat_UnetMask/'
basedir_StardistMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/v_cachoux/DIAP_c1_mov5/StarMask/'
basedir_UnetMask = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/v_cachoux/DIAP_c1_mov5/UnetMask/'

Model_Dir = '/media/sancere/Newton_Volume_1/CurieDeepLearningModelsCopies_forparaTrainPred'    #temporary path 


# In[ ]:


Path(basedir_StardistWater_StardistMask).mkdir(exist_ok = True)
Path(basedir_StardistWater_UnetMask).mkdir(exist_ok = True)
Path(basedir_StardistMask).mkdir(exist_ok = True)
Path(basedir_UnetMask).mkdir(exist_ok = True)



Raw_path = os.path.join(basedir, '*tif') #tif or TIF be careful

axis_norm = (0,1)   # normalize channels independently
axes = 'XY'

filesRaw = glob.glob(Raw_path)
# filesRaw.sort


# In[ ]:


count = 0
for fname in filesRaw:
 
    if os.path.exists(basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) == False     or os.path.exists((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False     or os.path.exists((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]))) == False :
            count = count + 1
            #if count%20 == 0:
                #clear_output()
           
            # Name = os.path.basename(os.path.splitext(fname)[0])
            x = imread(fname)
            y = normalizeFloat(x,1,99.8, axis = axis_norm)
            originalX = x.shape[0]
            originalY = x.shape[1]
        
            #zero pad after normalization
            y = zero_pad(y, 32, 32)  
            
            print('Processing file', fname)              
           
            #Stardist, label image, details, probability map, distance map 
            segmentationImage, details = modelstar.predict_instances(y)
            prob, dist = modelstar.predict(y)
            grid = modelstar.config.grid
            prob = cv2.resize(prob,  dsize=(prob.shape[1] * grid[1],prob.shape[0] * grid[0]))
            print('Processing file 2', fname) 
            
            #Seeds from Stardist + Whatershed, segmentation on probability map 
            Watershed, markers = SeedStarDistWatershed(prob,details['points'],modelstar.config.grid )
            print('Processing file 2b', fname) 
            
            #Convert Integer image to binary
            Binary_Watershed = Integer_to_border(Watershed[:originalX, :originalY])
                          
            #Generation of masks               
            StardistMask = Prob_to_Binary(prob[:originalX, :originalY], segmentationImage[:originalX, :originalY])    
            UnetMask = modelmask.predict(x, axes, n_tiles = (1, 2))
            UnetMask = UnetMask.astype('uint8')
            thresh = threshold_otsu(UnetMask)
            UnetMask = UnetMask > thresh
            UnetMask = remove_small_objects(UnetMask, 5000)
            UnetMask = UnetMask.astype('uint8')
            print('Processing file 3', fname) 
                                                                   
            #Create StardistWater_StardistMask and StardistWater_UnetMask with logical and operation            
            StardistWater_StardistMask = np.logical_and(StardistMask, Binary_Watershed)
            StardistWater_UnetMask = np.logical_and(UnetMask, Binary_Watershed)
            print('Processing file 4', fname) 
                                
            #Save different method segmentations
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_StardistMask + 'StarWat_StarMask_' + os.path.basename(os.path.splitext(fname)[0])) ,  StardistWater_StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_StardistWater_UnetMask + 'StarWat_UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , StardistWater_UnetMask, axes)
            print('Processing file 5', fname) 
            save_16bit_tiff_imagej_compatible((basedir_StardistMask + 'StarMask_' + os.path.basename(os.path.splitext(fname)[0])) , StardistMask, axes)
            save_8bit_tiff_imagej_compatible((basedir_UnetMask + 'UnetMask_' + os.path.basename(os.path.splitext(fname)[0]) ) , UnetMask, axes)


# In[ ]:




