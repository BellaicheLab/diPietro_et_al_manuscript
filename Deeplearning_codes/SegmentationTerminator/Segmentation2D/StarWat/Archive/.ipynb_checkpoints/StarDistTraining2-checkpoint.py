#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, unicode_literals, absolute_import, division

import sys
import numpy as np
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# get_ipython().run_line_magic('load_ext', 'tensorboard')

sys.path.append("../../Terminator")
from TerminatorUtils import StarDistDetection2D
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:


#Specify the location and name of the npz file for training and validation data
ImageDirectory = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/Training_Data_Sets/Training_Stardist/a_ComboData_Bin2/Original/*.tif'
LabelDirectory = '/run/user/1001/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/Training_Data_Sets/Training_Stardist/a_ComboData_Bin2/Integer/*.tif'

#Read and Write the h5 file, directory location and name
model_dir = '/run/media/sancere/DATA/Lucas_Model_to_use/Segmentation/'

copy_model_name = 'DrosophilaSegmentationSmartSeedsMatureElongated'
model_name = 'DrosophilaSegmentationSmartSeedsMatureMacroMiniElongatedLargePatchCyc5Cdk1Boris'


# In[ ]:


global Trainingmodel
#Initate training of the model

Trainingmodel = StarDistDetection2D(ImageDirectory  = ImageDirectory ,LabelDirectory = LabelDirectory, model_dir = model_dir, model_name = model_name,copy_model_name = copy_model_name, patch_size = (384, 384), n_rays = 512, epochs = 1000, use_gpu = True)

# IMPORTANT, previous patch size -> patch_size = (320, 128)

# In[ ]:




