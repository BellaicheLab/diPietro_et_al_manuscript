#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../Terminator/')
from glob import glob
from tqdm import tqdm
from tifffile import imread
from TerminatorUtils import npzfileGenerator
from csbdeep.utils import Path, download_and_extract_zip_file
from csbdeep.data import  create_patches, RawData
import os
import glob
from TerminatorUtils.helpers import save_8bit_tiff_imagej_compatibleZZ


# In[2]:


#Generate Patches of Training Data

BaseDirectory = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/l_sancere/Training_Data_Sets/Training_StarWat/Masks_Generator/Masks_Movies_Mix2/'
SaveNpzDirectory = '/run/media/sancere/DATA/Lucas_NextonCreated_npz/'
SaveName = 'Masks_Generator_Mix2_onlyequalized.npz'


# In[3]:


# patch_size = (320,320), n_patches_per_image = 16     # for bin1
# so try patch_size = (160,160), n_patches_per_image = 32 for bin2 

npzfileGenerator.generate_2D_patch_training_data(BaseDirectory, SaveNpzDirectory, SaveName, patch_size = (160,160), n_patches_per_image = 32)  

