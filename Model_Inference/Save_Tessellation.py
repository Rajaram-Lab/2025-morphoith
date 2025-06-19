"""
    Copyright (C) 2025, Rajaram Lab - UTSouthwestern 
    
    This file is part of 2025-morphoith.
    
    2025-morphoith is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    2025-morphoith is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with 2025-morphoith. If not, see <http://www.gnu.org/licenses/>.
    
    Aleksandra W. Nielsen, 2025
"""

# %%

import timm
import os
import sys
import glob
import yaml
import torch
import openslide as oSlide
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda:0")

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

from Utils.Tessellation import TessellationDataset, Reconstruct_Slide_Dataset, CropTessellation


# %% MorphoITH model.

model_morphoith = timm.create_model('vit_base_patch16_224').to(device)
permute_data = transforms.Compose([transforms.ToTensor()])
model_morphoith.load_state_dict(torch.load(files['WEIGHTS']['morphoith']))
model_morphoith.eval()

# %% Parallelized tesselation.

def run_tesselation(model, svsList, saveDir, patchSize=224, stride=100, numberOfChannels=1000):
    """Run WSI tesselation.

    Args:
        model (model): loaded PyTorch model.
        svsList (list): list of WSIs to tessellate.
        saveDir (str): profiles save location.
        patchSize (int): patch size. Defaults to 224.
        stride (int): stride. Defaults to 100.
    """    

    for slideFile in svsList:

        slideName = slideFile.split('/')[-1].split('.')[0]
        savePath = os.path.join(saveDir, slideName+'.npy')

        if not os.path.isfile(savePath):
        
            slide = oSlide.open_slide(slideFile)
            slideMpp = np.mean([float(slide.properties[p]) for p in slide.properties if 'mpp' in p.lower()])
            dsf = int(np.round(0.5 / slideMpp))
            
            pDataSet=TessellationDataset(slide,patchSize = patchSize, stride = stride, nBx = 24,
                                        downSampleFactor = dsf)
            pDataLoader=DataLoader(pDataSet,batch_size=pDataSet.batch_size,num_workers=16)
            numberOfChannels=numberOfChannels
            nPatches=pDataSet.numberOfBoxes*pDataSet.batch_size

            res=np.zeros((nPatches,numberOfChannels))
            for batchCounter,batchData in enumerate(tqdm(pDataLoader)):
                imgs=batchData.float().to(device)
                with torch.no_grad():
                    output=model(imgs)    
                res[(batchCounter*pDataSet.batch_size):((batchCounter+1)*pDataSet.batch_size),:]=output.cpu().detach().numpy()

            outMat_combined = Reconstruct_Slide_Dataset(res,pDataSet)
            outMat_cropped = CropTessellation(outMat_combined, slide, dsf, stride)
            np.save(savePath, outMat_cropped)

        else:
            print("Skipping "+slideName)

# %% Example of running tessellation using MorphoITH on cd31 dataset.

dataset = 'tcga'
saveDir = os.path.join(files['MASKS']['morphoith'], dataset)
svsList = glob.glob(os.path.join(...,'*.svs'))

run_tesselation(model_morphoith, svsList, saveDir)


# %%
