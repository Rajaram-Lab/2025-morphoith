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
import glob
import pickle
import torch
import yaml
import numpy as np
import pandas as pd
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("cuda")

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

import Utils.RetCCL_ResNet as ResNet
import Utils.Patch_Generation as pg

from Utils.Pytorch_Dataset import Dataset_Simple
from Utils.Load_PatientInfo import load_patients_info

# %%  Save other kinds of datasets profiles.

def save_profiles_tma1_folds(model, fold, saveDir, permute_data, modelType = None):
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    foldsPath=files['DATA']['tma']['bigtma_folds']

    testingFoldsFile=os.path.join(foldsPath,'Testing_Fold'+str(fold)+'.pkl')
    (testPatches,testLabels,_,_)=pickle.load(open(testingFoldsFile,'rb'))

    predScoresDict = {key: [] for key in np.unique(testLabels)}

    for label in np.unique(testLabels):
        indicesLabel = np.where(testLabels == label)
        patches = testPatches[indicesLabel]

        generator = Dataset_Simple(patches, permute_data)
        batch = DataLoader(generator, batch_size=16)
        output = []
        model.eval()
        with torch.no_grad():
            for data in batch:
                rawOutput = model(data.to(device))
                output.append(rawOutput.cpu().detach().numpy())
        predScoresDict[label] = np.concatenate(output, axis=0)
    
    np.save(os.path.join(saveDir, 'test_fold'+str(fold)), predScoresDict)
    
def save_profiles_tma2(model, saveDir, permute_data):

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    patchesPath = files['DATA']['tma']['tma2']
    patchesFiles = glob.glob(os.path.join(patchesPath, '*.hdf5'))
    patchesData,_,_,patchesNumber=pg.LoadPatchData(patchesFiles,returnSampleNumbers=True)
    for i in np.unique(patchesNumber):
        patchesIdx = np.where(patchesNumber == i)
        patches = patchesData[0][patchesIdx]

        generator = Dataset_Simple(patches, permute_data)
        batch = DataLoader(generator, batch_size=16)
        output = []
        model.eval()
        with torch.no_grad():
            for data in batch:
                rawOutput = model(data.to(device))
                output.append(rawOutput.cpu().detach().numpy())
        output = np.concatenate(output, axis=0)
        
        name = patchesFiles[int(i)].split('/')[-1].split('.')[0]
        np.save(os.path.join(saveDir, name), output)


def save_profiles_pilot(model, saveDir, permute_data):

    allSamplesInfo = load_patients_info()
    allSamplesInfo = allSamplesInfo.set_index('Patient.ID')
    direction = 'Flip'

    patchesPath = files['DATA']['pilot']['cores_20x_224px']
    patchesFiles = glob.glob(os.path.join(patchesPath, '*.hdf5'))
    allHdf5Names = [Path(item).stem for item in patchesFiles]

    for slideId in allHdf5Names:

        predScoresDict = {}
        hdf5File = os.path.join(patchesPath, slideId + '.hdf5')

        if os.path.exists(hdf5File):
            if not os.path.exists(os.path.join(saveDir, slideId + '.npy')):
                patchData, patchClasses, classDict = pg.LoadPatchData([hdf5File], returnSampleNumbers=False)
                
                for i in list(classDict.keys()):
                    if '_X2' in i:
                        classDict[i.split('_')[0]]=classDict[i]
                        del classDict[i]

                slideInfo = allSamplesInfo[allSamplesInfo['Image.Flip']==slideId+'.svs']

                for i in range(slideInfo.shape[0]):
                    sampleInfo = slideInfo.iloc[i]
                    sampleId = sampleInfo['Sample.ID']
                    annoName = sampleInfo['Punch.ID']

                    if annoName in list(classDict.keys()):
                        isInAnno = patchClasses == classDict[annoName]
                        patchesInfo = patchData[0][isInAnno]

                        if sampleId not in predScoresDict:
                            predScoresDict[sampleId] = {}
                            if direction not in predScoresDict[sampleId]:
                                predScoresDict[sampleId][direction] = {}

                        generator = Dataset_Simple(patchesInfo, permute_data)
                        batch = DataLoader(generator, batch_size=16)
                        output = []
                        model.eval()
                        with torch.no_grad():
                            for data in batch:
                                rawOutput = model(data.to(device))
                                output.append(rawOutput.cpu().detach().numpy())

                        predScoresDict[sampleInfo['Sample.ID']][direction] = np.concatenate(output, axis=0)
                np.save(os.path.join(saveDir, slideId), predScoresDict)

            else:
                print('Profiles done for')

        else:
            print('no patches')

# %% Extract profiles using MorphoITH.

model = timm.create_model('vit_base_patch16_224').to(device)
permute_data = transforms.Compose([transforms.ToTensor()])
model.load_state_dict(torch.load(files['WEIGHTS']['morphoith']))

# for example:
saveDir = os.path.join(files['FEATURES']['morphoith'],'TMA2', 'AllFolds')
save_profiles_tma2(model, saveDir, permute_data)
