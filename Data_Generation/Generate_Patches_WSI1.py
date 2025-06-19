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

import os
import sys
import yaml
import pandas as pd
import numpy as np
import openslide as oSlide

from tqdm import tqdm

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

figuresDir = files['FIGURES']
import Utils.Patch_Generation as pg
from Utils.Normalization import StainNormalizer

os.environ["KMP_WARNINGS"] = "0"

# %% 

def svs_annotation_files_search(svsId):
    """Fetches WSIs and their annotations paths.

    Args:
        svsId (str): name of the slide.

    Returns:
        svsFull (str): full path to the slide.
        annoFull (str): full path to the slide's annotations.
    """

    baseSvsDir=files['SLIDES']['pilot']
    baseAnnoDir=files['MASKS']['annotations_samples']
    batchSvsDirs=['Batch1/HnE/','Batch2/HnE','Batch3']
    batchAnnoDirs=['PilotBatch1_2.2_v3','QP22-Pilot_Batch2','QP22_Pilot_Batch3']

    svsFull = None
    annoFull = None

    for cnt in range(len(batchSvsDirs)):
        svsPath=os.path.join(baseSvsDir, batchSvsDirs[cnt],svsId+'.svs')
        if os.path.exists(svsPath):
            svsFull = svsPath
            break

    for cnt2 in range(len(batchAnnoDirs)):
        annoPath=os.path.join(baseAnnoDir, batchAnnoDirs[cnt2], 'annotations', svsId+'.txt')
        if os.path.exists(annoPath):
            annoFull = annoPath
            break

    return svsFull, annoFull

# %% Load patchesd info and normalization arguments.

slidesToNormalizeTo = pd.read_csv(files['DATASETS']['pilot'])
slidesToNormalizeTo = slidesToNormalizeTo[slidesToNormalizeTo['Sequencing']==True]['Name'].tolist()
patientNames = np.unique([i.split('-')[0] for i in slidesToNormalizeTo])

patchSaveDir=files['DATA']['pilot']['cores_20x_224px']
normScheme='Macenko'
normPatchesSaveDir=os.path.join(files['DATA']['pilot']['cores_20x_224px'],'Normalized'+normScheme)

# %% Create the patches.

distinguishAnnosInClass=False 
patchSizeList=[224] 
showProgress=False
maxPatchesPerAnno=250 
maxAvgPatchOverlap=8.0 
minFracPatchInAnno=0.95 
maskDsf=8

for slideId in tqdm(slidesToNormalizeTo):
    
    slideFile,annoFile=svs_annotation_files_search(slideId)
    hdf5File=os.path.join(patchSaveDir, slideId+'.hdf5')
    if not os.path.exists(hdf5File):
        if os.path.exists(annoFile):
            slide=oSlide.open_slide(slideFile)
            slideMpp = np.mean([float(slide.properties[p]) for p in slide.properties if 'mpp' in p.lower()])
            
            if 0.45 <= slideMpp <= 0.55: # 20X image
                downSampleLevels = [1]
            elif 0.20 <= slideMpp <= 0.30: # 40X image
                
                downSampleLevels = [2]
            else:
                sys.exit('Unrecognized mpp')
            
            mask,maskToClassDict=pg.MaskFromXML(annoFile,None,slide.dimensions,
                                                distinguishAnnosInClass=distinguishAnnosInClass,
                                                outerBoxLabel='Box',downSampleFactor=maskDsf)
            patchData,patchClasses,patchCenters=pg.PatchesFromMask(slide,mask,
                                                                   downSampleLevels,patchSizeList,
                                                                   maskToClassDict,
                                                                   maxPatchesPerAnno=maxPatchesPerAnno,
                                                                   showProgress=showProgress,
                                                                   maxAvgPatchOverlap=maxAvgPatchOverlap,
                                                                   minFracPatchInAnno=minFracPatchInAnno)
                
            pg.SaveHdf5Data(hdf5File,patchData,patchClasses,patchCenters,
                            downSampleLevels,patchSizeList,slideFile)   
        else:
            print('No annotation file:',annoFile,'. Skipping')  
    else:
        print(hdf5File,' already exists. Skipping')        

# %% Create normalized patches (per patient normalization).

allPatientsPatches = {}
allPatientsTargets = {}

# get the stats:
for patient in patientNames:
    if patient not in allPatientsPatches.keys():
        allPatientsPatches[patient] = []

    slideIds = [i for i in slidesToNormalizeTo if i.startswith(patient)]
    for slideId in slideIds:
         hdf5File = os.path.join(patchSaveDir, slideId + '.hdf5')
         patchData, patchClasses,classDict,sampleNumbers,patchCenters = pg.LoadPatchData([hdf5File],returnSampleNumbers=True,returnPatchCenters=True)
         allPatientsPatches[patient].append(patchData[0])

    allPatientsPatches[patient] = np.concatenate(allPatientsPatches[patient])
    patches = allPatientsPatches[patient]
    rdIdx = np.random.choice(patches.shape[0],250,replace=False)
    rdPatches = patches[rdIdx]
    patchSize = rdPatches.shape[1]
    numberOfPatches = rdPatches.shape[0]
    pixelData=np.zeros((numberOfPatches*patchSize*patchSize,3))  
    for n in range(numberOfPatches):
                pixelData[range(n*patchSize*patchSize,(n+1)*patchSize*patchSize),:]=np.resize(rdPatches[n],(patchSize*patchSize,3))
    pixelDataFin = np.uint8(pixelData.reshape(pixelData.shape[0],1,3))
    allPatientsTargets[patient] = pixelDataFin

    for n in range(numberOfPatches):
                pixelData[range(n*patchSize*patchSize,(n+1)*patchSize*patchSize),:]=np.resize(patches[n],(patchSize*patchSize,3))
    pixelDataFin = np.uint8(pixelData.reshape(pixelData.shape[0],1,3))
    allPatientsTargets[patient] = pixelDataFin

# normalize:
for patient in patientNames:
    print(patient)
    slideIds = [i for i in slidesToNormalizeTo if i.startswith(patient)]

    myNormalizer=StainNormalizer(normScheme)
    myNormalizer.fit(allPatientsTargets[patient])

    for slideId in tqdm(slideIds):

        normFile = os.path.join(normPatchesSaveDir,slideId+'.hdf5')
        if not os.path.exists(normFile):

            hdf5File = os.path.join(patchSaveDir, slideId + '.hdf5')
            patchData, patchClasses,classDict,sampleNumbers,patchCenters = pg.LoadPatchData([hdf5File],returnSampleNumbers=True,returnPatchCenters=True)
            patchDataNorm = []

            for patch in patchData[0]:
                # can't handle all white patches:
                if patch.mean() > 230:
                    patchDataNorm.append(patch)
                else:
                    normPatch = myNormalizer.transform(patch)
                    patchDataNorm.append(normPatch)
            
            patchDataNorm = np.asarray(patchDataNorm)
            classMapping = {v: k for k, v in classDict.items()}
            funMapping = np.vectorize(classMapping.get)
            mappedClasses = funMapping(patchClasses)
            mappedClasses = list(mappedClasses)
            pg.SaveHdf5Data(normFile,[patchDataNorm], mappedClasses, patchCenters, [1], [224], slideId)

        
# %%
