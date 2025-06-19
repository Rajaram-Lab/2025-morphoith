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
import glob
import h5py
import yaml
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2

from sklearn.model_selection import KFold

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

#%%
savePath = files['DATA']['tma']['bigtma_raw']
os.chdir(savePath)
hdf5List = [savePath+file for file in glob.glob("*.h5")]
patchesList = []
classesList = []
centersList = []

for h5File in hdf5List:
    hf = h5py.File(h5File, 'r')
    hf.keys()
    patches = hf.get('patches')
    patches = np.array(patches)
    patchesList.append(patches)
    classes = hf.get('classes')
    classes = np.array(classes)
    classesList.append(classes)
    centers = hf.get('centers')
    centers = np.array(centers)
    centersList.append(centers)

patchData = np.concatenate(patchesList)
classLabel_byte = np.concatenate(classesList)
centersData = np.concatenate(centersList)

classLabel = []
for i in range(classLabel_byte.shape[0]):
    classLabel.append(classLabel_byte[i].decode('UTF-8'))

classLabel = np.asarray(classLabel)

slideLabel = []
punchLabel = []
for i in range(len(classLabel)):
    slideLabel.append(classLabel[i].split('-')[0])
    punchLabel.append(classLabel[i].split('-')[1]+'-'+classLabel[i].split('-')[2])
slideLabel = np.asarray(slideLabel)
punchLabel = np.asarray(punchLabel)

#%% visualize the distribution of patches
ixList = []
for punch in np.unique(classLabel):
    ix = np.where(classLabel==punch)[0]
    ixList.append(len(ix))
plt.hist(ixList)

#%% filter out punches with less than x number of patches
minNumberofPathces = 100
ixList = []
for punch in np.unique(classLabel):
    ix = np.where(classLabel==punch)[0]
    if(len(ix)>minNumberofPathces):
        ixList.append(ix)

ixList = np.concatenate(ixList)
classLabel = classLabel[ixList]
slideLabel = slideLabel[ixList]
punchLabel = punchLabel[ixList]
patchData = patchData[ixList]

#%% visualize the distribution of patches
ixList = []
for punch in np.unique(classLabel):
    ix = np.where(classLabel==punch)[0]
    ixList.append(len(ix))
plt.hist(ixList)

#%%

uniqueSlides = np.unique(slideLabel)
print(uniqueSlides)
kf = KFold(n_splits=3)#, random_state=20, shuffle=True)
foldCount = 0
for train_index, test_index in kf.split(uniqueSlides):
    print("TRAIN:", train_index, "TEST:", test_index[:-1], "Validation:", test_index[-1])
    trainingSlides = uniqueSlides[train_index]
    testingSlides = uniqueSlides[test_index[:-1]]
    validationSlides = uniqueSlides[test_index[-1]]
    trainingIxList = []
    testingIxList = []
    validationIxList = []
    for slide in uniqueSlides:
        if(slide in trainingSlides):
            trainingIx =  np.where(slide==slideLabel)[0]
            trainingIxList.append(trainingIx)
        elif(slide in validationSlides):
            validationIx = np.where(slide==slideLabel)[0]
            validationIxList.append(validationIx)
        else:
            testingIx = np.where(slide==slideLabel)[0]
            testingIxList.append(testingIx)
    
    trainingIxList = np.concatenate(trainingIxList)
    validationIxList = np.concatenate(validationIxList)
    testingIxList = np.concatenate(testingIxList)
    
    trainingPatchData = patchData[trainingIxList]
    trainingClassLabel = classLabel[trainingIxList]
    trainingPunchLabel = punchLabel[trainingIxList]
    trainingSlideLabel = slideLabel[trainingIxList]
    
    validationPatchData = patchData[validationIxList]
    validationClassLabel = classLabel[validationIxList]
    validationPunchLabel = punchLabel[validationIxList]
    validationSlideLabel = slideLabel[validationIxList]
    
    testingPatchData = patchData[testingIxList]
    testingClassLabel = classLabel[testingIxList]
    testingPunchLabel = punchLabel[testingIxList]
    testingSlideLabel = slideLabel[testingIxList]
    
    print(trainingPatchData.shape)
    print(validationPatchData.shape)
    print(testingPatchData.shape)

    saveDir = files['DATA']['tma']['bigtma_folds']
    saveFile=os.path.join(saveDir,'Training'+'_Fold'+str(foldCount)+'.pkl')
    # pickle.dump([trainingPatchData, trainingClassLabel, trainingPunchLabel, trainingSlideLabel],
    #             open(saveFile,'wb'),protocol=4)
    
    saveFile=os.path.join(saveDir,'Validation'+'_Fold'+str(foldCount)+'.pkl')
    # pickle.dump([validationPatchData, validationClassLabel, validationPunchLabel, validationSlideLabel],
    #             open(saveFile,'wb'),protocol=4)
    
    saveFile=os.path.join(saveDir,'Testing'+'_Fold'+str(foldCount)+'.pkl')
    # pickle.dump([testingPatchData, testingClassLabel, testingPunchLabel, testingSlideLabel],
    #             open(saveFile,'wb'),protocol=4)

    foldCount = foldCount + 1
