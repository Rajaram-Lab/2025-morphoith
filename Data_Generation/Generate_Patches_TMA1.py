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
import h5py
import ntpath
import yaml
import progressbar
import openslide as oSlide
import glob as glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from PIL import Image

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

import Utils.Patch_Generation as pg

# %%
class PatchReader():

    def __init__(self,slide,downSampleFactor,downSampleTolerance=0.025):

        self.slide=slide
        assert downSampleFactor>=1 
        self.downSampleFactor=downSampleFactor
        
        if np.min(np.abs(np.array(slide.level_downsamples)-downSampleFactor))<downSampleTolerance:
          self.magLevel=int(np.argmin(np.abs(np.array(slide.level_downsamples)-downSampleFactor)))
          self.isResizeNeeded=False
          self.downSampleExtracted=self.downSampleFactor

        else:
          self.magLevel=int(np.where(np.array(slide.level_downsamples)<downSampleFactor)[0][-1])
          self.isResizeNeeded=True
          self.downSampleExtracted=slide.level_downsamples[self.magLevel]
  
    
    def LoadPatches(self,patchCenters,patchSize,showProgress=False):
      assert ((type(patchCenters) is np.ndarray) and patchCenters.shape[1]==2),sys.error('Invalid Patch Centers')
      numberOfPatches  =patchCenters.shape[0]
      
      if self.isResizeNeeded:
        patchSizeLoaded=np.int32(np.round(np.array(patchSize)*self.downSampleFactor/self.downSampleExtracted))
      else:
        patchSizeLoaded=patchSize
      
      patchData=np.zeros((numberOfPatches,patchSize[0],patchSize[1],3),np.uint8)
      cornerPositions=np.int32(np.floor(np.array(patchCenters)-self.downSampleExtracted*(patchSizeLoaded+1)/2.0))
      if showProgress:
        bar=progressbar.ProgressBar(max_value=numberOfPatches)
      for patchCounter in range(numberOfPatches):
        img=self.slide.read_region(cornerPositions[patchCounter],self.magLevel,patchSizeLoaded)
        if self.isResizeNeeded:
          img=img.resize(patchSize,Image.BILINEAR)
        patchData[patchCounter]=np.array(img,np.uint8)[:,:,np.arange(3)]
        if showProgress:
          bar.update(patchCounter)
      if showProgress:  
        bar.finish()      
      
      return patchData

# %%

TmaPath = files['SLIDES']['bigtma']
imgList = glob.glob(os.path.join(TmaPath,"*.svs"))

TmaPathInfo = files['MASKS']['bigtma']
TmaInfoList = [os.path.join(TmaPathInfo, i.split('/')[-1].split('.')[0]+'.qptma') for i in imgList]

patchReaders={}
patchCenters=[]
patchClasses=[]
magLevel='20X'
maskDSF=16 
patchSize=224
patchSizeList=[patchSize]

slideDsf={'5X':8,'20X':2,'40X':1}
downSampleLevels=[slideDsf[magLevel]]

savePath = files['DATA']['tma']['bigtma_raw']

for i in range(len(imgList)):
    print(imgList[i])
    if(True): 
        if(True):
            print('x')
            if(os.path.exists(TmaInfoList[i])):
                print(imgList[i])

                classData = []
                patchData = []
                patchDataX = []
                centersData = []
                slide=oSlide.open_slide(imgList[i])
                mag = slide.properties.get('aperio.AppMag')
                print('AppMag :', mag)
                magLevel = 1
                x, y = slide.dimensions
                bigImg=np.asarray(slide.read_region((0,0),magLevel,slide.level_dimensions[magLevel]))
                plt.imshow(bigImg)
                plt.show()
                bigImgTemp = np.logical_or(np.logical_or(bigImg[:,:,0]<230, bigImg[:,:,1]<230), bigImg[:,:,2]<230)# logical or
                bigImgTemp = bigImgTemp.astype(np.uint8)
                plt.imshow(bigImgTemp)
                plt.show()
                print(imgList[i])
                fileName = ntpath.basename(imgList[i].replace('.svs',""))
    
                if(mag=='20'):
                    dS = 1
                    patchReaders[dS]=pg.PatchReader(slide,dS)
                else:
                    dS = 2
                    patchReaders[dS]=pg.PatchReader(slide,dS)
            
                data = pd.read_csv(TmaInfoList[i], delimiter="\t", header=None,skiprows=[0,1,2,3,4,5])
                print(data)
                hf = h5py.File(savePath + fileName + '.h5', 'w')
                punchCounter=0
                for row in range(data.shape[0]):
                    if(data.iloc[row,5] == True):
                        numberOfCan = 200
                        x = int(data.iloc[row,1])
                        y = int(data.iloc[row,2])
                        w = int(data.iloc[row,3])
                        h = int(data.iloc[row,4])
                        img=np.asarray(slide.read_region((x,y),0,(w,h)))
                        plt.imshow(img)
                        plt.show()
                        kernel = np.ones((5,5))
                        imgTemp = np.logical_or(np.logical_or(img[:,:,0]<230, img[:,:,1]<230), img[:,:,2]<230)
                        imgTemp = imgTemp.astype(np.uint8)
                        plt.title('original mask')
                        boarderMask = np.ones((imgTemp.shape[0], imgTemp.shape[1]))
                        boarderMask[:,0:int(224/4)] = 0
                        boarderMask[:,boarderMask.shape[1]-int(224/4):] = 0
                        boarderMask[0:int(224/4),:] = 0
                        boarderMask[boarderMask.shape[0]-int(224/4):,:] = 0
                        mask = np.multiply(np.logical_and(imgTemp,boarderMask),1)

                        yRange = range(w)
                        xRange = range(h)

                        maskToImgDownScale=((slide.dimensions[0]/imgTemp.shape[1])+(slide.dimensions[1]/imgTemp.shape[0]))/2
                        absPatchSizes=np.array(downSampleLevels)*np.array(patchSizeList)
                        maxPatchSizeScaled=224*2
                        maxPatchesPerAnno=200 
                        maxAvgPatchOverlap=3.0
                        minFracPatchInAnno=0.4
                        candidateMask=ndi.uniform_filter(np.float32(imgTemp),maxPatchSizeScaled)>minFracPatchInAnno
                        plt.title('minFracPatchInAnno = '+ str(minFracPatchInAnno))

                        # exclude borders of image:
                        candidateMask[range(int(maxPatchSizeScaled/2)),:]=False
                        candidateMask[range(0,-int(maxPatchSizeScaled/2),-1),:]=False
                        candidateMask[:,range(int(maxPatchSizeScaled/2))]=False
                        candidateMask[:,range(0,-int(maxPatchSizeScaled/2),-1)]=False
                        plt.title('candidate mask')
                        plt.imshow(candidateMask)
                        plt.show()
                    
                        # these are potential positions for the patch center:
                        candidatePos=np.where(candidateMask)
                        
                        # determine number of patches to extract:
                        maxPatchesForOverlap=np.int32(np.sum(candidateMask)*(maxAvgPatchOverlap+1)/(maxPatchSizeScaled*maxPatchSizeScaled))
                        numberOfPatches=min(maxPatchesPerAnno,maxPatchesForOverlap)
                        print(numberOfPatches)
                        chosenIdx=np.random.choice(candidatePos[0].size,numberOfPatches)
                        cands = np.zeros([len(chosenIdx),2])

                        for j in range(len(chosenIdx)):
                            cands[j,0] = x+candidatePos[1][chosenIdx[j]]
                            cands[j,1] = y+candidatePos[0][chosenIdx[j]]

    
                        patchData.append(patchReaders[dS].LoadPatches(cands,np.array([224,224])))
                        print(fileName+"-"+data.iloc[row,0])
                        for k in range(len(chosenIdx)):
                            classData.append(fileName+"-"+data.iloc[row,0])
                        centersData.append(cands)

                        fig, axs = plt.subplots(2,10, figsize=(15, 6), facecolor='w', edgecolor='k')
                        fig.subplots_adjust(hspace = .5, wspace=.001)
                        
                        axs = axs.ravel()
                        counter = 0
                        if numberOfPatches>20:
                            for rIx in np.random.choice(numberOfPatches,20):
                                axs[counter].imshow(patchData[punchCounter][rIx])
                                counter=counter+1
                            plt.show()
                        punchCounter=punchCounter+1

                    else:
                        print('bad punch!')
                hf.create_dataset('patches', data=np.concatenate(patchData))
                hf.create_dataset('classes', data=np.asarray(classData,dtype='S'))
                hf.create_dataset('centers', data=np.concatenate(centersData))
                hf.close()
            else:
                print('qptma is missing')
        else:
            print('File is already created')


# %%
