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
import cv2
import yaml
import random
import skimage
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import openslide as oSlide

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

from matplotlib.lines import Line2D
from scipy.stats import fisher_exact

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

from Utils.Image_Utils import MaskFromXML
from Utils.Controls import MakeHalfMask, MakeCircleMask
from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Separation import GetAccuracies
from Utils.Visualization import apply_plot_settings, rgb2hex, add_significance

# %% Visualization settings, helper functions, useful lists.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

clusterCmap =  ["#FFFFFF", "#9CCB7A",  "#D4B578",  
                "#6A567A",  "#7C6A67",  "#9E3D3F",  
                "#3A7D8C",  "#C3738C",  "#E9B84A",  
                "#4C7C48",  "#2D1C39"]

# %% Make ground truth masks.

def get_masks(dataset, saveFig=False, numberOfRuns=10, uni=False):
    """Given dataset names, search for MorphoITH profiles, tumor classifier output, and generate negative controls.

    Args:
        dataset (string): 'bap1', 'setd2', or 'pbrm1'.

    Returns:
        profilesDir (dictionary): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDir (dictionary): arrays of tumor mask with excluded markers (value) for each sample (key).
        rgbDir (dictionary): arrays of rgb values corresponding to profilesDir (value) for each sample (key).
        clustersDir (dictionary): nested directory with {sample: cluster mask} (value) for clusters number (key).
        halfMasksDir (dictionary): lists of negative control masks (value) for each sample (key).
        circMasksDir (dictionary): lists of contiguity control masks (value) for each sample (key).
        halfOverlapDir (dictionary): nested directory with {sample: overlap value} (value) for region A/B of negative control (key).
        circOverlapDir (dictionary): nested directory with {sample: overlap value} (value) for region A/B of contiguity control (key).
    """    

    profilesDir = {}
    masksDir = {}
    rgbDir = {}
    clustersDir = {5:{},10:{},15:{},20:{}}
    if uni:
        clustersDir = {10:{}}

    halfMasksDir = {}
    circMasksDir = {}

    halfOverlapDir = {'A':{},'B':{}}
    circOverlapDir = {'A':{},'B':{}}
    
    samplesList = pd.read_csv(files['DATASETS']['mayo'])
    samplesList = samplesList[samplesList['Mutation']==dataset]
    samplesList = samplesList[samplesList['Annotations']==True]['Name'].tolist()

    for name in samplesList:
        name = str(name)
        
        # slide info:
        svsFile = os.path.join(files['SLIDES']['mayo'],name+'.svs')
        slide = oSlide.open_slide(svsFile)
        level = min(2,slide.level_count-1)
        imgLowRes = np.array(slide.read_region((0,0), level, slide.level_dimensions[level]))[:,:,range(3)]

        if dataset == 'bap1':
            annoFile = os.path.join(files['MASKS']['annotations'], dataset,name+'.xml')
            mask,_=MaskFromXML(annoFile,'Focal_BAP1',slideDim=(slide.dimensions),downSampleFactor=16)
        else:
            annoFile = os.path.join(files['MASKS']['annotations'], dataset,name+'.txt')
            mask,_=MaskFromXML(annoFile,[], slideDim=slide.dimensions,downSampleFactor=16)

        mask[mask != 0] = 1
    
        # tumor mask:
        tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],dataset,name+'.npy')
        tumor = np.load(tissueMaskFile)
        tumor = np.uint8(SmoothResponseAndClasses(tumor,smoothSize=0)[0])
        tumor[tumor != 5] = 0
        tumor[tumor == 5] = 1
        tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
        tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))

        decX = int(mask.shape[0]/tumor.shape[1])
        decY = int(mask.shape[0]/tumor.shape[1])
        assert decX == decY
        
        # mutations mask:
        mask = cv2.resize(mask, (tumor.shape[1], tumor.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask += 1
        mask = mask*tumor
        masksDir[name] = mask

        ### NEGATIVE CONTROLS ###
        halfMasks, halfOverlapA, halfOverlapB = MakeHalfMask(mask, numberOfRuns)
        circleMasks, circleOverlapA, circleOverlapB = MakeCircleMask(mask, numberOfRuns)
        halfMasksDir[name] = halfMasks
        circMasksDir[name] = circleMasks

        halfOverlapDir['A'][name] = halfOverlapA
        halfOverlapDir['B'][name] = halfOverlapB
        circOverlapDir['A'][name] = circleOverlapA
        circOverlapDir['B'][name] = circleOverlapB

        profilesFile = os.path.join(files['MASKS']['morphoith'],dataset,name+'.npy')
        if uni:
            profilesFile = os.path.join(files['MASKS']['uni'],'mayo',name+'.npy')

        profiles = np.load(profilesFile)
        idx1, idx2 = np.nonzero(np.uint8(mask)!=0)
        profilesFlat = profiles[idx1, idx2, :]
        profilesDir[name] = profiles

        for n_cluster in clustersDir:

            clusterFile = os.path.join(files['MASKS']['clusters_mayo'],name+'_cluster'+str(n_cluster)+'.npy')
            if uni:
                clusterFile = os.path.join(files['MASKS']['clusters_mayo_uni'],name+'_cluster'+str(n_cluster)+'.npy')

            outputLabels = np.load(clusterFile)
            outputLabels = outputLabels * tumor

            clustersDir[n_cluster][name] = outputLabels

        # # #*#*# VISUALIZATION #*#*#
        slideToVisualize = slidesToVisualize[(slidesToVisualize['Reason']=='Mutations') & (slidesToVisualize['Dataset']==dataset)]['Name'].tolist()
        if (name in slideToVisualize) & (uni != True):

            cmap = ListedColormap(['white', 'lightgrey', 'black'])
            wtPatch = mpatches.Patch(color='lightgrey', label='Wild type')
            lossPatch = mpatches.Patch(color='black', label='Loss')
            
            # MUTATIONS #
            fig = plt.figure(figsize=(5,5))
            plt.imshow(mask,cmap=cmap,interpolation='none')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.legend(handles=[wtPatch, lossPatch], bbox_to_anchor=(1.05, 1.0), loc='upper left',framealpha=1.0, facecolor='white')
            plt.title('Ground truth')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'mutations'+ name+'_'+dataset+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'mutations'+ name+'_'+dataset+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            # H&E #
            fig = plt.figure(figsize=(5,5))
            plt.imshow(imgLowRes,interpolation='none')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            plt.title('H&E image')
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            # RGB #
            idxR,idxC=np.nonzero(mask)
            profilesPCA = PCA(n_components=10).fit_transform(profilesFlat)
            m=np.percentile(profilesPCA,[1],axis=0)
            M=np.percentile(profilesPCA,[99],axis=0)
            profilesPcaScaled=np.clip((profilesPCA-m)/(M-m),0,1)
            pcaRGB=np.ones((mask.shape[0],mask.shape[1],3))
            pcaRGB[idxR,idxC,:]=profilesPcaScaled[:,range(3)]
            rgbDir[name] = pcaRGB[idxR,idxC,:]

            fig = plt.figure(figsize=(5,5))
            plt.imshow(pcaRGB)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            if name in slidesToVisualize[(slidesToVisualize['Reason']=='Mutations tsne')]['Name'].tolist():
                plt.title('Similarity measure\n')
            else:
                plt.title('MorphoITH')
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()
            
        # # #*#*# VISUALIZATION #*#*#
        slideToVisualize = slidesToVisualize[(slidesToVisualize['Reason']=='Heterogeneity') & (slidesToVisualize['Dataset']==dataset)]['Name'].tolist()
        if (name in slideToVisualize) & (uni != True):

            secondAnnoFile = os.path.join(files['MASKS']['annotations'], 'setd2',name+'.txt')
            secondMask,_=MaskFromXML(secondAnnoFile,[], slideDim=slide.dimensions,downSampleFactor=16)
            secondMask[secondMask != 0] = 2
            secondMask = cv2.resize(secondMask, (tumor.shape[1], tumor.shape[0]), interpolation=cv2.INTER_LINEAR)
            secondMask += 1
            secondMask = secondMask*tumor
            combinedMask = mask * secondMask
            
            bap1Patch = mpatches.Patch(color='steelblue', label='BAP1')
            setd2Patch = mpatches.Patch(color='forestgreen', label='SETD2')

            cmap = ListedColormap(['white', 'lightgrey', 'steelblue','forestgreen'])
            fig = plt.figure(figsize=(5,5))
            plt.imshow(combinedMask,cmap=cmap,interpolation='none')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            legend = plt.legend(handles=[bap1Patch, setd2Patch], bbox_to_anchor=(.45, 1.0), loc='upper left',
                                framealpha=1.0,prop={'size': 12}, facecolor='white',title='Loss region',title_fontsize='12')
            for text in legend.get_texts():
                text.set_fontstyle('italic')
            plt.title('Ground truth\n')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'mutations_combined_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'mutations_combined_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()
        
            # cluster visualization
            fig = plt.figure(figsize=(5,5))
            outputLabels = clustersDir[10][name] 
            plt.imshow((outputLabels == 6) | (outputLabels == 7) | (outputLabels == 9) | (outputLabels == 3),cmap=ListedColormap(clusterCmap),interpolation='none')
            plt.imshow(outputLabels,cmap=ListedColormap(clusterCmap),interpolation='none',alpha=.5)

            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'clusters_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'clusters_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()  

    return profilesDir, masksDir, rgbDir, clustersDir, halfMasksDir, circMasksDir, halfOverlapDir, circOverlapDir 



def cluster_overlap(vectorsDir, masksDir, clustersDir, masksHalfDir, circMasksDir, saveFig=False, numberOfRuns=10, uni=False):
    """Calculate overlap between morphological clusters and loss areas in focal cases.
    
    Args:
        vectorsDir (dictionary): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDir (dictionary): arrays of tumor mask with excluded markers (value) for each sample (key).
        clustersDir (dictionary): nested directory with {sample: cluster mask} (value) for clusters number (key).
        masksHalfDir (dictionary): lists of negative control masks (value) for each sample (key).
        circMasksDir (dictionary): lists of contiguity control masks (value) for each sample (key).

    Returns:
        splitDir (dictionary): nested directory with {sample: list of overlaps} (value) for clusters number (key) based on morphology.
        splitDirCircle (dictionary): nested directory with {sample: list of overlaps} (value) for clusters number (key) for contiguity control.
        splitDirHalfMask (dictionary): nested directory with {sample: list of overlaps} (value) for clusters number (key) for negative control.
    """

    splitDir = {5:{},10:{},15:{},20:{}}
    splitDirCircle = {5:{},10:{},15:{},20:{}}
    splitDirHalfMask = {5:{},10:{},15:{},20:{}}
    if uni:
        splitDir = {10:{}}
        splitDirCircle = {10:{}}
        splitDirHalfMask =  {10:{}}

    for n_cluster in clustersDir:
        for name in clustersDir[n_cluster]:

            clusterMask = clustersDir[n_cluster][name]
            focalMask = masksDir[name]
            profiles = vectorsDir[name]

            tumor = clusterMask.copy()
            tumor[tumor!=0]=1
            focalMask = focalMask * tumor

            maskError = focalMask.copy()
            maskError[maskError != 2]= 0
            maskErrorDilate = cv2.dilate(np.uint8(maskError),np.ones((5,5),np.uint8))
            maskErrorErode = cv2.erode(np.uint8(maskError),np.ones((5,5),np.uint8))
            maskErrorDilate = (maskErrorDilate - maskError) * tumor  
            maskErrorErode = (maskError - maskErrorErode) * tumor 
            maskErrorDilate[maskErrorDilate==2] = 2
            maskErrorErode[maskErrorErode==2] = 1
            maskError = focalMask + maskErrorDilate + maskErrorErode

            halfMasks = masksHalfDir[name]
            circleMasks = circMasksDir[name]

            clusters = np.unique(clusterMask)
            values = np.unique(maskError)

            overlapTable = pd.DataFrame(0, index=clusters[1:], columns=values[:-1])
            for cluster in clusters[1:]:
                for value in values[:-1]:
                    overlapCount = np.sum((maskError == value) & (clusterMask == cluster))
                    overlapTable.loc[cluster, value] = overlapCount
            fractionOverlapTable = overlapTable.div(overlapTable.sum(axis=1), axis=0)
            splitList = list(fractionOverlapTable[2].values)
            splitDir[n_cluster][name] = splitList

            splitDirCircle[n_cluster][name] = []
            splitDirHalfMask[n_cluster][name] = []
            for n in range(numberOfRuns):
                maskCircle = circleMasks[n]
                maskHalf = halfMasks[n]

                overlapTableCircleMask = pd.DataFrame(0, index=clusters[1:], columns=values[:-1])
                overlapTableHalfMask = pd.DataFrame(0, index=clusters[1:], columns=values[:-1])

                for cluster in clusters[1:]:
                    for value in values[:-1]:
                        overlapCountCircleMask = np.sum((maskCircle == value) & (clusterMask == cluster))
                        overlapCountHalfMask = np.sum((maskHalf == value) & (clusterMask == cluster))
                        overlapTableCircleMask.loc[cluster, value] = overlapCountCircleMask
                        overlapTableHalfMask.loc[cluster, value] = overlapCountHalfMask

                fractionOverlapTableCircle = overlapTableCircleMask.div(overlapTableCircleMask.sum(axis=1), axis=0)
                splitListCircle = list(fractionOverlapTableCircle[2].values)
                
                fractionOverlapTableHalfMask = overlapTableHalfMask.div(overlapTableHalfMask.sum(axis=1), axis=0)
                splitListHalfMask= list(fractionOverlapTableHalfMask[2].values)


                splitDirCircle[n_cluster][name].extend(splitListCircle)
                splitDirHalfMask[n_cluster][name].extend(splitListHalfMask)

            # overlap visualization
            slideToVisualize = slidesToVisualize[(slidesToVisualize['Reason']=='Overlap')]['Name'].tolist()
            if (name in slideToVisualize) & (uni != True):

                dataset = slidesToVisualize[(slidesToVisualize['Reason']=='Overlap')]['Dataset']
                dataset = dataset.item()

                overlapMask = clusterMask.copy()
                overlapMask = overlapMask.astype(np.float64)
                for i in np.unique(clusterMask)[1:]:
                    overlapMask[overlapMask==i] = fractionOverlapTable.loc[i][2.0]
                maskedZeros = clusterMask==0

                if n_cluster==10:

                    # overlap
                    fig = plt.figure(figsize=(5,5))
                    plt.imshow(np.ma.array(overlapMask, mask=maskedZeros),cmap='coolwarm',interpolation='none',vmin=0,vmax=1)
                    plt.xticks([],[])
                    plt.yticks([],[])
                    cbar = plt.colorbar(orientation='horizontal',shrink=0.9,pad=0.05)
                    cbar.set_label('Overlap\n')
                    cbar.ax.tick_params(labelrotation=0)
                    plt.tight_layout()
                    fig.patch.set_alpha(0.0)
                    if saveFig:
                        fig.savefig(os.path.join(figuresDir, 'overlap_'+name+'_'+dataset+'.png'), bbox_inches='tight')
                        fig.savefig(os.path.join(figuresDir, 'overlap_'+name+'_'+dataset+'.svg'),format='svg',dpi=600,bbox_inches='tight')
                    plt.show()   
                
                    # cluster visualization
                    fig = plt.figure(figsize=(5,5))
                    plt.imshow(clusterMask,cmap=ListedColormap(clusterCmap),interpolation='none')
                    plt.xticks([],[])
                    plt.yticks([],[])
                    plt.tight_layout()
                    fig.patch.set_alpha(0.0)
                    if saveFig:
                        fig.savefig(os.path.join(figuresDir, 'clusters_'+name+'_'+dataset+'.png'), bbox_inches='tight')
                        fig.savefig(os.path.join(figuresDir, 'clusters_'+name+'_'+dataset+'.svg'),format='svg',dpi=600,bbox_inches='tight')
                    plt.show()   

                    # H&E
                    svsFile = os.path.join(files['SLIDES']['mayo'],name+'.svs')
                    slide = oSlide.open_slide(svsFile)
                    level = min(2,slide.level_count-1)
                    imgLowRes = np.array(slide.read_region((0,0), level, slide.level_dimensions[level]))[:,:,range(3)]

                    fig = plt.figure(figsize=(5,5))
                    plt.imshow(imgLowRes,interpolation='none')
                    plt.xticks([],[])
                    plt.yticks([],[])
                    plt.tight_layout()
                    fig.patch.set_alpha(0.0)
                    if saveFig:
                        fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.png'), bbox_inches='tight')
                        fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
                    plt.show()  

                    # morphoith
                    idxR,idxC=np.nonzero(focalMask)
                    profilesFlat = profiles[idxR,idxC,:]
                    profilesPCA = PCA(n_components=10).fit_transform(profilesFlat)
                    m=np.percentile(profilesPCA,[1],axis=0)
                    M=np.percentile(profilesPCA,[99],axis=0)
                    profilesPcaScaled=np.clip((profilesPCA-m)/(M-m),0,1)
                    pcaRGB=np.ones((focalMask.shape[0],focalMask.shape[1],3))
                    pcaRGB[idxR,idxC,:]=profilesPcaScaled[:,range(3)]

                    fig = plt.figure(figsize=(5,5))
                    plt.imshow(pcaRGB)
                    plt.xticks([],[])
                    plt.yticks([],[])
                    plt.tight_layout()
                    fig.patch.set_alpha(0.0)
                    if saveFig:
                        fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.png'), bbox_inches='tight')
                        fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
                    plt.show()  
                    
                    cmap = ListedColormap(['white', 'lightgrey', 'black'])
                    wtPatch = mpatches.Patch(color='lightgrey', label='Wild type')
                    lossPatch = mpatches.Patch(color='black', label='Loss')

                    # mutations
                    fig = plt.figure(figsize=(5,5))
                    plt.imshow(focalMask,cmap=cmap,interpolation='none')
                    plt.legend(handles=[wtPatch, lossPatch], bbox_to_anchor=(.58, 1.0), loc='upper left',framealpha=1.0, facecolor='white',fontsize=14)
                    plt.xticks([],[])
                    plt.yticks([],[])
                    plt.tight_layout()
                    fig.patch.set_alpha(0.0)
                    if saveFig:
                        fig.savefig(os.path.join(figuresDir, 'mutations_focal_'+name+'.png'), bbox_inches='tight')
                        fig.savefig(os.path.join(figuresDir, 'mutations_focal_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
                    plt.show()  


    return splitDir, splitDirCircle, splitDirHalfMask


def run_masks(dataset, profilesDict, masksDict, rgbDict, masksHalfDir, masksCircDir, saveFig=False, numberOfRuns=10, uni=False):
    """
    Args:
        dataset (str): 'bap1', 'setd2', or 'pbrm1'
        profilesDict (dictionary): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDict (dictionary): arrays of tumor mask with excluded markers (value) for each sample (key).
        rgbDict (dictionary): arrays of rgb values corresponding to profilesDir (value) for each sample (key).
        masksHalfDir (dictionary): lists of negative control masks (value) for each sample (key).
        masksCircDir (dictionary): lists of contiguity control masks (value) for each sample (key).


    Returns:
        accuracies (dictionary):
        splitAccuracies (dictionary):
        circAccuracies (dictionary):
        lossSizes (dictionary):
        wtSizes (dictionary):
    """

    accuracies = {}
    circAccuracies = {}
    splitAccuracies = {}
    seedsDatasets = {'bap1':1,'setd2':2,'pbrm1':3}
    lossSizes = {}
    wtSizes = {}

    subsetOfProfiles = []

    for cnt, name in enumerate(list(profilesDict.keys())):

        profiles = profilesDict[name]
        mask = masksDict[name]
        maskFlat = mask[mask!=0]
        idx1, idx2 = np.nonzero(np.uint8(mask)!=0)
        profilesFlat = profiles[idx1, idx2, :]
        
        if np.unique(mask).shape[0] < 2:
            continue

        if name not in accuracies.keys():
            accuracies[name] = []
            circAccuracies[name] = []
            splitAccuracies[name] = []
            lossSizes[name] = []
            wtSizes[name] = []
        
        lossIdx = np.where(maskFlat==2)[0]
        wtIdx = np.where(maskFlat==1)[0]
        
        np.random.seed(seedsDatasets[dataset]*(cnt+1))
        assert lossIdx.shape[0] > 300
        lossIdxSubset = np.random.choice(lossIdx,300)
        
        np.random.seed(seedsDatasets[dataset]*(cnt+1))
        assert wtIdx.shape[0] > 300
        wtIdxSubset = np.random.choice(wtIdx,300)
        
        idxSubset= np.concatenate([lossIdxSubset,wtIdxSubset])
        labelSubset = maskFlat[idxSubset]
        profilesSubset = profilesFlat[idxSubset]

        subsetOfProfiles.append(profilesFlat[np.random.choice(idxSubset,20)])

        profilesPcaSubset = PCA(n_components=10).fit_transform(profilesSubset)
        acc = GetAccuracies(profilesPcaSubset, labelSubset)
        accuracies[name].append(acc)
        
        uniq, uniqCnt = np.unique(mask,return_counts=True)
        lossSize = uniqCnt[np.where(uniq==2)[0][0]]
        wtSize = uniqCnt[np.where(uniq==1)[0][0]]
        
        lossSizes[name].append(lossSize)
        wtSizes[name].append(wtSize)

        randomMethods = {'circle':[masksCircDir, circAccuracies], 'split': [masksHalfDir, splitAccuracies]}
        
        for randomMethod in randomMethods:
            randomAcc= []
            for i in range(numberOfRuns):
                randomMask = randomMethods[randomMethod][0][name][i]
                randomMaskFlat = randomMask[randomMask!=0]
                randomLossIdx =  np.where(randomMaskFlat==2)[0]
                randomWtIdx = np.where(randomMaskFlat==1)[0]
                
                np.random.seed(seedsDatasets[dataset]*(cnt+1)*(i+1))
                randomLossIdxSubset = np.random.choice(randomLossIdx,300)
                np.random.seed(seedsDatasets[dataset]*(cnt+1)*(i+1))
                randomWtIdxSubset = np.random.choice(randomWtIdx,300)
                
                randomIdxSubset = np.concatenate([randomLossIdxSubset,randomWtIdxSubset])
                randomLabelSubset = randomMaskFlat[randomIdxSubset]
                randomProfilesSubset = profilesFlat[randomIdxSubset]

                randomProfilesPcaSubset = PCA(n_components=10).fit_transform(randomProfilesSubset)
                r_acc = GetAccuracies(randomProfilesPcaSubset, randomLabelSubset)
                randomAcc.append(r_acc)
            
            randomMethods[randomMethod][1][name].extend(randomAcc)
        
        slideToVisualize = slidesToVisualize[(slidesToVisualize['Reason']=='Mutations tsne')]['Name'].tolist()
        if (name in slideToVisualize) & (uni != True):

            colorDict = {'Wild type':'lightgrey','Loss':'black'}
            labelsNamed = np.asarray(['Loss' if l==2 else 'Wild type' for l in labelSubset])
            tsnePos = TSNE(init='random',n_components=2, metric='cosine', perplexity=15, random_state=123, learning_rate=200, square_distances=True).fit_transform(profilesFlat[idxSubset])
            tsneDf = pd.DataFrame({'t-SNE 1':tsnePos[:,0], 't-SNE 2':tsnePos[:,1], 'Status':labelsNamed})
            fig = plt.figure(figsize=(5,5))
            sns.scatterplot(data=tsneDf,x='t-SNE 1',y='t-SNE 2',hue='Status',palette=colorDict,s=100)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',title='Status',framealpha=1.0, facecolor='white')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.title('Feature vectors colored by ground truth\n')
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, name+'_mutations_tsne.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, name+'_mutations_tsne.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            rgbColors = rgbDict[name]
            tsneDf['RGB'] = [rgb2hex(i*255) for i in rgbColors[idxSubset]]
            fig = plt.figure(figsize=(5,5))
            sns.scatterplot(data=tsneDf,x='t-SNE 1',y='t-SNE 2',color=tsneDf.RGB,s=100,legend=None)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.title('Feature vectors colored by morphological similarity\n')
            fig.patch.set_alpha(0.0)
            if saveFig:
                tsneDf[['Status','t-SNE 1','t-SNE 2','RGB']].to_csv(os.path.join(figuresDir, 'morphoith_tsne_'+name+'.png'))
                fig.savefig(os.path.join(figuresDir, 'morphoith_tsne_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_tsne_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()
    subsetOfProfiles = np.concatenate(subsetOfProfiles)
    return accuracies, splitAccuracies, circAccuracies, lossSizes, wtSizes, subsetOfProfiles

# %%

def __main__(saveFig=False, numberOfRuns=10, saveCalc=False, uni=False, runCal=False):

    mutationsDf = pd.DataFrame(columns=['Accuracy','Sample'])
    circMutationsDf = pd.DataFrame(columns=['Accuracy','Sample', 'Overlap A', 'Overlap B'])
    splitMutationsDf = pd.DataFrame(columns=['Accuracy','Sample', 'Overlap A', 'Overlap B'])
    splitAllDf = pd.DataFrame(columns=['Cluster', 'Name', 'Overlap', 'Dataset'])
    splitAllRandomDf = pd.DataFrame(columns=['Cluster', 'Name', 'Overlap', 'Dataset'])
    splitAllHalfMaskDf = pd.DataFrame(columns=['Cluster', 'Name', 'Overlap', 'Dataset'])
    lossDfFull = pd.DataFrame(columns=['Name', 'Size loss'])
    wtDfFull = pd.DataFrame(columns=['Name', 'Size wild type'])

    subsetOfAllProfiles = []

    if uni:
        suffix = '_uni'
    else:
        suffix = ''

    for dataset in ['setd2', 'pbrm1', 'bap1']:

        vectorsDir, masksDir, rgbDir, clustersDir, masksHalfDir, circMasksDir, halfOverlapDir, circOverlapDir = get_masks(dataset, saveFig, numberOfRuns, uni=uni)
        splitDir, splitDirRandom, splitDirHalfMask = cluster_overlap(vectorsDir, masksDir, clustersDir, masksHalfDir, circMasksDir, saveFig, numberOfRuns, uni=uni)

        splitFlat  = [(clus, name, value) for clus, idx_dict in splitDir.items() for name, value in idx_dict.items()]
        splitPd = pd.DataFrame(splitFlat, columns=['Cluster', 'Name', 'Overlap'])
        splitPd['Dataset'] = dataset
        splitAllDf = pd.concat([splitAllDf,splitPd],ignore_index=True)

        splitRandomFlat  = [(clus, name, value) for clus, idx_dict in splitDirRandom.items() for name, value in idx_dict.items()]
        splitRandomPd = pd.DataFrame(splitRandomFlat, columns=['Cluster', 'Name', 'Overlap'])
        splitRandomPd['Dataset'] = dataset
        splitAllRandomDf = pd.concat([splitAllRandomDf,splitRandomPd],ignore_index=True)
        
        splitHalfMaskFlat  = [(clus, name, value) for clus, idx_dict in splitDirHalfMask.items() for name, value in idx_dict.items()]
        splitHalfMaskPd = pd.DataFrame(splitHalfMaskFlat, columns=['Cluster', 'Name', 'Overlap'])
        splitHalfMaskPd['Dataset'] = dataset
        splitAllHalfMaskDf = pd.concat([splitAllHalfMaskDf,splitHalfMaskPd],ignore_index=True)

        accNorm, accSplit, accCircle, lossSizes, wtSizes, subsetOfProfiles  = run_masks(dataset, vectorsDir, masksDir, rgbDir, masksHalfDir, circMasksDir, uni=uni)
        subsetOfAllProfiles.append(subsetOfProfiles)

        normDf = pd.Series(accNorm, name='Accuracy').rename_axis('Sample').explode().reset_index()
        normDf['Dataset'] = dataset
        
        lossDf = pd.Series(lossSizes, name='Size loss').rename_axis('Sample').explode().reset_index()
        lossDf['Dataset'] = dataset
        
        wtDf = pd.Series(wtSizes, name='Size wild type').rename_axis('Sample').explode().reset_index()
        wtDf['Dataset'] = dataset

        circDf = pd.Series(accCircle, name='Accuracy').rename_axis('Sample').explode().reset_index()
        circDf['Dataset'] = dataset
        circOverADf = pd.Series(circOverlapDir['A'], name='Overlap').rename_axis('Sample').explode().reset_index()
        circOverBDf = pd.Series(circOverlapDir['B'], name='Overlap').rename_axis('Sample').explode().reset_index()
        circDf['Overlap A'] = circOverADf['Overlap']
        circDf['Overlap B'] = circOverBDf['Overlap']

        splitdDf = pd.Series(accSplit, name='Accuracy').rename_axis('Sample').explode().reset_index()
        splitdDf['Dataset'] = dataset
        splitOverADf = pd.Series(halfOverlapDir['A'], name='Overlap').rename_axis('Sample').explode().reset_index()
        splitOverBDf = pd.Series(halfOverlapDir['B'], name='Overlap').rename_axis('Sample').explode().reset_index()

        splitdDf['Overlap A'] = splitOverADf['Overlap']
        splitdDf['Overlap B'] = splitOverBDf['Overlap']

        mutationsDf = pd.concat([mutationsDf,normDf])
        circMutationsDf = pd.concat([circMutationsDf,circDf])
        splitMutationsDf = pd.concat([splitMutationsDf,splitdDf])
        lossDfFull = pd.concat([lossDfFull,lossDf])
        wtDfFull = pd.concat([wtDfFull,wtDf])
        
    subsetOfAllProfiles = np.concatenate(subsetOfAllProfiles)
    if runCal:
        if uni:
            np.save(os.path.join(os.path.join(figuresDir,'profilesMayo_UNI.npy')), subsetOfAllProfiles)
        else:
            np.save(os.path.join(os.path.join(figuresDir,'profilesMayo.npy')), subsetOfAllProfiles)
        
        
    mutationsDf = mutationsDf.reset_index(drop=True)
    circMutationsDf = circMutationsDf.reset_index(drop=True)
    splitMutationsDf = splitMutationsDf.reset_index(drop=True)
    lossDfFull = lossDfFull.reset_index(drop=True)
    wtDfFull = wtDfFull.reset_index(drop=True)

    if saveCalc:
        mutationsDf.to_csv(os.path.join(figuresDir, 'normMS.csv'),index=False)
        circMutationsDf.to_csv(os.path.join(figuresDir, 'circMS.csv'),index=False)
        splitMutationsDf.to_csv(os.path.join(figuresDir, 'halfMS.csv'),index=False)

    if not uni:
        # what's the percentage of loss area?
        assert lossDfFull['Sample'].values.all() == wtDfFull['Sample'].values.all()
        lossDfFull['Fraction loss area'] = lossDfFull['Size loss'] / (wtDfFull['Size wild type'] + lossDfFull['Size loss'])

        print('Min loss area',lossDfFull['Fraction loss area'].min())
        print('Max loss area',lossDfFull['Fraction loss area'].max())

        print('5th percentile of loss areas', np.percentile(lossDfFull['Fraction loss area'].values,5))
        print('95th percentile of loss areas', np.percentile(lossDfFull['Fraction loss area'].values,95))

        mutColDir = {'bap1':'steelblue','setd2':'forestgreen','pbrm1':'orange'}
        fig, axes = plt.subplots(1, 3, figsize=(4,4), sharey=True)

        for i, dataset in enumerate(['bap1','setd2','pbrm1']):
            ax = axes[i]
            subsetLossDf = lossDfFull[lossDfFull['Dataset'] == dataset]

            sns.rugplot(ax=ax, data=subsetLossDf, y='Fraction loss area', height=1, linewidth=5, color=mutColDir[dataset])
            ax.set_title(dataset.upper(), fontstyle='italic')
            ax.set_xticks([])
            ax.set_ylim(0.05,0.95)
            ax.set_ylabel('Fraction of loss area')
            fig.patch.set_alpha(0.0)
            if i != 0:
                ax.set_ylabel('')

        plt.tight_layout()
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'muations_loss_area.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'muations_loss_area.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()

    # plot separation for driver mutations:
    fig = plt.figure(figsize=(5,5))
    ax = sns.swarmplot(data=mutationsDf,  x='Dataset', y='Accuracy', hue = 'Dataset', s=10,legend=None, order=['bap1', 'setd2', 'pbrm1'], hue_order = ['bap1','setd2','pbrm1'], palette=['steelblue','forestgreen','orange'])

    patches1 = Line2D([], [], linestyle="--", ms=8, color='dimgrey', label='95th pct. of contiguity baseline')
    patches2 = Line2D([], [], linestyle="--", ms=8, color='silver', label='95th pct. of negative control')

    plt.title('Separation of driver gene mutation\n')
    plt.ylim((0.49,1.05))
    plt.xlim((-1,2.8))
    plt.ylabel('Separation')

    plt.legend(handles=[patches1, patches2], loc='lower left', framealpha=1.0, facecolor='white', fontsize=12.5)
    curretXticks = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())  
    ax.set_xticklabels([label.upper() for label in curretXticks], fontstyle='italic')
    for label in ax.get_xticklabels():
        label.set_fontstyle('italic')
    ax.set_xlabel('')

    cPrct = np.percentile(list(circMutationsDf['Accuracy']), 95)
    sPrct = np.percentile(list(splitMutationsDf['Accuracy']),95)

    plt.axhline(y=cPrct, color='dimgrey', linestyle='--', label='y=2')
    plt.axhline(y=sPrct, color='silver', linestyle='--', label='y=2')

    x_offset = plt.xlim()[1] - 3.77
    plt.text(x=x_offset, y=cPrct + 0.01, s='Full', color='black', verticalalignment='bottom', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=(cPrct + sPrct) / 2, s='Partial', color='dimgrey', verticalalignment='center', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=sPrct - 0.01, s='None', color='silver', verticalalignment='top', horizontalalignment='left',fontsize=15)
    plt.ylabel('Separation')
    fig.patch.set_alpha(0.0)
    if saveFig:
        mutationsDf[['Dataset','Accuracy']].to_csv(os.path.join(figuresDir, 'separation_mutations_bap1.csv'))
        fig.savefig(os.path.join(figuresDir, 'separation_mutations'+suffix+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'separation_mutations'+suffix+'.svg'),format='svg',dpi=600, bbox_inches='tight')
    plt.show()

    if not uni:
        # separation for BAP1 only:
        data=mutationsDf[mutationsDf['Dataset']=='bap1']
        data.index=['Driver gene']*data.shape[0]

        fig = plt.figure(figsize=(5,5))
        ax = sns.swarmplot(data=data,  x=data.index, y='Accuracy', hue=data.index, palette=['cadetblue'],s=15)
        plt.title('Separation of driver gene mutation\n')
        plt.ylim((0.49,1.05))
        plt.xlim((-0.5,0.5))
        plt.ylabel('Separation')
        curretXticks = [label.get_text() for label in ax.get_xticklabels()]
        ax.set_xticks(ax.get_xticks())  
        ax.set_xticklabels([label.upper() for label in curretXticks])

        plt.axhline(y=cPrct, color='dimgrey', linestyle='--', label='y=2')
        plt.axhline(y=sPrct, color='silver', linestyle='--', label='y=2')

        x_offset = plt.xlim()[1] - .99
        plt.text(x=x_offset, y=cPrct + 0.01, s='Full', color='black', verticalalignment='bottom', horizontalalignment='left',fontsize=15)
        plt.text(x=x_offset, y=(cPrct + sPrct) / 2, s='Partial', color='dimgrey', verticalalignment='center', horizontalalignment='left',fontsize=15)
        plt.text(x=x_offset, y=sPrct - 0.01, s='None', color='silver', verticalalignment='top', horizontalalignment='left',fontsize=15)

        patches1 = Line2D([], [], linestyle="--", ms=8, color='dimgrey', label='95th pct. of contiguity baseline')
        patches2 = Line2D([], [], linestyle="--", ms=8, color='silver', label='95th pct. of negative control')

        plt.legend(handles=[patches1,patches2],loc='lower left',framealpha=1.0, facecolor='white',fontsize=12.5)
        fig.patch.set_alpha(0.0)
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'separation_bap1.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'separation_bap1.svg'),format='svg',dpi=600, bbox_inches='tight')
        plt.show()

        # overlap - stripplot:
        for n_cluster in [5,10,15,20]:

            print('Number of clusters N='+str(n_cluster))

            splitAllDf['Source'] ='Ground truth'
            splitAllRandomDf['Source'] = 'Negative control A'
            splitAllHalfMaskDf['Source'] = 'Negative control B'

            mergedDf = pd.concat([splitAllDf, splitAllRandomDf, splitAllHalfMaskDf], ignore_index=True)
            mergedDf = mergedDf[mergedDf['Cluster']==n_cluster].reset_index(drop=True)
            random.seed(321)
            mergedDf['Overlap'] = mergedDf['Overlap'].apply(lambda x: random.sample(x, min(10, len(x))))
            mergedDf = mergedDf.explode('Overlap')
            mergedDf = mergedDf.reset_index(drop=True)

            # how many counts between 0.2 and 0.8 overlap?
            middleOverlap = mergedDf[(mergedDf['Overlap'] >= 0.2) & (mergedDf['Overlap'] <= 0.8)]
            middleOverlapCnt = middleOverlap.groupby(['Dataset', 'Source'])['Overlap'].count().reset_index()
            middleOverlapCnt.rename(columns={'Overlap': 'Middle overlap'}, inplace=True)

            otherOverlap = mergedDf[(mergedDf['Overlap'] < 0.2) + (mergedDf['Overlap'] > 0.8)]
            otherOverlapCnt = otherOverlap.groupby(['Dataset', 'Source'])['Overlap'].count().reset_index()
            otherOverlapCnt.rename(columns={'Overlap': 'Other overlap'}, inplace=True)

            allOverlap = pd.merge(middleOverlapCnt, otherOverlapCnt, on=['Dataset', 'Source'], how='left')
            allOverlap['Middle overlap percentage'] = allOverlap['Middle overlap']/(allOverlap['Middle overlap']+allOverlap['Other overlap']) * 100

            # overlap - only those that are in the overlap (.2-.8) region:
            g = sns.catplot(data=allOverlap, x='Dataset', y='Middle overlap percentage', hue='Source', kind='bar',palette=['darkviolet','grey','darkgrey'],legend=False,
                            hue_order=['Ground truth','Negative control A','Negative control B'],order=['bap1','setd2','pbrm1'],aspect=1.5)
            ax = g.axes[0][0]
            max_height = 73
            if n_cluster==5:
                max_height=98
            xticks=['BAP1','SETD2','PBRM1']

            allOverlap_Ordered = allOverlap.copy()
            allOverlap_Ordered['Dataset'] = pd.Categorical(allOverlap_Ordered['Dataset'], categories=['bap1','setd2','pbrm1'], ordered=True)
            allOverlap_Ordered['Source']  = pd.Categorical(allOverlap_Ordered['Source'],  categories=['Ground truth','Negative control A','Negative control B'], ordered=True)
            allOverlap_Ordered = allOverlap_Ordered.sort_values(['Source','Dataset']).reset_index(drop=True)

            for bar, raw in zip(ax.patches,allOverlap_Ordered['Middle overlap']):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,  
                    height + 0.5,                 
                    f"N={raw:.0f}",               
                    ha='center', va='bottom',
                    fontdict={'size': 15}
                )

            for i, dataset in enumerate(xticks):
                dataset = dataset.lower()
                subset = allOverlap[allOverlap['Dataset'] == dataset]
                
                ours_vs_negA = [[subset.loc[subset['Source'] == 'Ground truth', 'Middle overlap'].values[0], subset.loc[subset['Source'] == 'Ground truth', 'Other overlap'].values[0]],
                                [subset.loc[subset['Source'] == 'Negative control A', 'Middle overlap'].values[0], subset.loc[subset['Source'] == 'Negative control A', 'Other overlap'].values[0]]]
                
                ours_vs_negB = [[subset.loc[subset['Source'] == 'Ground truth', 'Middle overlap'].values[0], subset.loc[subset['Source'] == 'Ground truth', 'Other overlap'].values[0]],
                                [subset.loc[subset['Source'] == 'Negative control B', 'Middle overlap'].values[0], subset.loc[subset['Source'] == 'Negative control B', 'Other overlap'].values[0]]]
                
                _, p_value_A = fisher_exact(ours_vs_negA)
                _, p_value_B = fisher_exact(ours_vs_negB)
                
                x_ours = i - 0.25  
                x_negA = i  
                x_negB = i + 0.25 
                
                add_significance(ax, x_ours, x_negA, max_height -35, p_value_A)
                add_significance(ax, x_ours, x_negB, max_height -10, p_value_B)

            ax.set_ylim(0, 80)

            for cnt,ax in enumerate(g.axes.flat):
                ax.set_xlabel('') 
                ax.set_xticklabels(xticks, fontdict={'fontstyle': 'italic'})
                ax.set_ylabel('Clusters split between\nloss and wild type (%)')
                ax.set_yticks([0, 25, 50, 75]) 
                if n_cluster==5:
                    ax.set_yticks([0, 25, 50, 75,100]) 
                

            plt.legend(handles=[mpatches.Patch(color='darkviolet', label='Ground truth'),
                                mpatches.Patch(color='grey', label='Contiguity baseline'),
                                mpatches.Patch(color='darkgrey', label='Negative control')], bbox_to_anchor=(.5, -.15), loc='upper center',framealpha=1.0, facecolor='white')

            g.fig.patch.set_alpha(0.0)
            g.fig.patch.set_visible(False)
            if saveFig:
                allOverlap[['Dataset','Middle overlap percentage','Source']].to_csv(os.path.join(figuresDir, 'overlap_bars.png'))
                g.savefig(os.path.join(figuresDir, 'overlap_bars_clusters'+str(n_cluster)+'.png'), bbox_inches='tight')
                g.savefig(os.path.join(figuresDir, 'overlap_bars_clusters'+str(n_cluster)+'.svg'),format='svg',dpi=600, bbox_inches='tight')
            plt.show()

# %%\
if __name__ == "__main__":
    
    __main__()
