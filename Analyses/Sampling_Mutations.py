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
import skimage
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openslide as oSlide

from matplotlib.colors import ListedColormap
from scipy import stats
from matplotlib.lines import Line2D

os.chdir(os.path.dirname(os.path.dirname(__file__)))

from Utils.Image_Utils import MaskFromXML
from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Clusters import calculate_feature_representation
from Utils.Visualization import apply_plot_settings

with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %% Visualization settings, helper functions, useful dictionaries.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

clusterCmap = ["#FFFFFF", "#3A7D8C", "#B4526F",  
               "#E9B84A", "#9CCB7A", "#4C7C48", 
               "#2D1C39", "#D4B578", "#7C6A67"]

# %% Make ground truth masks.

def get_masks(dataset, uni=False):
    """Given a dataset name, return corresponding MorphoITH profiles and tumor classifier output.

    Returns:
        profilesDir (directory): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDir (directory): arrays of tumor mask with excluded markers (value) for each sample (key).

    """

    profilesDir = {}
    masksDir = {}

    samplesList = pd.read_csv(files['DATASETS']['mayo'])
    samplesList = samplesList[samplesList['Mutation']==dataset]
    samplesList = samplesList[samplesList['Annotations']==True]['Name'].tolist()

    for name in samplesList:
        name = str(name)

        svsFile = os.path.join(files['SLIDES']['mayo'],name+'.svs')
        slide = oSlide.open_slide(svsFile)

        if dataset == 'bap1':
            annoFile = os.path.join(files['MASKS']['annotations'], dataset,str(name)+'.xml')
            mask,_=MaskFromXML(annoFile,'Focal_BAP1',slideDim=(slide.level_dimensions[0][0],slide.level_dimensions[0][1]),downSampleFactor=16)
        else:
            annoFile = os.path.join(files['MASKS']['annotations'], dataset,str(name)+'.txt')
            mask,_=MaskFromXML(annoFile,[], slideDim=slide.dimensions,downSampleFactor=16)

        mask[mask != 0] = 1
        mask = np.uint8(skimage.morphology.remove_small_objects(mask == 1, 100))

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

        mask = cv2.resize(mask, (tumor.shape[1], tumor.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask += 1
        mask = mask*tumor

        # 1 == WT, 2 == loss, 3 == uncertain loss:
        maskError = mask.copy()
        maskError[maskError != 2]= 0
        maskErrorDilate = cv2.dilate(np.uint8(maskError),np.ones((5,5),np.uint8))
        maskErrorErode = cv2.erode(np.uint8(maskError),np.ones((5,5),np.uint8))
        maskErrorDilate = (maskErrorDilate - maskError) * tumor  
        maskErrorErode = (maskError - maskErrorErode) * tumor 

        maskErrorDilate[maskErrorDilate==2] = 2
        maskErrorErode[maskErrorErode==2] = 1

        maskError = mask + maskErrorDilate + maskErrorErode
        masksDir[name] = maskError

        profilesFile = os.path.join(files['MASKS']['morphoith'],dataset,name+'.npy')
        if uni:
            profilesFile = os.path.join(files['MASKS']['uni'],'mayo',name+'.npy')
        profiles = np.load(profilesFile)
        profilesDir[name] = profiles

    return profilesDir, masksDir


# %%

def sampling(profilesDict, masksDict, saveFig=False, uni=False, showFig=True):
    """ Sample most dissimilar morphologically points and check their driver mutation status.

    Args:
        profilesDir (directory): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDir (directory): arrays of tumor mask with excluded markers (value) for each sample (key).

    Returns:
        samplingResults (directory): sampling results (values) for number of sampling sites (key) based on morphology.
        randomSamplingResults (directory): random sampling results (values) for number of sampling sites (key).
    """    

    samplingResults = {2:[],3:[],4:[]}
    randomSamplingResults = {2:[],3:[],4:[]}

    for cnt, name in enumerate(list(profilesDict.keys())):

        profiles = profilesDict[name]
        originalMask = masksDict[name]
        sampledMask = originalMask.copy()
        tumor = originalMask.copy()
        tumor[tumor!=0]=1

        sampleNumber = 5

        for sample in range(2, sampleNumber):

            clusterSaveDir = files['MASKS']['clusters_mayo']
            if uni:
                clusterSaveDir = files['MASKS']['clusters_mayo_uni']

            clusterFile = os.path.join(clusterSaveDir,name+'_cluster'+str(sample)+'.npy')
            clusterMask = np.load(clusterFile)
            clusterMask = clusterMask * tumor
            originalMask = originalMask*(clusterMask!=0)

            idxR, idxC = np.where((originalMask!=0)&(originalMask!=3))
            labels = originalMask[idxR, idxC]
            profilesFlat = profiles[idxR,idxC]

            clus = clusterMask[idxR,idxC]
            _, repIndices, _ = calculate_feature_representation(profilesFlat,clus)
            closestIdx = np.array(list(repIndices.values()))
            status = labels[closestIdx]

            if (set([1,2]).issubset(set(status))):
                samplingResults[sample].append(1)
            else:
                samplingResults[sample].append(0)

            for i in range(25):
                np.random.seed(sample*(cnt+1)*i)
                randomStatus = list(np.random.choice(labels, sample))
                if (set([1,2]).issubset(set(randomStatus))):
                    randomSamplingResults[sample].append(1)
                else:
                    randomSamplingResults[sample].append(0)
            

        #*#*# VISUALIZATION #*#*#
        slideToVisualize = slidesToVisualize[slidesToVisualize['Reason']=='Sampling']['Name'].tolist()
        if (name in slideToVisualize) & (showFig == True):

            fig = plt.figure(figsize=(14,7))
            for sample in range(2, sampleNumber):

                clusterFile = os.path.join(clusterSaveDir,name+'_cluster'+str(sample)+'.npy')
                clusterMask = np.load(clusterFile)
                clusterMask = clusterMask * tumor
                originalMask = originalMask*(clusterMask!=0)

                idxR, idxC = np.where((originalMask!=0)&(originalMask!=3))
                profilesFlat = profiles[idxR,idxC]
                clus = clusterMask[idxR,idxC]
                _, repIndices, _ = calculate_feature_representation(profilesFlat,clus)
                closestIdx = np.array(list(repIndices.values()))

                sampledMask = np.zeros_like(originalMask)
                exchangeLabels = sampledMask[idxR,idxC]
                exchangeLabels[closestIdx] = 4
                sampledMask[idxR,idxC] = exchangeLabels
                sampledMaskDilated = cv2.dilate(sampledMask, np.ones((15,15), np.uint8), iterations=1)         
                dilIndx = np.where(sampledMaskDilated==4)

                overlayMask = np.zeros_like(clusterMask)
                overlayMask[dilIndx] = 1

                if sample==2:
                    changeColorMap = {0:0, 1: 2, 2: 1}
                    clusterMask = np.vectorize(changeColorMap.get)(clusterMask)

                plt.subplot(1, sampleNumber-2,sample-1)
                plt.imshow(clusterMask,cmap=ListedColormap(clusterCmap[:int(np.max(clus))+1]),interpolation='none',alpha=0.5)
                plt.imshow(np.ma.array(overlayMask, mask=overlayMask!=1),cmap=ListedColormap(['#000000']),interpolation='none')

                plt.xticks([],[])
                plt.yticks([],[])
                plt.tight_layout()
                plt.title('\n')

            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, name+'_sampling.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, name+'_sampling.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

    return samplingResults, randomSamplingResults


# %%
def __main__(saveFig=False, uni=False, showFig=True):

    dataframeAll = pd.DataFrame()
    dataframeAll_UNI = pd.DataFrame()
    randomDataFrameAll = pd.DataFrame()

    if uni:
        suffix = '_uni'
        showFig = False
    else:
        suffix = ''

    for dataset in ['setd2', 'pbrm1', 'bap1']:

        vectorsDict, masksDict = get_masks(dataset)
        samplingRes, randomSamplingRes = sampling(vectorsDict, masksDict, showFig=showFig)

        dataframe = pd.DataFrame(samplingRes)
        randomDataFrame = pd.DataFrame(randomSamplingRes)

        dataframe['Dataset'] = [dataset] * dataframe.shape[0]
        randomDataFrame['Dataset'] = [dataset] * randomDataFrame.shape[0]

        dataframeAll = pd.concat([dataframeAll,dataframe])
        randomDataFrameAll = pd.concat([randomDataFrameAll,randomDataFrame])

        if uni:
            vectorsDict, masksDict = get_masks(dataset, uni=uni)
            samplingRes, randomSamplingRes = sampling(vectorsDict, masksDict, uni=uni, showFig=showFig)

            dataframe = pd.DataFrame(samplingRes)
            randomDataFrame = pd.DataFrame(randomSamplingRes)

            dataframe['Dataset'] = [dataset] * dataframe.shape[0]
            dataframeAll_UNI = pd.concat([dataframeAll_UNI,dataframe])
        

    dataframeAll = dataframeAll.set_index('Dataset')
    randomDataFrameAll = randomDataFrameAll.set_index('Dataset')

    dataframeAll.index = dataframeAll.index.str.upper()
    randomDataFrameAll.index = randomDataFrameAll.index.str.upper()

    if uni:
        dataframeAll_UNI = dataframeAll_UNI.set_index('Dataset')
        dataframeAll_UNI.index = dataframeAll_UNI.index.str.upper()


    fig = plt.figure(figsize=(12,5))
    focalColors = {'BAP1':'steelblue', 'SETD2':'forestgreen', 'PBRM1':'orange'}

    for cnt, dataset in enumerate(['BAP1', 'SETD2','PBRM1']):
        plt.subplot(1,3,cnt+1)
        sns.lineplot(data=dataframeAll.T[dataset], palette=sns.color_palette([focalColors[dataset]],1),legend=None,linewidth=4)
        ax = sns.lineplot(data=randomDataFrameAll.T[dataset], palette=sns.color_palette(['silver'],1),legend=None,linewidth=4)
        ax.lines[1].set_linestyle("--")
        if uni:
            sns.lineplot(data=dataframeAll_UNI.T[dataset], palette=sns.color_palette([focalColors[dataset]],1),legend=None,linewidth=4,alpha=0.4)
            ax.lines[2].set_linestyle(":")

        plt.title(dataset, fontstyle='italic')
        plt.ylim(0.05,1.05)
        plt.xticks([2, 3, 4])

        if cnt == 0:
            plt.ylabel('Probability')
            if not uni:
                handle_random = Line2D([], [], linestyle="--", color='silver', label='Random sampling', linewidth=4)
                plt.legend(handles=[handle_random],loc='lower right')

        elif cnt == 1:
            plt.xlabel('Number of samples')
            plt.yticks([],[])
        else:
            plt.yticks([],[])

        if (uni == True) & (cnt == 2):
            handle_random = Line2D([], [], linestyle="--", color='silver', label='Random sampling', linewidth=4)
            handle_mith = Line2D([], [], linestyle="-", color=focalColors[dataset], alpha=0.4, label='MorphoITH encoder', linewidth=4)
            handle_uni = Line2D([], [], linestyle=":", color=focalColors[dataset], alpha=0.4, label='UNI', linewidth=4)
            plt.legend(handles=[handle_mith,handle_uni,handle_random],loc='upper left',frameon=False,fontsize=15)
                
    plt.suptitle('Probability of both loss and wild type being captured during sampling',y=0.97)
    plt.tight_layout()
    fig.patch.set_alpha(0.0)
    if saveFig:
        dataframeAll.to_csv(os.path.join(figuresDir, 'sampling_focal.csv'))
        fig.savefig(os.path.join(figuresDir, 'sampling_focal'+suffix+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'sampling_focal'+suffix+'.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.show()

    if not uni:
        for dataset in ['BAP1', 'SETD2', 'PBRM1']:
            print(dataset)

            allFlat = dataframeAll.T[dataset].values.flatten()
            randomFlat = randomDataFrameAll.T[dataset].values.flatten()
            
            table = [[allFlat.sum(), allFlat.size - allFlat.sum()],[randomFlat.sum(), randomFlat.size - randomFlat.sum()]]
            oddsratio, p_fe = stats.fisher_exact(table, alternative='greater')
            print(f"odds ratio = {oddsratio:.2f}, one‐sided p = {p_fe:.3g}")


# %%

if __name__ == "__main__":
    
    __main__()
    