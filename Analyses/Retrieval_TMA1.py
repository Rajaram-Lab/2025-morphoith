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
import yaml
import ast
import scipy
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.dirname(__file__)))

from Utils.Visualization import apply_plot_settings

with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %% Visualization settings, useful dictionaries and lists.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

# list of models to compare:
modelsList = [['COMBINED_PATCH/BYOL_ViTb_TMA0', 'CLOSE_PATCH/BYOL_ViTb_TMA0', 'SAME_PATCH/BYOL_ViTb_TMA0',
              'COMBINED_PATCH/BYOL_ResNet50_TMA0', 'CLOSE_PATCH/BYOL_ResNet50_TMA0', 'SAME_PATCH/BYOL_ResNet50_TMA0'],
              ['COMBINED_PATCH/Triplet_ViTb_TMA0', 'CLOSE_PATCH/Triplet_ViTb_TMA0', 'SAME_PATCH/Triplet_ViTb_TMA0',
              'COMBINED_PATCH/Triplet_ResNet50_TMA0', 'CLOSE_PATCH/Triplet_ResNet50_TMA0', 'SAME_PATCH/Triplet_ResNet50_TMA0'],
              ['CLOSE_PATCH/MoCoV2_ResNet50_TMA0', 'SAME_PATCH/MoCoV2_ResNet50_TMA0'],]

pretrainedModelsList = ['ResNet50_RetCCL', 'ViTl_UNI', 'ViTh_UNIv2', 'ViTb_CONCH','ViTh_Virchow', 'ViTh_Virchowv2','ViTg_GigaPath', 'ViTl_CONCHv1.5']

profilesDir = files['FEATURES']['main']
taskDir = files['TASKS']['retrieval_bigtma_fold0']

cmap = {'ViTb / COMBINED' : '#8B4513',
        'ViTb / CLOSE' : '#D2691E',
        'ViTb / SAME' : '#F4A460',
        'ViTb / ImageNet' : '#696969',
        'ViTb / CONCH' : 'violet',
        'ViTh / UNIv2' : '#006400',
        'ViTh / Virchow' : 'grey',
        'ResNet50 / COMBINED': '#0000CD',
        'ResNet50 / CLOSE' : '#8A2BE2',
        'ResNet50 / SAME' : '#DDA0DD',
        'ResNet50 / ImageNet' : '#A9A9A9',
        'ResNet50 / RetCCL': 'red',
        'ViTl / UNI' : '#32CD32',
        'ViTg / GigaPath' : '#FFEE8C',
        'ViTh / Virchow' : 'grey',
        'ViTh / Virchowv2' : 'darkgrey',
        'ViTl / CONCHv1.5' : 'skyblue',
        'ViTb / CONCH' : 'dodgerblue'}

# %% 

def retrieval_plot(file, nNeigh=100):
    """Calculate nearest neighbors and whether they are from the same TMA core as the input patch.

    Args:
        file (str): .npy file obtained after "run_retrieval" function.

    Returns:
        nNeighbors (string): number of neighbors (N).
        inSamePunch (array): information about nearest neighbors (N) for each patch of interest.
    """
    try:
        rawProfiles=np.load(file+'.npy',allow_pickle=True)
        rawProfiles = rawProfiles.item()

    except:
        rawProfiles=np.load(file+'.npz')
        rawProfiles = {key: rawProfiles[key].item() if rawProfiles[key].ndim == 0 else rawProfiles[key] for key in rawProfiles.files}

    punchNameList=[]
    profiles=[]
    for punchName in rawProfiles.keys():
        data = rawProfiles[punchName]
        profiles.append(data)
        punchNameList+=[punchName]*data.shape[0]
    profiles=np.concatenate(profiles)

    _,punchNumbers=np.unique(punchNameList,return_inverse=True)

    nbrs = NearestNeighbors(n_neighbors=nNeigh+1, algorithm='auto', metric='cosine',n_jobs=16).fit(profiles)
    _,indices = nbrs.kneighbors(profiles)
    inSamePunch=np.equal(punchNumbers[indices[:,1:]],np.expand_dims(punchNumbers[indices[:,0]],axis=1))
    
    return nNeigh, inSamePunch

def run_retrieval(modelsList, epochs=10):
    """Run retrieval for all the models, backbones, input types.

    Args:
        modelsList (list): models used in retreival task.
    """
    for fileName in tqdm(sum(modelsList, [])):                
        
        fileNameSplit = fileName.split('/')
        modelName = fileNameSplit[1]+'_'+fileNameSplit[0]+'_fold0'              
        if not os.path.exists(os.path.join(taskDir,modelName+'.npy')):
            
            filePath = os.path.join(profilesDir,fileName,'TMA','Fold0')
            dic = {}

            for i in range(epochs):
                file=os.path.join(filePath,'epoch'+str(i)+'/test_fold0')
                _, inSamePunch = retrieval_plot(file)
                dic[i] = inSamePunch
            np.save(os.path.join(taskDir,modelName), dic)
            
        else:
            print('Done', modelName)
            
    # pre-trained models:
    for fileName in tqdm(pretrainedModelsList):
        modelName = fileName+'_fold0'
        filePath = os.path.join(profilesDir,'BARE_MODELS',fileName,'TMA')
        
        if not os.path.exists(os.path.join(taskDir,modelName+'.npy')):
            _, inSamePunch = retrieval_plot(os.path.join(filePath,'test_fold0'))
            np.save(os.path.join(taskDir, modelName), {0:inSamePunch})
            

def fetch_retrieval(modelsList, pretrainedModelsList, epochs=10, nNeigh=10, saveFig=False):
    """Featch retrieval files and choose the best epochs.

    Args:
        modelsList (list): models used in retreival task.

    Returns:
        bestResultsDir (dictionary): best retrieval results for each model.
    """
    bestResultsDir = {}
    bestResultsDirSimple = {}

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    for cnt, fileNames in enumerate(modelsList):
        bestEpochsDict = {}

        for fileName in fileNames:
            fileNameSplit = fileName.split('/')
            modelName = fileNameSplit[1]+'_'+fileNameSplit[0]+'_fold0'       
            dic = np.load(os.path.join(taskDir,modelName+'.npy'), allow_pickle=True)
            dic = dic.item()
            allmeanSamePunch = []
            for i in range(epochs):
                inSamePunch = dic[i]
                meanSamePunch = np.mean(inSamePunch,axis=0)
                allmeanSamePunch.append(meanSamePunch)
            allmeanSamePunch = np.array(allmeanSamePunch)
            bestEpoch = np.argmax(allmeanSamePunch[:,-1])
            bestEpochsDict[modelName] = allmeanSamePunch[bestEpoch]
            if fileName == 'COMBINED_PATCH/BYOL_ViTb_TMA0':
                mainEpochsDict = bestEpochsDict.copy()
        for model, values in bestEpochsDict.items():
            modelstrList = model.split('_')
            modelNameShort = modelstrList[1]+' / '+modelstrList[3]
            axes[cnt].plot(np.arange(1,nNeigh+1), values, label=modelNameShort,linewidth=5, color=cmap[modelNameShort])

            atLastNeigh = values[-1]
            bestResultsDir[modelstrList[0]+' / '+modelNameShort] = atLastNeigh
            bestResultsDirSimple[modelNameShort] = atLastNeigh
        
        handles, labels = axes[0].get_legend_handles_labels()
        sorted_items = sorted(labels, key=lambda x: bestResultsDir.get('BYOL / '+x), reverse=True)
        sorted_handles = [handles[labels.index(lbl)] for lbl in sorted_items]
        plt.legend(sorted_handles,
                    sorted_items,
                    loc='center left',
                    bbox_to_anchor=(1, 0.5),
                    title='BACKBONE / INPUT',
                    framealpha=1.0,
                    facecolor='white')

        axes[cnt].set_ylim(0.5,1.0)
        axes[cnt].set_xticks([2,4,6,8,10])
        axes[cnt].set_title(modelstrList[0])
        if cnt == 1:
         axes[cnt].set_xlabel('N-th nearest neighbor')
        if cnt == 0:
            axes[cnt].set_ylabel('Fraction of patches\nfrom the same core')
        else:
            axes[cnt].set_yticks([],[])

    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    if saveFig:
        pd.DataFrame(bestEpochsDict).to_csv(os.path.join(figuresDir,'retrieval_tma_fold0_'+modelstrList[0]+'.csv'))
        fig.savefig(os.path.join(figuresDir,'retrieval_tma_fold0_'+modelstrList[0]+'.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'retrieval_tma_fold0_'+modelstrList[0]+'.svg'),format='svg',dpi=600,bbox_inches='tight') 
    plt.show()

    fig, ax = plt.subplots(figsize=(5,5))
    model = 'BYOL_ViTb_TMA0_COMBINED_PATCH_fold0'
    values = mainEpochsDict[model]
    modelstrList = model.split('_')
    modelNameShort = modelstrList[1]+' / '+modelstrList[3]
    plt.plot(np.arange(1,nNeigh+1), values, label=modelNameShort,linewidth=5, color=cmap[modelNameShort])

    for pretrainedFileName in pretrainedModelsList:     
        bareValues = np.load(os.path.join(taskDir,pretrainedFileName+'_fold0.npy'),allow_pickle=True)
        bareValues = bareValues.item()[0]
        bareValues = np.mean(bareValues,axis=0)
        pretrainedName = pretrainedFileName.replace('_', ' / ')
        plt.plot(np.arange(1,nNeigh+1), bareValues, label=pretrainedName, linewidth=5, color=cmap[pretrainedName])
        
        bestResultsDirSimple[pretrainedName] = bareValues[-1]

    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_items = sorted(labels, key=lambda x: bestResultsDirSimple.get(x), reverse=True)
    sorted_handles = [handles[labels.index(lbl)] for lbl in sorted_items]
    plt.legend(sorted_handles,
                sorted_items,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                title='BACKBONE / INPUT',
                framealpha=1.0,
                facecolor='white')

    ax.set_ylim(0.5,1.0)
    plt.xlabel('N-th nearest neighbor')
    plt.ylabel('Fraction of patches\nfrom the same core')
    fig.patch.set_alpha(0.0)

    if saveFig:
        pd.DataFrame(bestEpochsDict).to_csv(os.path.join(figuresDir,'retrieval2_tma_fold0_'+modelstrList[0]+'.csv'))
        fig.savefig(os.path.join(figuresDir,'retrieval2_tma_fold0_'+modelstrList[0]+'.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'retrieval2_tma_fold0_'+modelstrList[0]+'.svg'),format='svg',dpi=600,bbox_inches='tight') 
    plt.show()

    return bestResultsDir


def get_histogram_ranks(methods,nNeighbors=10):
    """Calculate ranks of nearest neighbors from the same TMA core/grade.

    Args:
        methodsData (dictionary): calculated information about each feature profile, with its patch, TMA core information, and number of nearest neighbors from the same core/grade.
        nNeighbors (int, optional): number of neighbors to analyze. Defaults to 10.

    Returns:
        histPlotMelt (DataFrame): formatted information about statistis regarding patches having nearest neighbors from the same TMA core/grade.
    """

    histPlot = pd.DataFrame()
    for method in methods:
        
        cleanProfiles = method['profiles']

        # get randomly sampled distributions of cosine distances for different methods:
        numberOfPoints = 1000

        np.random.seed(20)
        chosenPointsIdx = np.random.choice(cleanProfiles.shape[0], numberOfPoints)
        np.random.seed(None)


        profilesSimDist = scipy.spatial.distance.pdist(cleanProfiles[chosenPointsIdx], 'cosine')
        profilesSimDist = np.sort(profilesSimDist)

        # get n random input patches:
        np.random.seed(10)
        randomInputPatches = np.random.choice(cleanProfiles.shape[0],nNeighbors)
        np.random.seed(None)

        for chosenPatchIdx in randomInputPatches:
            
            inputProfile = cleanProfiles[chosenPatchIdx]
            closestPatchesIdx = method['indices'][0][chosenPatchIdx][1:]
            closestPatches = cleanProfiles[closestPatchesIdx]
            distanceClosest = cosine_distances(inputProfile.reshape(1,-1),closestPatches)[0]
            distanceClosestRank = [np.searchsorted(profilesSimDist, i) for i in distanceClosest]
            distanceClosestRank = [(i / len(profilesSimDist)) for i in distanceClosestRank]
            
            sameCoreProfilesIdx = np.where(method['rows']==method['rows'][chosenPatchIdx])[0]
            sameCoreProfiles = cleanProfiles[sameCoreProfilesIdx]
            sameCoreProfiles = sameCoreProfiles[:nNeighbors,:]
            
            distanceSameCore = cosine_distances(inputProfile.reshape(1,-1),sameCoreProfiles)[0]
            distanceSameCoreRank = [np.searchsorted(profilesSimDist, i) for i in distanceSameCore]
            distanceSameCoreRank = [(i / len(profilesSimDist)) for i in distanceSameCoreRank]



            histPlot = histPlot.append(pd.DataFrame({'Method':list(),
                                                    'Nearest neighbors patches':list(distanceClosestRank),
                                                    'Patches from the same core':list(distanceSameCoreRank)}))

    return histPlot

# %% Check retrieval at 10th nearest neighbor.

def __main__(saveFig=False, runCal=False):

    if runCal:
        run_retrieval(modelsList)

    bestResultsDir = fetch_retrieval(modelsList, pretrainedModelsList, saveFig=saveFig)
    labelsOfInterest = [i for i in cmap.keys() if any(kw in i for kw in ['CLOSE','SAME','COMBINE'])]
    updatedLabelsDict = dict(zip([tuple(i.split(' / ')) for i in labelsOfInterest],labelsOfInterest))
    cmap2 = dict(zip([tuple(i.split(' / ')) for i in labelsOfInterest], [cmap[i] for i in labelsOfInterest]))

    splitKeys = [key.split(' / ') for key in bestResultsDir.keys()]
    structureData = [{'Method': key[0], 'Architecture': key[1], 'Type': key[2], 'Value': bestResultsDir[" / ".join(key)]} for key in splitKeys]

    resultsDf = pd.DataFrame(structureData)
    fig, ax = plt.subplots(figsize=(9,5))
    sns.barplot(data=resultsDf,x='Method',y='Value',hue=resultsDf[['Architecture','Type']].apply(tuple, axis=1),
                order=['BYOL','Triplet','MoCoV2'],palette=cmap2,hue_order=list(cmap2.keys()))
    plt.xlabel('')
    plt.ylabel('Fraction of patches\nfrom the same core')
    handles, labels = plt.gca().get_legend_handles_labels()
    updatedLabels = [updatedLabelsDict[ast.literal_eval(label)] for label in labels]
    plt.legend(handles, updatedLabels,
            loc='center left', bbox_to_anchor=(1, 0.5), 
            title='BACKBONE / INPUT',framealpha=1.0, facecolor='white')
    plt.title('Retrieval of 10 nearest neighbors')
    fig.patch.set_alpha(0.0)  
    if saveFig:
        resultsDf.to_csv(os.path.join(figuresDir,'10th_neigh_retrieval.csv'))
        fig.savefig(os.path.join(figuresDir,'10th_neigh_retrieval.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'10th_neigh_retrieval.svg'),format='svg',dpi=600,bbox_inches='tight') 
    plt.show()


# %%
