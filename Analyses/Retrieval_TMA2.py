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
import glob
import scipy
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import openslide as oSlide

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import Utils.Patch_Generation as pg
from Utils.Visualization import apply_plot_settings

with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %% Visualization settings, useful lists.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

paletteColors = [(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
                (0.8, 0.8, 0.8)]

# %% Visualization settings, helper functions, useful lists.

def get_nearest_neighbors(nNeighbors=10, runCal=False):
    """Featch information about the nearest neighbors.

    Args:
        nNeighbors (int, optional): number of neighbors to analyze. Defaults to 10.

    Returns:
        methodsData (dictionary): calculated information about each feature profile, with its patch, TMA core information, and number of nearest neighbors from the same core/grade.
        punchInfo (DataFrame): metadata for each TMA core.
    """

    # load profiles using all the models:
    patchDir=files['DATA']['tma']['tma2']
    hdf5List=glob.glob(os.path.join(patchDir,'*.hdf5'))

    patchData,patchClasses,excelRowToClassDict,_=pg.LoadPatchData(hdf5List,returnSampleNumbers=True)    
    patchData=patchData[0]

    patchesProfiles = []
    patchesProfilesUNI = []

    for i in hdf5List:
        fileName = i.split('/')[-1].split('.')[0]
        profiles = np.load(os.path.join(files['FEATURES']['morphoith'],'TMA2','AllFolds', fileName+'.npy'))
        patchesProfiles.append(profiles)
        profilesUNI = np.load(os.path.join(files['FEATURES']['main'],'BARE_MODELS','ViTl_UNI','TMA2', fileName+'.npy'))
        patchesProfilesUNI.append(profilesUNI)
        
    patchesProfiles = np.vstack(patchesProfiles)
    patchesProfilesUNI = np.vstack(patchesProfilesUNI)

    nLabels=len(np.unique([excelRowToClassDict[k] for k in excelRowToClassDict]))
    classToExcelRow= np.zeros(nLabels,dtype=np.uint16)
    for k in excelRowToClassDict:
        classToExcelRow[excelRowToClassDict[k]]=int(k)
    
    patchExcelRows=classToExcelRow[np.int16(patchClasses)]

    # exclude blacklisted slides, get metadata, discard cores with less than 100 patches:
    uniquePunchLabels, patchesCount = np.unique(patchExcelRows, return_counts=True)
    punchInfoFile=files['METADATA']['tma2']
    punchInfo = pd.read_csv(punchInfoFile)
    punchInfo = punchInfo.loc[uniquePunchLabels]

    punchInfo['BlackListed'] = punchInfo['BlackListed'] | punchInfo['Patient in training']

    allPatchesGrades = np.asarray([punchInfo.loc[i]['Grades_Payal'].item() for i in patchExcelRows])   
    allPatchesBlackslited = np.asarray([punchInfo.loc[i]['BlackListed'].item() for i in patchExcelRows])  
    allSlides = np.asarray([punchInfo.loc[i]['SVS'] for i in patchExcelRows])    

    boolBlacklisted = np.array([False if allPatchesBlackslited[i] else True for i in range(allPatchesBlackslited.shape[0])])
    boolGrade = np.array([False if allPatchesGrades[i]==0 else True for i in range(allPatchesGrades.shape[0])])
    boolPatchesCount = np.array([False if patchesCount[np.where(uniquePunchLabels==i)[0]][0] < 100 else True for i in patchExcelRows])
    boolCombined = boolBlacklisted * boolGrade * boolPatchesCount

    cleanPatchesGrades = allPatchesGrades[boolCombined]
    cleanPatchesProfiles = patchesProfiles[boolCombined]
    cleanPatches = patchData[boolCombined]
    cleanPatchExcelRows = patchExcelRows[boolCombined]
    cleanPatches = patchData[boolCombined,:]
    cleanPatchesProfilesUNI = patchesProfilesUNI[boolCombined]

    if runCal:
        np.save(os.path.join(figuresDir, 'tmaTesting_profilesMorphoITH'), cleanPatchesProfiles)
        np.save(os.path.join(figuresDir, 'tmaTesting_profilesUNI'), cleanPatchesProfilesUNI)

    # get nearest neighbors for all the methods:
    _,punchNumbers=np.unique(cleanPatchExcelRows,return_inverse=True)

    config = {'method':'MorphoITH', 'profiles':cleanPatchesProfiles, 'color':'#8B4513', 'indices':[], 'meanSamePunch':[], 'meanSameGrade':[], 'indices':[], 'meanSamePunch':[], 'meanSameGrade':[]}

    profiles = config['profiles']
    nbrs = NearestNeighbors(n_neighbors=nNeighbors+1, algorithm='auto', metric='cosine',n_jobs=16).fit(profiles)
    _,indices = nbrs.kneighbors(profiles)
    config['indices'].append(indices)

    inSamePunch = np.equal(punchNumbers[indices[:,1:]],np.expand_dims(punchNumbers[indices[:,0]],axis=1))
    sameGrade = np.equal(cleanPatchesGrades[indices[:,1:]],np.expand_dims(cleanPatchesGrades[indices[:,0]],axis=1))

    meanSamePunch = np.mean(inSamePunch,axis=0)
    meanSameGrade = np.mean(sameGrade,axis=0)
    config['meanSamePunch'].append(meanSamePunch)
    config['meanSameGrade'].append(meanSameGrade)

    return config, punchInfo, cleanPatchExcelRows, cleanPatchesGrades, punchNumbers, cleanPatches

def get_histogram_ranks(methodsData,nNeighbors=10,rows=None, grades=None):
    """Calculate ranks of nearest neighbors from the same TMA core/grade.

    Args:
        methodsData (dictionary): calculated information about each feature profile, with its patch, TMA core information, and number of nearest neighbors from the same core/grade.
        nNeighbors (int, optional): number of neighbors to analyze. Defaults to 10.

    Returns:
        histPlotMelt (DataFrame): formatted information about statistis regarding patches having nearest neighbors from the same TMA core/grade.
    """

    histPlot = pd.DataFrame()
    cleanProfiles = methodsData['profiles']

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
        closestPatchesIdx = methodsData['indices'][0][chosenPatchIdx][1:]
        closestPatches = cleanProfiles[closestPatchesIdx]
        distanceClosest = cosine_distances(inputProfile.reshape(1,-1),closestPatches)[0]
        distanceClosestRank = [np.searchsorted(profilesSimDist, i) for i in distanceClosest]
        distanceClosestRank = [(i / len(profilesSimDist)) for i in distanceClosestRank]
        
        sameCoreProfilesIdx = np.where(rows==rows[chosenPatchIdx])[0]
        sameCoreProfiles = cleanProfiles[sameCoreProfilesIdx]
        sameCoreProfiles = sameCoreProfiles[:nNeighbors,:]
        
        randomSameGradeIdx = np.where((grades==grades[chosenPatchIdx])&(rows!=rows[chosenPatchIdx]))[0]
        randomSameGradeIdx = randomSameGradeIdx[np.linspace(0, len(randomSameGradeIdx) - 1, nNeighbors, dtype=int)]
        randomSameGrade = cleanProfiles[randomSameGradeIdx]
        
        differentCoreAndGradeIdx = np.where((grades!=grades[chosenPatchIdx])&(rows!=rows[chosenPatchIdx]))[0]
        differentCoreAndGradeIdx = differentCoreAndGradeIdx[np.linspace(0, len(differentCoreAndGradeIdx) - 1, nNeighbors, dtype=int)]
        differentCoreAndGrade = cleanProfiles[differentCoreAndGradeIdx]
        
        distanceSameGrade = cosine_distances(inputProfile.reshape(1,-1),randomSameGrade)[0]
        distanceSameGradeRank = [np.searchsorted(profilesSimDist, i) for i in distanceSameGrade]
        distanceSameGradeRank = [(i / len(profilesSimDist)) for i in distanceSameGradeRank]
        
        distanceSameCore = cosine_distances(inputProfile.reshape(1,-1),sameCoreProfiles)[0]
        distanceSameCoreRank = [np.searchsorted(profilesSimDist, i) for i in distanceSameCore]
        distanceSameCoreRank = [(i / len(profilesSimDist)) for i in distanceSameCoreRank]
        
        distanceDifferentBoth = cosine_distances(inputProfile.reshape(1,-1),differentCoreAndGrade)[0]
        distanceDifferentBothRank = [np.searchsorted(profilesSimDist, i) for i in distanceDifferentBoth]
        distanceDifferentBothRank = [(i / len(profilesSimDist)) for i in distanceDifferentBothRank]


        histPlot = histPlot.append(pd.DataFrame({'Nearest neighbors patches':list(distanceClosestRank),
                                                 'Patches from the same core':list(distanceSameCoreRank),
                                                 'Patches with the same grade':list(distanceSameGradeRank),
                                                 'Patches from different core and grade':list(distanceDifferentBothRank)}))

    histPlotMelt = histPlot.melt(value_vars=['Nearest neighbors patches', 'Patches from the same core', 'Patches with the same grade','Patches from different core and grade'], 
                                 var_name='Subgroup', value_name='Rank of cosine distance')
    return histPlotMelt

def calculate_feature_representation(data):
    """Get feature vector closest to the mean feature vector from the sample.

    Args:
        data (array): featue profiles from the sample.

    Returns:
        representative_feature (array): representative feature profile from the sample.
    """

    mean_feature = np.mean(data, axis=0)
    distances = np.linalg.norm(data - mean_feature, axis=1)
    representative_index = np.argmin(distances)
    representative_feature = data[representative_index]
        
    return representative_feature

def save_tma_distances(config, punchInfo, runCal=False, saveFig=False):
    """Calculate and save all cosine distanced between patches within TMA cores.

    Args:
        config (dictionary): calculated information about each feature profile, with its patch, TMA core information, and number of nearest neighbors from the same core/grade.
        punchInfo (_type_): _description_
        runCal (DataFrame): allows for saving the calcualtions. Defaults to False.
        saveFig (bool): allows for saving figures as .svg and .png. Defaults to False.
    """

    uName,uNums=np.unique(config['rows'],return_inverse=True)
    inPunchDistances=[]
    repInPunchDistances=[]
    punchRowsList=[]
    medianInPunchDist=[]
    for i in range(len(uName)):
        isInPunch=uNums==i
        punchProfiles=config['profiles'][isInPunch,:]
        punchRows=config['rows'][isInPunch]
        distances = pdist(punchProfiles,metric='cosine')
        inPunchDistances.append(distances)
        repInPunchDistances.append(calculate_feature_representation(punchProfiles))
        punchRowsList.append(punchRows)
        medianInPunchDist.append(np.median(distances))

    inPunchDistances=np.concatenate(inPunchDistances)
    repInPunchDistances=np.vstack(repInPunchDistances)

    if runCal:
        np.save(os.path.join(figuresDir,'heterogeneity','tmaDistancesMedian.npy'),medianInPunchDist)
    
    # plot examples of TMA cores:
    idxMax = punchRowsList[np.where(medianInPunchDist==np.max(medianInPunchDist))[0][0]][0]
    idxMin = punchRowsList[np.where(medianInPunchDist==np.min(medianInPunchDist))[0][0]][0]

    for idx in [idxMax, idxMin]:
        pointPd = punchInfo.loc[idx]
        svsFile = os.path.join(files['SLIDES']['tma2'],pointPd['SVS'])
        slide = oSlide.open_slide(svsFile)

        margin = 300
        regionA = slide.read_region((pointPd['CornerX']-margin,pointPd['CornerY']-margin), 0, (pointPd['PunchSize']+2*margin,pointPd['PunchSize']+2*margin))

        fig = plt.figure()
        plt.imshow(np.array(regionA))
        plt.xticks([],[])
        plt.yticks([],[])
        fig.patch.set_alpha(0.0)
        if idx == idxMax:
            saveName = '_highest'
        elif idx == idxMin:
            saveName = '_lowest'
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'tma_example_'+saveName+'.png'),bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'tma_example_'+saveName+'.svg'),format='svg',dpi=600,bbox_inches='tight')  
        plt.show()

# %%

def __main__(saveFig=False, runCal=False):

    nNeighbors = 10
    config, punchInfo, rows, grades, cores, patches = get_nearest_neighbors(nNeighbors,runCal)
    if runCal:
        save_tma_distances(config, punchInfo, saveFig=saveFig, runCal=runCal)

    # plot nearest neighbors fraction. 
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(1,nNeighbors+1), config['meanSameGrade'][0],linewidth=3, color='black')
    plt.plot(np.arange(1,nNeighbors+1), config['meanSamePunch'][0],linewidth=3, color='black')

    plt.fill_between(np.arange(1,nNeighbors+1), config['meanSameGrade'][0], 1.01, color=paletteColors[2], alpha=1, label='Different grade and core')
    plt.fill_between(np.arange(1,nNeighbors+1), config['meanSameGrade'][0], config['meanSamePunch'][0], color=paletteColors[1],  alpha=1, label='Same grade, different core') 
    plt.fill_between(np.arange(1,nNeighbors+1), 0.5, config['meanSamePunch'][0], color=paletteColors[0], alpha=.75, label='Same grade and core') 

    plt.yticks(np.arange(0.5,1.01,0.1))
    plt.xlabel('N-th nearest neighbor')
    plt.ylabel('Fraction of patches\nfrom the same source')
    legend = fig.legend(loc='lower left', bbox_to_anchor=(.17,.15),framealpha=1.0, facecolor='white',fontsize=13)
    fig.patch.set_alpha(0.0)
    if saveFig:
        pd.DataFrame({'Same punch':config['meanSamePunch'][0], 'Same grade':config['meanSameGrade'][0]}).to_csv(os.path.join(figuresDir,'neigh_vs_grade.png'))
        fig.savefig(os.path.join(figuresDir,'neigh_vs_grade.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'neigh_vs_grade.svg'),format='svg',dpi=600,bbox_inches='tight') 
    plt.show()

    # visualization of nearest neighbors.
    chosenPatch = 1000
    indices = config['indices'][0]

    fig, axes = plt.subplots(2,5, figsize=(22,9))
    for i,ax in enumerate(axes.flat):
        if i ==0:
            ax.imshow(patches[chosenPatch])
            ax.annotate("G{}, input core (N={})".format(grades[chosenPatch],i), xy=(0, 1.1), xycoords='axes fraction', 
            fontsize=24, color='black',
            horizontalalignment='left', verticalalignment='top')
            ax.axis('off')
        else:
            isSameCore = cores[chosenPatch] == cores[indices[chosenPatch,i]]
            if isSameCore:
                isSameCore = 'same core'
            else:
                isSameCore = 'other core'
            ax.imshow(patches[indices[chosenPatch,i]])
            ax.annotate("G{}, {} (N={})".format(grades[indices[chosenPatch,i]], isSameCore,i), 
            xy=(0, 1.1), xycoords='axes fraction', fontsize=24, color='black',
            horizontalalignment='left', verticalalignment='top')
            ax.axis('off')
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_visual.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_visual.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.show()

    # similarity distributions.
    histPlotMelt = get_histogram_ranks(config, nNeighbors, rows=rows, grades=grades)
    fig = plt.figure(figsize=(5,5))
    sns.boxplot(x='Subgroup', y='Rank of cosine distance', data=histPlotMelt, fliersize=0, width=0.6, palette=paletteColors, order=['Patches from the same core', 'Patches with the same grade','Patches from different core and grade'])
    plt.xticks([])
    plt.xlabel('')

    for patch in legend.get_patches():
        patch.set_edgecolor('black')
    fig.patch.set_alpha(0.0)
    if saveFig:
        histPlotMelt.to_csv(os.path.join(figuresDir,'retrieval_tma2_cosine_distances.csv'))
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_cosine_distances.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_cosine_distances.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.show()

    # t-sne of the patches: calculations:
    np.random.seed(1)
    nRand = 10000
    randIdx = np.random.randint(config['profiles'].shape[0],size=nRand)
    profilesToView = config['profiles'][randIdx,:]
    punchesSelected = rows[randIdx]
    gradesSelected = grades[randIdx]
    punchesImagesSelected = patches[randIdx]

    tsne = TSNE(init='random', n_components=2, random_state=42, learning_rate=200, square_distances=True)
    X_tsne = tsne.fit_transform(profilesToView)
    _,punchIds = np.unique(punchesSelected,return_inverse=True)

    # t-sne of the patches: plot:
    fig, ax = plt.subplots(figsize=(7,7))
    sns.scatterplot(x=X_tsne[:,0],y=X_tsne[:,1], hue=gradesSelected,palette={1: 'darkgreen',2: 'gold', 3: 'coral', 4: 'maroon'})
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Colored by grade\n')
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_tsne_grade.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_tsne_rade.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.show()

    # highlight chosen core:
    chosenCore = 5
    fig, ax = plt.subplots(figsize=(7,7))
    isInPunch=punchIds==chosenCore
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=punchesSelected, palette='tab10', legend=None)

    xs = [int(X_tsne[:,0].min())-10, int(X_tsne[:,0].max())+10, int(X_tsne[:,0].max())+10, int(X_tsne[:,0].min())-10, int(X_tsne[:,0].min())-10]
    ys = [int(X_tsne[:,1].min())-10, int(X_tsne[:,1].min())-10, int(X_tsne[:,1].max())+10, int(X_tsne[:,1].max())+10, int(X_tsne[:,1].min())-10]
    plt.fill(xs, ys, color="white",alpha=0.75,)

    sns.scatterplot(x=X_tsne[isInPunch,0],y=X_tsne[isInPunch,1], color='red', legend=None)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Highlighted random core\n')
    fig.patch.set_alpha(0.0)
    if saveFig:
        tsneDf = pd.DataFrame({'TSNE_X':X_tsne[:,0],'TSNE_Y':X_tsne[:,1],'Grade':gradesSelected,'Core':punchesSelected})
        tsneDf.to_csv(os.path.join(figuresDir,'retrieval_tma2_tsne_grade_core.csv'))
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_tsne_core.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir,'retrieval_tma2_tsne_core.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.show()

# %%

if __name__ == "__main__":
    
    __main__()
    
