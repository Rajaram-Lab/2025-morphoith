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
import itertools
import glob as glob
import seaborn as sns
import numpy as np
import pandas as pd
import openslide as oSlide
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

os.chdir(os.path.dirname(os.path.dirname(__file__)))

from Utils.Image_Utils import MaskFromXML,GetQPathTextAnno
from Utils.Correct_Annotations import CheckAnnotations
from Utils.Controls import MakeHalfMask, MakeCircleMaskOnTumor
from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Separation import GetAccuracies
from Utils.Visualization import apply_plot_settings

with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %% Visualization settings, helper functions, useful dictionaries.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

# get input slides that have been annotated by pathologists:
inputPilotSamplesList = pd.read_csv(files['DATASETS']['pilot'])
inputPilotSamplesList = inputPilotSamplesList[inputPilotSamplesList['Annotations']==True]['Name'].tolist()
inputCd31SamplesList = pd.read_csv(files['DATASETS']['cd31'])
inputCd31SamplesList = inputCd31SamplesList[inputCd31SamplesList['Annotations']==True]['Name'].tolist()
inputSamplesList = inputPilotSamplesList+inputCd31SamplesList

# get paths to slides for pilot:
basePilotDir = files['SLIDES']['pilot']
pilotHneDirs = ['Batch1/HnE/','Batch2/HnE/','Batch3/']
fileToPathDict={}
for d in pilotHneDirs:
    pilotFiles=glob.glob(os.path.join(basePilotDir,d,'*.svs'))
    for f in pilotFiles:
        fileToPathDict[os.path.split(f)[-1].split('.')[0]]=f

# get colors for annotations:
gradeColorDict={'Tumor':'lightgrey','1':'darkgreen','2':'lightgreen','3':'orange','4':'red','High':'orange', 'Low':'darkgreen'}
archColorDict={'Tumor':'lightgrey',
               'Papillary':'purple',
               'Large nest':'olive','Small nest':'darkgreen',
               'Large nest-Alveolar':'lightcoral','Alveolar':'violet',
               'Large nest-Trabecular':'gold', 'Trabecular':'orange',
               'Macrocystic':'steelblue','Solid':'darkred',
               'Solid-trabecular':'pink', 'BleedingFollicle':'red',
               'Nest':'darkgreen','TubuloPapillary':'lightpink','Rhabdoid':'thistle','Alveolar-papillary':'hotpink'}
pal=sns.color_palette("Paired", 11)

# %% 

def get_arrays(samplesList, uni=False, saveFig=False):
    """Given a samples list, extract corresponding MorphoITH profiles, tissue classifier output, pen marks exclusions, and pathologists annotations.

    Args:
        samplesList (list): list of samples for which there are pathological annotations about areas with specific grade/architecture available.
        saveFig (bool, optional): allows for saving figures as .svg and .png. Defaults to False.

    Returns:
        profilesDict (dictionary): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDict (dictionary): arrays of tumor mask with excluded markers (value) for each sample (key).
        labelsDict (dictionary): arrays of pathologists annotations mask (value) for each sample (key).
        tumorDict (dictionary): arrays of tumor mask  (value) for each sample (key).
    """
    profilesDict = {}
    masksDict = {}
    labelsDict = {}
    tumorDict = {}

    for name in samplesList:
        name = str(name)
        
        if name[:2] == 'KC':
            dataset = 'pilot'
            svsFile = fileToPathDict[name]
        else:
            dataset = 'cd31'
            svsFile = os.path.join(files['SLIDES']['cd31'],name+'.svs')

        profilesFile = os.path.join(files['MASKS']['morphoith'],dataset,name+'.npy')
        if uni:
            profilesFile = os.path.join(files['MASKS']['uni'],dataset,name+'.npy')
        profiles = np.load(profilesFile,allow_pickle=True)
        slide = oSlide.open_slide(svsFile)

        level=min(2,slide.level_count-1)
        imgLowRes=np.array(slide.read_region((0,0), level, slide.level_dimensions[level]))[:,:,range(3)]

        decX = int(slide.dimensions[1]/profiles.shape[0])
        decY = int(slide.dimensions[0]/profiles.shape[1])
        assert decX == decY

        annoFile = os.path.join(files['MASKS']['annotations'],name+'.txt')
        _,regionNames,_,_=GetQPathTextAnno(annoFile)
        annoMaskOrg,classDict=MaskFromXML(annoFile,[], slideDim=slide.dimensions,downSampleFactor=slide.level_downsamples[level])
        annoMask = cv2.resize(annoMaskOrg,(profiles.shape[1],profiles.shape[0]))

        classDict, regionNames = CheckAnnotations(name, classDict, regionNames)
        gradesDir = {}
        architecturesDir = {}

        for key, value in classDict.items():
            [n.rsplit('-',1)[0][1:] for n in regionNames if n!='+null']
            if value != '+null':
                grade = value.rsplit('-',1)[1]
                arch = value.rsplit('-',1)[0][1:]
            else:
                grade = np.nan
                arch = np.nan
            gradesDir[key] = int(grade)
            architecturesDir[key] = arch
        labelsDict[name] = (gradesDir,architecturesDir)

        tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],dataset,name+'.npy')
        tumor = np.load(tissueMaskFile)
        tumor = np.uint8(SmoothResponseAndClasses(tumor,smoothSize=0)[0])
        tumor[tumor != 5] = 0
        tumor[tumor == 5] = 1
        tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
        tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))

        annoMask = annoMask * tumor

        if dataset == 'pilot':
            maskFile = os.path.join(files['MASKS']['markers'][dataset], name+'.svs_mask.png')
            markers = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
            markers = cv2.resize(markers,(profiles.shape[1],profiles.shape[0]))
            markers[markers==3]=4
            markers[markers!=4]=0
            markers[markers==4]=1
            tumorNoMarkers = cv2.subtract(tumor, markers)
            annoMaskNoMarkers = annoMask * tumorNoMarkers
        else:
            tumorNoMarkers = tumor.copy()
            annoMaskNoMarkers = annoMask.copy()
            
        profilesDict[name] = profiles
        masksDict[name] = annoMaskNoMarkers
        tumorDict[name] = tumorNoMarkers

        #*#*# VISUALIZATION #*#*#
        slideToVisualize = slidesToVisualize[slidesToVisualize['Reason']=='Annotations']['Name'].tolist()
        if (name in slideToVisualize) & (uni != True):

            ### ARCHITECTRURE ###
            archColors = [archColorDict[architecturesDir[i]] for i in architecturesDir]
            sortedArchs = list(architecturesDir.values())
            sortedArchs.sort()
            patches = [mpatches.Patch(color=archColorDict[i], label=i) for i in sortedArchs]
            fig = plt.figure(figsize=(7,10))
            plt.imshow(tumor,cmap=ListedColormap(['white','lightgrey']),interpolation='none')
            plt.imshow(np.ma.array(annoMask, mask=[annoMask==0]),cmap=ListedColormap(archColors),interpolation='none')
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.0), loc='upper left',framealpha=1.0, facecolor='white')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'arch_grade_map_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'arch_grade_map_'+name+'.svg'),dpi=600, bbox_inches='tight')
            plt.show()

            ### H&E ###
            fig = plt.figure(figsize=(7,10))
            plt.imshow(imgLowRes)
            plt.xticks([],[])
            plt.yticks([],[])
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.png'), format='svg', dpi=600, bbox_inches='tight')
            plt.show()

            ### SCATTERPLOT ###
            np.random.seed(12345)
            rnIdx = np.random.choice(annoMaskNoMarkers[annoMaskNoMarkers!=0].shape[0],600)
            archs = [architecturesDir[i] for i in annoMaskNoMarkers[annoMaskNoMarkers!=0]]
            grades = [gradesDir[i] for i in annoMaskNoMarkers[annoMaskNoMarkers!=0]]

            tsnePos= TSNE(init='random', n_components=2, metric='cosine', perplexity=30, random_state=1, learning_rate=200, square_distances=True).fit_transform(profiles[annoMaskNoMarkers!=0][rnIdx])
            tsneDf=pd.DataFrame({'t-SNE 1':tsnePos[:,0],
                                 't-SNE 2':tsnePos[:,1],
                                 'Architecture':np.array(archs)[rnIdx],
                                 'Grade':np.array(grades)[rnIdx]})
            
            markersDict = {1:'X',2:'o',3:'^',4:'s'}
            markersList = [markersDict[int(g)] for g in np.unique(np.array(grades)[rnIdx])]

            fig = plt.figure(figsize=(7,5))
            sns.scatterplot(data=tsneDf,x='t-SNE 1',y='t-SNE 2',hue='Architecture',style='Grade',palette=archColorDict,s=100,markers=markersList)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',framealpha=1.0, facecolor='white')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                tsneDf.to_csv(os.path.join(figuresDir, 'arch_grade_scatterplot_'+name+'.csv'), index=[0])
                fig.savefig(os.path.join(figuresDir, 'arch_grade_scatterplot_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'arch_grade_scatterplot_'+name+'.svg'), format='svg', dpi=600, bbox_inches='tight')
                tsneDf.to_csv()
            plt.show()

    return profilesDict, masksDict, labelsDict, tumorDict


def run_masks(profilesDictionary, masksDictionary, labelsDictionary, tumorsDictionary, numberOfRuns=10):
    """Given morphological profiles and necessary masks (tumor, pen marks, annotations), yield separation accuracies.

    Args:
        profilesDictionary (dictionary): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDictionary (dictionary): arrays of tumor mask with excluded markers (value) for each sample (key).
        labelsDictionary (dictionary): arrays of pathologists annotations mask (value) for each sample (key).
        tumorsDictionary (dictionary): arrays of tumor mask  (value) for each sample (key).

    Returns:
        gradeResults (dictionary): separation measures (accuracies), including for negative controls, with information about overlap and grade change for each sample.
        archResults (dictionary): separation measures (accuracies), including for negative controls, with information about overlap and grade change for each sample.
    """  

    gradeResults = {'Accuracy':{}, 'Sample':{}, 'Negative control (circle)':{},
                    'Negative control (split)':{}, 'Grade change':{},
                    'Overlap A':{}, 'Overlap B':{}}
    archResults = {'Accuracy':{}, 'Sample':{}, 'Negative control (circle)':{},
                    'Negative control (split)':{}, 'Grade change':{},
                    'Overlap A':{}, 'Overlap B':{}}
    
    for name in list(profilesDictionary.keys()):
        name = str(name)

        profiles = profilesDictionary[name]
        mask = masksDictionary[name]
        tumor = tumorsDictionary[name]
        gradesDir,architecturesDir = labelsDictionary[name]

        if np.unique(mask).shape[0] < 2:
            continue
        
        annoInfo = {'grades':[gradesDir,gradeResults], 'architectures':[architecturesDir,archResults]}

        for annoType in annoInfo:
            annoDir = annoInfo[annoType][0]
            annoResults = annoInfo[annoType][1]
            howManyPoints = 300
            
            uniqueAnno = np.unique(list(annoDir.values()))
            if uniqueAnno.shape[0] >= 2:
                uniqueParis = list(itertools.combinations(uniqueAnno, 2))
                
                for cnt, pair in enumerate(uniqueParis):

                    partA = pair[0]
                    partB = pair[1]

                    uniqueWithPartA = [key for key, value in annoDir.items() if value == partA]
                    uniqueWithPartB = [key for key, value in annoDir.items() if value == partB]
                    uniqueWithBoth = uniqueWithPartA + uniqueWithPartB
                    
                    pairMask = mask.copy()
                    pairMask[~np.isin(pairMask, uniqueWithBoth)] = 0
                    mask_partA = np.isin(pairMask, uniqueWithPartA) 
                    mask_partB = np.isin(pairMask, uniqueWithPartB) 

                    pairMask[(mask_partA) & (pairMask != 1)] = 1 
                    pairMask[(mask_partB) & (pairMask != 2)] = 2  

                    pairMaskFlat = pairMask[pairMask!=0]
                    profilesFlat = profiles[pairMask!=0]

                    p1Idx = np.where(pairMaskFlat==1)[0]
                    p2Idx = np.where(pairMaskFlat==2)[0]
                    
                    np.random.seed((cnt+1)*123)
                    p1IdxSubset = np.random.choice(p1Idx,howManyPoints)
                    np.random.seed((cnt+1)*321)
                    p2IdxSubset = np.random.choice(p2Idx,howManyPoints)
                    
                    idxSubset = np.concatenate([p1IdxSubset,p2IdxSubset])
                    labelSubset = pairMaskFlat[idxSubset]
                    profilesSubset = profilesFlat[idxSubset]

                    profilesPcaSubset = PCA(n_components=10).fit_transform(profilesSubset)
                    acc = GetAccuracies(profilesPcaSubset, labelSubset)

                    pair = str(pair[0])+' / '+str(pair[1])
                    if pair not in annoResults['Accuracy'].keys():
                        annoResults['Accuracy'][pair] = []
                        annoResults['Sample'][pair] = []
                        annoResults['Grade change'][pair] = []
                        annoResults['Negative control (circle)'][pair] = []
                        annoResults['Negative control (split)'][pair] = []
                        annoResults['Overlap A'][pair] = []
                        annoResults['Overlap B'][pair] = []
                        
                    annoResults['Accuracy'][pair].append(acc)
                    annoResults['Sample'][pair].append(name)
                    
                    # is there a grade change?
                    gradesPairA = [gradesDir[key] for key in uniqueWithPartA]
                    gradesPairB = [gradesDir[key] for key in uniqueWithPartB]
                    
                    if (np.unique(gradesPairA) != np.unique(gradesPairB))[0] or (np.unique(gradesPairA).shape[0] > 1 or np.unique(gradesPairB).shape[0] > 1):
                        annoResults['Grade change'][pair].append('Grade change')
                    else:
                        annoResults['Grade change'][pair].append('No grade change')
                    

                    ### NEGATIVE CONTROLS ###
                    pairHalfMasks, halfOverlapA, halfOverlapB = MakeHalfMask(pairMask,numberOfRuns)
                    pairCircleMasks, circleOverlapA, circleOverlapB = MakeCircleMaskOnTumor(pairMask, tumor,numberOfRuns)
                    
                    randomMethods = {'circle':[pairCircleMasks, 'Negative control (circle)', circleOverlapA, circleOverlapB], 
                                    'split': [pairHalfMasks, 'Negative control (split)', halfOverlapA, halfOverlapB]}
                        
                    for randomMethod in randomMethods:

                        for i in range(numberOfRuns):
                            randomMask = randomMethods[randomMethod][0][i]
                            randomMaskFlat = randomMask[randomMask!=0]
                            randomP1Idx =  np.where(randomMaskFlat==1)[0]
                            randomP2Idx = np.where(randomMaskFlat==2)[0]
                            
                            np.random.seed((cnt+1)*(i+1)*123)
                            randomP1IdxSubset = np.random.choice(randomP1Idx,howManyPoints)
                            np.random.seed((cnt+1)*(1+i)*321)
                            randomP2IdxSubset = np.random.choice(randomP2Idx,howManyPoints)
                            
                            randomIdxSubset = np.concatenate([randomP1IdxSubset,randomP2IdxSubset])
                            randomLabelSubset = randomMaskFlat[randomIdxSubset]
                            randomProfilesSubset = profilesFlat[randomIdxSubset]

                            randomProfilesPcaSubset = PCA(n_components=10).fit_transform(randomProfilesSubset)
                            r_acc = GetAccuracies(randomProfilesPcaSubset, randomLabelSubset)
                            annoResults[randomMethods[randomMethod][1]][pair].append(r_acc)
                            annoResults['Overlap A'][pair].append(randomMethods[randomMethod][2])
                            annoResults['Overlap B'][pair].append(randomMethods[randomMethod][3])

    return gradeResults, archResults


# %% 

def __main__(saveFig=False, numberOfRuns=10, uni=False):

    profilesDict, masksDict, labelsDict, tumorDict = get_arrays(inputSamplesList, saveFig=saveFig, uni=uni)
    gradeResults, archResults = run_masks(profilesDict, masksDict, labelsDict, tumorDict, numberOfRuns)

    if uni:
        suffix = '_uni'
    else:
        suffix = ''

    gradeNormDf = pd.Series(gradeResults['Accuracy'], name='Separation').rename_axis('Pair').explode().reset_index()
    gradeNormSampleDf = pd.Series(gradeResults['Sample'], name='Sample').rename_axis('Pair').explode().reset_index()
    assert gradeNormDf['Pair'].all() == gradeNormSampleDf['Pair'].all()
    gradeNormDf['Sample'] = gradeNormSampleDf['Sample']


    archsNormDf = pd.Series(archResults['Accuracy'], name='Separation').rename_axis('Pair').explode().reset_index()
    archsNormSampleDf = pd.Series(archResults['Sample'], name='Sample').rename_axis('Pair').explode().reset_index()
    archsNormGradeChangeDf = pd.Series(archResults['Grade change'], name='Grade change').rename_axis('Pair').explode().reset_index()

    assert archsNormDf['Pair'].all() == archsNormSampleDf['Pair'].all()
    assert archsNormDf['Pair'].all() == archsNormGradeChangeDf['Pair'].all()

    archsNormDf['Sample'] = archsNormSampleDf['Sample']
    archsNormDf['Grade change'] = archsNormGradeChangeDf['Grade change']

    # GRADES:
    gradeAverageCircle = {}
    gradeAverageSplit = {}
    gradeCircleGrouped = []
    gradeSplitGrouped = []

    for key, value in gradeResults['Negative control (circle)'].items():
        averaged_value = [np.mean(value[i:i+10]) for i in range(0, len(value), 10)]
        gradeAverageCircle[key] = averaged_value
        gradeCircleGrouped.extend([value[i:i+10] for i in range(0, len(value), 10)])

    for key, value in gradeResults['Negative control (split)'].items():
        averaged_value = [np.mean(value[i:i+10]) for i in range(0, len(value), 10)]
        gradeAverageSplit[key] = averaged_value
        gradeSplitGrouped.extend([value[i:i+10] for i in range(0, len(value), 10)])

    gradeCircleDf = pd.Series(gradeAverageCircle, name='Negative control (circle, average)').rename_axis('Pair').explode().reset_index()
    gradeCircleDf['All values'] = gradeCircleGrouped

    gradeHalfDf = pd.Series(gradeAverageSplit, name='Negative control (split, average)').rename_axis('Pair').explode().reset_index()
    gradeHalfDf['All values'] = gradeSplitGrouped

    assert gradeNormDf['Pair'].all() == gradeCircleDf['Pair'].all()
    gradeNormDf['Negative control (circle, average)'] = gradeCircleDf['Negative control (circle, average)']
    assert gradeNormDf['Pair'].all() == gradeHalfDf['Pair'].all()
    gradeNormDf['Negative control (split, average)'] = gradeHalfDf['Negative control (split, average)']

    # ARCHITECTURES:
    archAverageCircle = {}
    archAverageSplit = {}
    archCircleGrouped = []
    archSplitGrouped = []

    for key, value in archResults['Negative control (circle)'].items():
        averaged_value = [np.mean(value[i:i+10]) for i in range(0, len(value), 10)]
        archAverageCircle[key] = averaged_value
        archCircleGrouped.extend([value[i:i+10] for i in range(0, len(value), 10)])

    for key, value in archResults['Negative control (split)'].items():
        averaged_value = [np.mean(value[i:i+10]) for i in range(0, len(value), 10)]
        archAverageSplit[key] = averaged_value
        archSplitGrouped.extend([value[i:i+10] for i in range(0, len(value), 10)])

    archsCircleDf = pd.Series(archAverageCircle, name='Negative control (circle, average)').rename_axis('Pair').explode().reset_index()
    archsCircleDf['All values'] = archCircleGrouped
    archsHalfDf = pd.Series(archAverageSplit, name='Negative control (split, average)').rename_axis('Pair').explode().reset_index()
    archsHalfDf['All values'] = archSplitGrouped

    assert archsNormDf['Pair'].all() == archsCircleDf['Pair'].all()
    archsNormDf['Negative control (circle, average)'] = archsCircleDf['Negative control (circle, average)']
    assert archsNormDf['Pair'].all() == archsHalfDf['Pair'].all()
    archsNormDf['Negative control (split, average)'] = archsHalfDf['Negative control (split, average)']

    # get combined grades:
    pairsDict = {'1 / 2':np.nan, '2 / 3': 'Low / high', '3 / 4': np.nan,
                '2 / 4' : 'Low / high', '1 / 4': 'Low / high',
                '1 / 3': 'Low / high'}
    gradeNormDf['Pair (grouped)'] = gradeNormDf['Pair'].map(pairsDict)

    # get combined archs names:
    archCombinedDict = {'Small nest':'1',
                        'Macrocystic':'1',
                        'Large nest':'1',
                        'Alveolar':'2',
                        'Trabecular':'2',
                        'Papillary':'2',
                        'Solid':'3',
                        'Rhabdoid':'3',
                        'Large nest-Alveolar':'2',
                        'Large nest-Trabecular':'2',
                        'Alveolar-papillary':'2'}

    archsNormDf["Pair (grouped)"] = np.nan
    archsCircleDf["Pair (grouped)"] = np.nan
    archsHalfDf["Pair (grouped)"] = np.nan
    for index, row in archsNormDf.iterrows():
        oldPair = row['Pair']
        oldPairA, oldPairB = oldPair.split(' / ')
        if (oldPairA in archCombinedDict) & (oldPairB in archCombinedDict):            
            newPair = [archCombinedDict[oldPairA], archCombinedDict[oldPairB]]
            if newPair[0] != newPair[1]:
                newPair.sort()
                newPair = " / ".join(newPair)
                archsNormDf.loc[index, "Pair (grouped)"] = newPair
                archsCircleDf.loc[index, "Pair (grouped)"] = newPair
                archsHalfDf.loc[index, "Pair (grouped)"] = newPair

    archsNormDf = archsNormDf[~archsNormDf.Pair.str.contains('-')]
    archsCircleDf = archsCircleDf[~archsCircleDf.Pair.str.contains('-')]
    archsHalfDf = archsHalfDf[~archsHalfDf.Pair.str.contains('-')]
    archsNormDf = archsNormDf[~archsNormDf.Pair.str.contains('BleedingFollicle')]
    archsCircleDf = archsCircleDf[~archsCircleDf.Pair.str.contains('BleedingFollicle')]
    archsHalfDf = archsHalfDf[~archsHalfDf.Pair.str.contains('BleedingFollicle')]

    # percentiles:
    perct = 95
    allGradesCircle95thPerc = np.percentile(np.concatenate(list(gradeCircleDf['All values'])),perct)
    allGradesSplit95thPerc = np.percentile(np.concatenate(list(gradeHalfDf['All values'])),perct)

    allArchsCircle95thPerc = np.percentile(np.concatenate(list(archsCircleDf['All values'])),perct)
    allArchsSplit95thPerc = np.percentile(np.concatenate(list(archsHalfDf['All values'])),perct)

    # change pairs symbol:
    gradeNormDf['Pair'] = gradeNormDf['Pair'].apply(lambda x: x.replace('/','vs'))
    gradeNormDf['Pair (grouped)'] = gradeNormDf['Pair (grouped)'].apply(lambda x: x.replace('/','vs') if isinstance(x, str) else x)
    gradeCircleDf['Pair'] = gradeCircleDf['Pair'].apply(lambda x: x.replace('/','vs'))
    gradeHalfDf['Pair'] = gradeHalfDf['Pair'].apply(lambda x: x.replace('/','vs'))

    archsNormDf['Pair'] = archsNormDf['Pair'].apply(lambda x: x.replace('/','vs'))
    archsNormDf['Pair (grouped)'] = archsNormDf['Pair (grouped)'].apply(lambda x: x.replace('/','vs') if isinstance(x, str) else x)
    archsCircleDf['Pair'] = archsCircleDf['Pair'].apply(lambda x: x.replace('/','vs'))
    archsHalfDf['Pair'] = archsHalfDf['Pair'].apply(lambda x: x.replace('/','vs'))

    # 1) ALL GRADES
    fig = plt.figure()
    sns.scatterplot(gradeNormDf.sort_values(by='Pair'), x='Pair', y='Separation', hue='Pair', s=150,legend=None)
    plt.axhline(y=allGradesCircle95thPerc, color='dimgrey', linestyle='--')
    plt.axhline(y=allGradesSplit95thPerc, color='silver', linestyle='--')

    patches1 = Line2D([], [], linestyle="--", ms=8, color='dimgrey', label='95th pct. of contiguity baseline')
    patches2 = Line2D([], [], linestyle="--", ms=8, color='silver', label='95th pct. of negative control')
    plt.legend(handles=[patches1,patches2],loc='lower left',framealpha=1.0, facecolor='white',fontsize=12.5)

    x_offset = plt.xlim()[1] - 4

    plt.text(x=x_offset, y=allGradesCircle95thPerc + 0.01, s='Full', color='black', verticalalignment='bottom', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=(allGradesCircle95thPerc + allGradesSplit95thPerc) / 2, s='Partial', color='dimgrey',  verticalalignment='center', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=allGradesSplit95thPerc - 0.01, s='None', color='silver', verticalalignment='top', horizontalalignment='left',fontsize=15)

    plt.title('Grades\n')
    plt.xlabel('Grade pairs')
    plt.ylabel('Separation')
    plt.ylim(0.41,1.05)
    fig.patch.set_alpha(0.0)
    if saveFig:
        gradeNormDf[['Pair','Separation']].to_csv(os.path.join(figuresDir, 'separation_all_grades.csv'), index=[0])
        fig.savefig(os.path.join(figuresDir, 'separation_all_grades'+suffix+'.png'))
        fig.savefig(os.path.join(figuresDir, 'separation_all_grades'+suffix+'.svg'),format='svg',dpi=600)   
    plt.show()


    # 2) ALL ARCHITECTURES 
    fig = plt.figure()
    sca = sns.scatterplot(archsNormDf.sort_values(by='Pair'), x='Pair', y='Separation', hue='Pair', s=150,style='Grade change',markers=['X','o'])

    plt.axhline(y=allArchsCircle95thPerc, color='dimgrey', linestyle='--')
    plt.axhline(y=allArchsSplit95thPerc, color='silver', linestyle='--')
    handles, labels = sca.get_legend_handles_labels()

    patches1 = Line2D([], [], linestyle="--", ms=8, color='dimgrey', label='95th pct. of contiguity baseline')
    patches2 = Line2D([], [], linestyle="--", ms=8, color='silver', label='95th pct. of negative control')

    x_offset = plt.xlim()[1] - 14

    plt.text(x=x_offset, y=allArchsCircle95thPerc + 0.01, s='Full', color='black', verticalalignment='bottom', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=(allArchsCircle95thPerc + allArchsSplit95thPerc) / 2, s='Partial', color='dimgrey', verticalalignment='center', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=allArchsSplit95thPerc - 0.01, s='None', color='silver', verticalalignment='top', horizontalalignment='left',fontsize=15)

    sca.legend(handles=handles[-2:]+[patches1,patches2], labels=labels[-2:]+['95th pct. of contiguity baseline','95th pct. of negative control'],fontsize=12.5)
    plt.title('Architectures\n')
    plt.xlabel('Architecture pairs')
    plt.ylabel('Separation')
    plt.xticks(rotation=90)
    plt.ylim(0.41,1.05)
    fig.patch.set_alpha(0.0)
    if saveFig:
        archsNormDf[['Pair','Separation']].to_csv(os.path.join(figuresDir, 'separation_all_archs.csv'))
        fig.savefig(os.path.join(figuresDir, 'separation_all_archs'+suffix+'.png'))
        fig.savefig(os.path.join(figuresDir, 'separation_all_archs'+suffix+'.svg'),format='svg',dpi=600)    
    plt.show()


    # 1) COMBINED GRADES
    fig = plt.figure(figsize=(5,5))
    ax = sns.swarmplot(gradeNormDf.sort_values(by='Pair (grouped)').dropna(), x='Pair (grouped)', y='Separation',hue='Pair (grouped)',s=15,palette=['Palevioletred'], legend=None)
    plt.axhline(y=allGradesCircle95thPerc, color='dimgrey', linestyle='--')
    plt.axhline(y=allGradesSplit95thPerc, color='silver', linestyle='--')

    patches1 = Line2D([], [], linestyle="--", ms=15, color='dimgrey', label='95th pct. of contiguity baseline')
    patches2 = Line2D([], [], linestyle="--", ms=15, color='silver', label='95th pct. of negative control')

    x_offset = plt.xlim()[1] - .99

    plt.text(x=x_offset, y=allGradesCircle95thPerc + 0.01, s='Full', color='black', verticalalignment='bottom', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=(allGradesCircle95thPerc + allGradesSplit95thPerc) / 2, s='Partial', color='dimgrey', verticalalignment='center', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=allGradesSplit95thPerc - 0.01, s='None', color='silver', verticalalignment='top', horizontalalignment='left',fontsize=15)

    plt.legend(handles=[patches1,patches2],loc='lower left',framealpha=1.0, facecolor='white',fontsize=12.5)
    plt.title('Grades\n')
    plt.ylim(0.49,1.05)
    plt.xlabel('Grade pair')
    ax.set_xlim((-0.5,0.5))
    fig.patch.set_alpha(0.0)
    if saveFig:
        gradeNormDf[['Pair (grouped)','Separation']].sort_values(by='Pair (grouped)').dropna().to_csv(os.path.join(figuresDir, 'separation_combined_grades.csv'))
        fig.savefig(os.path.join(figuresDir, 'separation_combined_grades'+suffix+'.png'))
        fig.savefig(os.path.join(figuresDir, 'separation_combined_grades'+suffix+'.svg'),format='svg',dpi=600)    
    plt.show()


    # 2) COMBINED ARCHITECTURES 
    fig = plt.figure(figsize=(5,5))
    archsOrder = ['1 vs 2','1 vs 3','2 vs 3']
    ax = sns.swarmplot(archsNormDf.sort_values(by='Pair (grouped)').dropna(), x='Pair (grouped)', y='Separation', hue='Pair (grouped)', s=7, palette=['yellowgreen','darkcyan', 'darkslateblue'], legend=None,order=archsOrder)
    plt.axhline(y=allArchsCircle95thPerc, color='dimgrey', linestyle='--')
    plt.axhline(y=allArchsSplit95thPerc, color='silver', linestyle='--')

    plt.title('Architecture classes\n')
    plt.ylim(0.49,1.05)
    plt.xlabel('Architecture group pair')
    plt.ylabel('Separation')

    patches1 = Line2D([], [], linestyle="--", ms=8, color='dimgrey', label='95th pct. of contiguity baseline')
    patches2 = Line2D([], [], linestyle="--", ms=8, color='silver', label='95th pct. of negative control')
    plt.legend(handles=[patches1,patches2],loc='lower left',framealpha=1.0, facecolor='white',fontsize=12.5)

    x_offset = plt.xlim()[1] - 3.47

    plt.text(x=x_offset, y=allArchsCircle95thPerc + 0.01, s='Full', color='black', verticalalignment='bottom', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=(allArchsCircle95thPerc + allArchsSplit95thPerc) / 2, s='Partial', color='dimgrey', verticalalignment='center', horizontalalignment='left',fontsize=15)
    plt.text(x=x_offset, y=allArchsSplit95thPerc - 0.01, s='None', color='silver', verticalalignment='top', horizontalalignment='left',fontsize=15)

    ax.set_xlim((-1,3))
    fig.patch.set_alpha(0.0)
    if saveFig:
        archsNormDf[['Pair (grouped)','Separation']].dropna().to_csv(os.path.join(figuresDir, 'separation_combined_archs.csv'))
        fig.savefig(os.path.join(figuresDir, 'separation_combined_archs'+suffix+'.png'))
        fig.savefig(os.path.join(figuresDir, 'separation_combined_archs'+suffix+'.svg'),format='svg',dpi=600)     
    plt.show()

    if not uni:
        # architecture heatmap:
        archsHeatmapDf = archsNormDf.copy()
        archsHeatmapDf = archsHeatmapDf[~archsHeatmapDf.Pair.str.contains('-')]
        archNormDfMean = archsHeatmapDf[['Pair','Separation','Negative control (circle, average)','Negative control (split, average)']].groupby('Pair').mean()
        archNormDfMean[['PairA', 'PairB']] = archNormDfMean.index.to_series().str.split(' vs ', 1, expand=True)

        archNormHeatmap = pd.crosstab(index = archNormDfMean['PairA'], columns = archNormDfMean['PairB'], values=archNormDfMean['Separation'], aggfunc='sum')
        archNormHeatmap = archNormHeatmap.combine_first(archNormHeatmap.T).fillna(0.0)
        archNormHeatmap = archNormHeatmap.reindex(index=['Small nest','Large nest','Alveolar','Trabecular','Solid'],columns=['Small nest','Large nest','Alveolar','Trabecular','Solid'])

        archNormHeatmap.index.name = ''
        archNormHeatmap.columns.name = ''
        archNormHeatmap = archNormHeatmap.where(np.triu(np.ones(archNormHeatmap.shape)).astype(bool))
        archNormHeatmap.replace(np.nan, 0, inplace=True)

        archRandHeatmap = pd.crosstab(index = archNormDfMean['PairA'], columns = archNormDfMean['PairB'], values=archNormDfMean['Negative control (circle, average)'], aggfunc='sum')
        archRandHeatmap = archRandHeatmap.combine_first(archRandHeatmap.T).fillna(0.0)
        archRandHeatmap = archRandHeatmap.reindex(index=['Small nest','Large nest','Alveolar','Trabecular','Solid'],columns=['Small nest','Large nest','Alveolar','Trabecular','Solid'])

        archRandHeatmap.index.name = ''
        archRandHeatmap.columns.name = ''
        archRandHeatmap = archRandHeatmap.where(np.triu(np.ones(archRandHeatmap.shape)).astype(bool))
        archRandHeatmap.replace(np.nan, 0, inplace=True)

        archsCombined = archRandHeatmap.T + archNormHeatmap
        archsCombined.replace(0, np.nan, inplace=True)

        fig = plt.figure(figsize=(6,5))
        plt.fill_between([5,0],[5,0],[5,5],color="whitesmoke")
        a = sns.heatmap(archsCombined, annot=True, cmap='RdYlGn', vmin=0.60,vmax=1.0)
        plt.title('Architectures separation\n')
        angle = np.rad2deg(np.arctan2(45,45))

        plt.text(0.15,0.7, '*', ha='left', va='bottom', transform_rotates_text=True, rotation=angle, rotation_mode='anchor', fontsize=20)
        for _, spine in a.spines.items(): 
            spine.set_visible(True) 
            spine.set_linewidth(1) 
            spine.set_linestyle("-") 

        fig.patch.set_alpha(0.0)
        if saveFig:
            archsCombined.to_csv(os.path.join(figuresDir, 'separation_archs_heatmap.csv'))
            fig.savefig(os.path.join(figuresDir, 'separation_archs_heatmap.png'))
            fig.savefig(os.path.join(figuresDir, 'separation_archs_heatmap.svg'),format='svg',dpi=600)    
        plt.show()

        # grade heatmap:
        gradeNormDfMean = gradeNormDf[['Pair','Separation','Negative control (circle, average)','Negative control (split, average)']].groupby('Pair').mean()
        gradeNormDfMean[['PairA', 'PairB']] = gradeNormDfMean.index.to_series().str.split(' vs ', 1, expand=True)

        gradeNormHeatmap = pd.crosstab(index = gradeNormDfMean['PairA'], columns = gradeNormDfMean['PairB'], values=gradeNormDfMean['Separation'], aggfunc='sum')
        gradeNormHeatmap = gradeNormHeatmap.combine_first(gradeNormHeatmap.T).fillna(0.0)

        gradeNormHeatmap.index.name = ''
        gradeNormHeatmap.columns.name = ''
        gradeNormHeatmap = gradeNormHeatmap.sort_index(axis=0)
        gradeNormHeatmap = gradeNormHeatmap.sort_index(axis=1)
        gradeNormHeatmap = gradeNormHeatmap.where(np.triu(np.ones(gradeNormHeatmap.shape)).astype(bool))
        gradeNormHeatmap.replace(np.nan, 0, inplace=True)

        gradeRandHeatmap = pd.crosstab(index = gradeNormDfMean['PairA'], columns = gradeNormDfMean['PairB'], values=gradeNormDfMean['Negative control (split, average)'], aggfunc='sum')
        gradeRandHeatmap = gradeRandHeatmap.combine_first(gradeRandHeatmap.T).fillna(0.0)
        gradeRandHeatmap.index.name = ''
        gradeRandHeatmap.columns.name = ''
        gradeRandHeatmap = gradeRandHeatmap.sort_index(axis=0)
        gradeRandHeatmap = gradeRandHeatmap.sort_index(axis=1)
        gradeRandHeatmap = gradeRandHeatmap.where(np.triu(np.ones(gradeRandHeatmap.shape)).astype(bool))
        gradeRandHeatmap.replace(np.nan, 0, inplace=True)

        gradeCombined = gradeRandHeatmap.T + gradeNormHeatmap
        gradeCombined.replace(0, np.nan, inplace=True)

        fig = plt.figure(figsize=(6,5))
        plt.fill_between([5,0],[5,0],[5,5],color="whitesmoke")
        a = sns.heatmap(gradeCombined, annot=True, cmap='RdYlGn', vmin=0.60,vmax=1.0)
        plt.yticks(rotation=360)
        plt.title('Grades separation\n')
        angle = np.rad2deg(np.arctan2(45,45))

        plt.text(0.15,0.7, '*', ha='left', va='bottom', transform_rotates_text=True, rotation=angle, rotation_mode='anchor', fontsize=20)
        for _, spine in a.spines.items(): 
            spine.set_visible(True) 
            spine.set_linewidth(1) 
            spine.set_linestyle("-") 

        fig.patch.set_alpha(0.0)
        if saveFig:
            gradeCombined.to_csv(os.path.join(figuresDir, 'separation_grades_heatmap.csv'))
            fig.savefig(os.path.join(figuresDir, 'separation_grades_heatmap.png'))
            fig.savefig(os.path.join(figuresDir, 'separation_grades_heatmap.svg'),format='svg',dpi=600)    
        plt.show()

# %%

if __name__ == "__main__":
    
    __main__()
    
# %%
