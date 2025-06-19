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
import sklearn
import tqdm
import skimage
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openslide as oSlide
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Controls import MakeHalfMask, MakeCircleMask
from Utils.Separation import GetAccuracies
from Utils.Visualization import apply_plot_settings, rgb2hex

# %% Visualization settings, useful dictionaries.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
tcgaMeta = pd.read_csv(files['METADATA']['tcga']['grade'])
tcgaMeta['SVS name']=tcgaMeta['Friendly Path'].apply(lambda x: x.split('/')[-1].split('.')[0])

figuresDir = files['FIGURES']
apply_plot_settings()

tissueColorsMap = {0: 'white', 1: 'red', 2: 'yellow',
                   3: 'green', 4: 'orange', 5: 'blue', 6: 'black', 7: 'cyan'}
classIxToNameDict = {0: 'Background', 1: 'Blood', 2: 'Fat',
                     3: 'Normal', 4: 'Stroma', 5: 'Tumor',
                     6: 'Necrosis', 7: 'Inflammatory'}

# %%

def get_mask_and_profiles(name, dataset, uni=False):
    """ Given a slide name, return its MorphoITH profiles, and tissue classifier output, along with low-res H&E image.

    Args:
        name: slide name.
        dataset: "tcga".

    Returns:
        mask (array): tumor and non-tumor mask.
        profiles (array): feature vectors after tessellation.
        imgLowRes: low resolution of H&E image.
        tissueMask: tissue mask.
    """

    name = str(name)
    svsFile = os.path.join(files['SLIDES'][dataset], name+'.svs')
    profilesFile = os.path.join(files['MASKS']['morphoith'], dataset, name+'.npy')
    if uni:
        profilesFile = os.path.join(files['MASKS']['uni'], dataset, name+'.npy')
    profiles = np.load(profilesFile)
    tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'], dataset, name+'.npy')
    tumorOrg = np.load(tissueMaskFile)
    tissueMask = np.uint8(SmoothResponseAndClasses(tumorOrg,smoothSize=0)[0])
    tissueMaskSmooth = np.uint8(SmoothResponseAndClasses(tumorOrg, 3)[0])
    tissueBackground = np.uint8(skimage.morphology.remove_small_objects(tissueMask != 0, 100))
    tissueBackground = cv2.erode(np.uint8(tissueBackground), np.ones((1, 1), np.uint8))
    tissueMask = tissueMaskSmooth*tissueBackground

    # marker detection:
    maskFile = os.path.join(files['MASKS']['markers']['tcga'],name+'.svs_mask.png')
    markers = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    markers = cv2.resize(markers,(profiles.shape[1],profiles.shape[0]))
    markers[markers==3]=4
    markers[markers!=4]=0
    markers[markers==4]=1
    markers = markers == 0
    tissueMask = tissueMask * markers

    slide = oSlide.open_slide(svsFile)
    level = min(2, slide.level_count-1)
    imgLowRes = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, range(3)]

    # get tumor area:
    tumor = tissueMask.copy()
    tumor[tumor != 5] = 0
    tumor[tumor == 5] = 2

    # get nontumor area:
    nonTumor = tissueMask.copy()
    nonTumor[nonTumor == 5] = 0
    nonTumor[nonTumor != 0] = 1

    mask = tumor + nonTumor

    return mask, profiles, imgLowRes, tissueMask

def get_rgb_visualization(nameList, dataset, numberOfRuns, saveFig):
    """Given a list of slides, visualize them for figures.

    Args:
        nameList (list): slides to visualize.
    """

    slideToVisualizeSep = slidesToVisualize[(slidesToVisualize['Reason'] == 'Separation')]['Name'].tolist()
    slideToVisualizeCommon = slidesToVisualize[( slidesToVisualize['Reason'] == 'Tissues')]['Name'].tolist()
    slideToVisualizeNegativeControl = slidesToVisualize[(slidesToVisualize['Reason'] == 'Negative control')]['Name'].tolist()

    sampledProfiles = []
    for name in nameList:

        mask, profiles, imgLowRes, tissueMask = get_mask_and_profiles(name, dataset)
        idx1, idx2 = np.nonzero(mask)
        profilesFlat = profiles[idx1, idx2, :]
        tissueBackground = np.uint8(
            skimage.morphology.remove_small_objects(tissueMask != 0, 100))
        tissueBackground = cv2.erode(
            np.uint8(tissueBackground), np.ones((1, 1), np.uint8))
        tissueMask = tissueMask*tissueBackground

        idx1, idx2 = np.where(mask != 0)

        if name in slideToVisualizeCommon:
            m = np.percentile(profilesFlat, [1], axis=0)
            M = np.percentile(profilesFlat, [99], axis=0)
            profilesClip = np.clip(profilesFlat, m, M)
            sampledProfiles.append(profilesClip)

    if len(sampledProfiles) > 0:
        pcaModule = sklearn.decomposition.PCA(n_components=3)
        minMaxModule = sklearn.preprocessing.MinMaxScaler()

        sampledProfiles = np.vstack(sampledProfiles)
        pcaModule.fit(sampledProfiles)
        pcaValuesAll = pcaModule.transform(sampledProfiles)
        minMaxModule.fit(pcaValuesAll)

    for name in nameList:
        print(name)
        mask, profiles, imgLowRes, tissueMask = get_mask_and_profiles(name, dataset)
        idx1, idx2 = np.nonzero(mask)
        profilesFlat = profiles[idx1, idx2, :]

        tcgaName = tcgaMeta[tcgaMeta['SVS name']==name]['id'].item()
        print(tcgaName)

        if name in slideToVisualizeSep:
            
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(mask,cmap=ListedColormap(['white','lightgrey','black']))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('Tumor vs non-tumor')
            plt.legend(handles=[mpatches.Patch(color='black', label='Tumor'),
                                mpatches.Patch(color='lightgrey', label='Non-tumor')], bbox_to_anchor=(1.05, 1.0), loc='upper left',framealpha=1.0, facecolor='white')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'tumor_nontumor_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'tumor_nontumor_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            fig = plt.figure(figsize=(5, 5))
            plt.imshow(imgLowRes)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('H&E image')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'hne_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            uniqueTissues = np.unique(tissueMask)
            tissueColorsList = ListedColormap(list(tissueColorsMap.values()))

            fig = plt.figure(figsize=(5, 5))
            plt.imshow(tissueMask, cmap=tissueColorsList,
                        interpolation='none', vmin=0, vmax=len(classIxToNameDict)-1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            patches = [mpatches.Patch(
                color=tissueColorsMap[i], label=classIxToNameDict[i]) for i in uniqueTissues[1:]]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.0),
                        loc='upper left', framealpha=1.0, facecolor='white')
            plt.title('Region classifier')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'tissues_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'tissues_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            profilesPca = PCA(n_components=10).fit_transform(profilesFlat)
            m = np.percentile(profilesPca, [1], axis=0)
            M = np.percentile(profilesPca, [99], axis=0)
            profilesClip = np.clip((profilesPca-m)/(M-m), 0, 1)

            pcaRGB = np.ones((mask.shape[0], mask.shape[1], 3))
            pcaRGB[idx1, idx2, :] = profilesClip[:, range(3)]

            fig = plt.figure(figsize=(5, 5))
            plt.imshow(pcaRGB)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('MorphoITH')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

        if name in slideToVisualizeCommon:

            uniqueTissues = np.unique(tissueMask)
            tissueColorsList = ListedColormap(list(tissueColorsMap.values()))

            tissuePalette = dict(
                zip(list(classIxToNameDict.values()), list(tissueColorsMap.values())))
            tumorOnlyTissues = tissueMask[idx1, idx2]

            np.random.seed(123)
            randIdx = np.random.choice(profilesFlat.shape[0], 1000)

            profilesRdx = profilesFlat[randIdx]
            tissueLabels = tumorOnlyTissues[randIdx]
            tissueLabels = [classIxToNameDict[i] for i in tissueLabels]

            tsnePos = TSNE(init='random', n_components=2, metric='cosine', perplexity=30,random_state=123, learning_rate=200, square_distances=True).fit_transform(profilesRdx)
            tsneDf = pd.DataFrame({'t-SNE 1': tsnePos[:, 0], 't-SNE 2': tsnePos[:, 1], 'Tissue': tissueLabels})

            fig = plt.figure(figsize=(5, 5))
            sns.scatterplot(data=tsneDf[tsneDf.Tissue != 'Background'], x='t-SNE 1', y='t-SNE 2', hue='Tissue', palette=tissuePalette, s=100)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
                       title='Tissue', framealpha=1.0, facecolor='white')
            plt.xticks([], [])
            plt.yticks([], [])
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'tissues_scatterplot_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'tissues_scatterplot_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            fig = plt.figure(figsize=(5, 5))
            plt.imshow(tissueMask, cmap=tissueColorsList, interpolation='none', vmin=0, vmax=len(classIxToNameDict)-1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            patches = [mpatches.Patch(
                color=tissueColorsMap[i], label=classIxToNameDict[i]) for i in uniqueTissues[1:]]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.0),
                       loc='upper left', framealpha=1.0, facecolor='white')
            fig.patch.set_alpha(0.0)
            plt.title('Region classifier')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'tissues_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'tissues_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            m = np.percentile(profilesFlat, [1], axis=0)
            M = np.percentile(profilesFlat, [99], axis=0)
            profilesFlatClip = np.clip(profilesFlat, m, M)

            profilesPca = pcaModule.transform(profilesFlatClip)
            profilesPcaScaled = minMaxModule.transform(profilesPca)

            pcaRGB = np.ones((mask.shape[0], mask.shape[1], 3))
            pcaRGBFlat = profilesPcaScaled[:, range(3)]
            pcaRGB[idx1, idx2, :] = pcaRGBFlat

            tsneDf['RGB'] = [rgb2hex(i) for i in pcaRGBFlat[randIdx]*255]
            fig = plt.figure(figsize=(5, 5))
            sns.scatterplot(data=tsneDf, x='t-SNE 1', y='t-SNE 2', color=tsneDf.RGB, s=100, legend=None)
            plt.xticks([], [])
            plt.yticks([], [])
            fig.patch.set_alpha(0.0)
            if saveFig:
                tsneDf.to_csv(os.path.join(figuresDir, 'scatterplot_'+name+'.csv'))
                fig.savefig(os.path.join(figuresDir, 'morphoith_scatterplot_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_scatterplot_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            fig = plt.figure(figsize=(5, 5))
            plt.imshow(pcaRGB)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('MorphoITH')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()
            
        if name in slideToVisualizeNegativeControl:
                
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(mask,cmap=ListedColormap(['white','lightgrey','black']))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('Tumor vs non-tumor\n')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'tumor_nontumor_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'tumor_nontumor_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()
            
            np.random.seed(12345)
            rnIdx = np.random.choice(mask[mask!=0].shape[0],600)
            tsnePos = TSNE(init='random', n_components=2, metric='cosine', perplexity=30, random_state=1, learning_rate=200, square_distances=True).fit_transform(profiles[mask!=0][rnIdx])
            tsneDf = pd.DataFrame({'t-SNE 1':tsnePos[:,0],'t-SNE 2':tsnePos[:,1], 'Color original':mask[mask!=0][rnIdx]})
            
            fig = plt.figure(figsize=(5,5))
            ax=sns.scatterplot(data=tsneDf,x='t-SNE 1',y='t-SNE 2',hue='Color original',s=100,legend=None,palette={1: "lightgrey", 2: "black"})
            ax.set_xlabel(ax.get_xlabel(), fontsize=24) 
            ax.set_ylabel(ax.get_ylabel(), fontsize=24)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'scatterplot_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'scatterplot_'+name+'.svg'), format='svg', dpi=600, bbox_inches='tight')
            plt.show()
            
            ### HALF MASK ###
            halfMasks,_,_ = MakeHalfMask(mask,numberOfRuns)
            halfMask = halfMasks[2]

            fig = plt.figure(figsize=(5,5))
            plt.imshow(halfMask,cmap=ListedColormap(['white','lightgrey','black']),interpolation='none')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('Negative control B\n')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, name+'_negconB.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, name+'_negconB.svg'),format='svg',dpi=600, bbox_inches='tight')
            plt.show()

            tsneDf['Color neg B'] = halfMask[halfMask!=0][rnIdx]
            
            fig = plt.figure(figsize=(5,5))
            ax=sns.scatterplot(data=tsneDf,x='t-SNE 1',y='t-SNE 2',hue='Color neg B',s=100,legend=None,palette={1: "lightgrey", 2: "black"})
            ax.set_xlabel(ax.get_xlabel(), fontsize=24) 
            ax.set_ylabel(ax.get_ylabel(), fontsize=24)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'neg_b_scatterplot_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'neg_b_scatterplot_'+name+'.svg'), format='svg', dpi=600, bbox_inches='tight')
            plt.show()

            ### CIRCLES ###
            circleMasks,_,_ = MakeCircleMask(mask, numberOfRuns)
            circleMask = circleMasks[2]

            fig = plt.figure(figsize=(5,5))
            plt.imshow(circleMask,cmap=ListedColormap(['white','lightgrey','black']),interpolation='none')
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('Negative control A\n')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, name+'_negconA.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, name+'_negconA.svg'),format='svg',dpi=600, bbox_inches='tight')
            plt.show()

            tsneDf['Color neg A'] = circleMask[circleMask!=0][rnIdx]
            
            fig = plt.figure(figsize=(5,5))
            ax=sns.scatterplot(data=tsneDf,x='t-SNE 1',y='t-SNE 2',hue='Color neg A',s=100,legend=None,palette={1: "lightgrey", 2: "black"})
            ax.set_xlabel(ax.get_xlabel(), fontsize=24) 
            ax.set_ylabel(ax.get_ylabel(), fontsize=24)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'neg_a_scatterplot_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'neg_a_scatterplot_'+name+'.svg'), format='svg', dpi=600, bbox_inches='tight')
            plt.show()

            ### RGB ###
            pcaModuleSingle = sklearn.decomposition.PCA(n_components=10)
            profilesPca = pcaModuleSingle.fit_transform(profiles[mask!=0])
            m = np.percentile(profilesPca, [1], axis=0)
            M = np.percentile(profilesPca, [99], axis=0)
            profilesPcaScaled = np.clip((profilesPca-m)/(M-m),0,1)

            pcaRGB = np.ones((mask.shape[0], mask.shape[1], 3))
            pcaRGBFlat = profilesPcaScaled[:, range(3)]
            pcaRGB[idx1, idx2, :] = pcaRGBFlat

            tsneDf['RGB'] = [rgb2hex(i) for i in pcaRGBFlat[rnIdx]*255]
            fig = plt.figure(figsize=(5, 5))
            ax = sns.scatterplot(data=tsneDf, x='t-SNE 1', y='t-SNE 2', color=tsneDf.RGB, s=100, legend=None)
            ax.set_xlabel(ax.get_xlabel(), fontsize=24) 
            ax.set_ylabel(ax.get_ylabel(), fontsize=24)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            if saveFig:
                tsneDf.to_csv(os.path.join(figuresDir, name+'_morphoith_scatterplot.csv'))
                fig.savefig(os.path.join(figuresDir, 'morphoith_scatterplot_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_scatterplot_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

            fig = plt.figure(figsize=(5, 5))
            plt.imshow(pcaRGB)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            fig.patch.set_alpha(0.0)
            plt.title('MorphoITH\n')
            if saveFig:
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(figuresDir, 'morphoith_'+name+'.svg'),format='svg',dpi=600,bbox_inches='tight')
            plt.show()

def run_masks(dataset, numberOfRuns):
    """Given a dataset (here: TCGA), return separation scores for its slides (tumor vs non-tumor areas).

    Args:
        dataset (string): "tcga".
        numberOfRuns (int): number of controls per slide.

    Returns:
        accuracies (dictionary):  separation measures (accuracies) (values) for slides (keys).
        halfAccuracies (dictionary): separation measures (accuracies) (values) for slides (keys) using contiguity control.
        circleAccuracies (dictionary): separation measures (accuracies) (values) for slides (keys) using negative control.
    """

    accuracies = {}
    halfAccuracies = {}
    circleAccuracies = {}

    samplesList = pd.read_csv(files['DATASETS'][dataset])
    samplesList = samplesList['Name'].tolist()

    for cnt, name in enumerate(tqdm.tqdm(samplesList)):

        name = str(name)
        mask, profiles, _, _ = get_mask_and_profiles(name, dataset)
        idx1, idx2 = np.nonzero(mask)
        profilesFlat = profiles[idx1, idx2, :]

        maskFlat = mask[mask != 0]

        halfMasks,_,_ = MakeHalfMask(mask, numberOfRuns)
        circleMasks,_,_ = MakeCircleMask(mask, numberOfRuns)

        p1Idx = np.where(maskFlat == 2)[0]
        p2Idx = np.where(maskFlat == 1)[0]
        howManyPoints = 300
        if (p1Idx.shape[0] > howManyPoints) & (p2Idx.shape[0] > howManyPoints):
        
            np.random.seed(123*cnt)
            p1IdxSubset = np.random.choice(p1Idx, howManyPoints)
            np.random.seed(321*cnt)
            p2IdxSubset = np.random.choice(p2Idx, howManyPoints)

            idxSubset = np.concatenate([p1IdxSubset, p2IdxSubset])
            labelSubset = maskFlat[idxSubset]
            profilesSubset = profilesFlat[idxSubset]

            profilesPcaSubset = PCA(n_components=10).fit_transform(profilesSubset)
            acc = GetAccuracies(profilesPcaSubset, labelSubset)
            accuracies[name] = [acc]

            randomMethods = {'circle': [circleMasks, circleAccuracies], 'split': [halfMasks, halfAccuracies]}
            
            for randomMethod in randomMethods:
                randomMethods[randomMethod][1][name] = []
                for i in range(numberOfRuns):
                    randomMask = randomMethods[randomMethod][0][i]
                    randomMaskFlat = randomMask[randomMask != 0]
                    randomLossIdx = np.where(randomMaskFlat == 2)[0]
                    randomWtIdx = np.where(randomMaskFlat == 1)[0]

                    np.random.seed(12*cnt*i)
                    randomLossIdxSubset = np.random.choice(randomLossIdx, howManyPoints)
                    np.random.seed(21*cnt*i)
                    randomWtIdxSubset = np.random.choice(randomWtIdx, howManyPoints)

                    randomIdxSubset = np.concatenate([randomLossIdxSubset, randomWtIdxSubset])
                    randomLabelSubset = randomMaskFlat[randomIdxSubset]
                    randomProfilesSubset = profilesFlat[randomIdxSubset]

                    randomProfilesPcaSubset = PCA(n_components=10).fit_transform(randomProfilesSubset)
                    r_acc = GetAccuracies(randomProfilesPcaSubset, randomLabelSubset)
                    randomMethods[randomMethod][1][name].append(r_acc)

    return accuracies, halfAccuracies, circleAccuracies

# %% 

def __main__(saveFig=False, runCal=False, numberOfRuns=10, uni=False):

    dataset = 'tcga'
    if not uni:
        slideToVisualize = slidesToVisualize[(slidesToVisualize['Dataset'] == 'tcga')]['Name'].tolist()
        get_rgb_visualization(slideToVisualize, dataset, numberOfRuns, saveFig)
        suffix = ''
    else:
        suffix = '_uni'

    if runCal:
        accNorm, accHalf, accCirc = run_masks(dataset, numberOfRuns)

        normDf = pd.Series(accNorm, name='Accuracy').rename_axis('Sample').explode().reset_index()
        circDf = pd.Series(accCirc, name='Accuracy').rename_axis('Sample').explode().reset_index()
        halfDf = pd.Series(accHalf, name='Accuracy').rename_axis('Sample').explode().reset_index()

        normDf.to_csv(os.path.join(figuresDir, 'normTT'+suffix.upper()+'.csv'))
        circDf.to_csv(os.path.join(figuresDir, 'circTT'+suffix.upper()+'.csv'))
        halfDf.to_csv(os.path.join(figuresDir, 'halfTT'+suffix.upper()+'.csv'))

    else:
        normDf = pd.read_csv(os.path.join(figuresDir, 'normTT'+suffix.upper()+'.csv'),index_col=[0])
        circDf = pd.read_csv(os.path.join(figuresDir, 'circTT'+suffix.upper()+'.csv'),index_col=[0])
        halfDf = pd.read_csv(os.path.join(figuresDir, 'halfTT'+suffix.upper()+'.csv'),index_col=[0])

    normDf['Dataset'] = dataset.upper()
    blackListedTcga = files['METADATA']['tcga']['blacklist']
    blackListedTcgaPd = pd.read_csv(blackListedTcga)
    blackListedTcgaPd['Sample'] = blackListedTcgaPd['FriendlyName'].apply(lambda x: x.split('.')[0])
    blackListedTcgaPd = blackListedTcgaPd[['Sample','ReasonForBlackList']]

    mergedPd = pd.merge(normDf, blackListedTcgaPd,on='Sample', how='left')
    mergedPd['ReasonForBlackList'] = mergedPd['ReasonForBlackList'].fillna('Not blacklisted')
    mergedPd = mergedPd[mergedPd['ReasonForBlackList']=='Not blacklisted']

    circDf = circDf[~circDf['Sample'].isin(list(blackListedTcgaPd['Sample'].values))]
    halfDf = halfDf[~halfDf['Sample'].isin(list(blackListedTcgaPd['Sample'].values))]

    fig = plt.figure(figsize=(5, 5))
    ax = sns.swarmplot(data=mergedPd,  x='Dataset', y='Accuracy',s=4,legend=None,color="purple")

    patches1 = Line2D([], [], linestyle="--", ms=8, color='dimgrey',label='95th pct. of contiguity baseline')
    patches2 = Line2D([], [], linestyle="--", ms=8, color='silver',label='95th pct. of negative control')

    plt.title('Separation of tumor vs non-tumor\n')
    plt.ylim((0.49, 1.05))
    plt.ylabel('Separation')

    percentileCutOff = 95
    cPrct = np.percentile(list(circDf['Accuracy']), percentileCutOff)
    sPrct = np.percentile(list(halfDf['Accuracy']), percentileCutOff)

    plt.legend(handles=[patches1, patches2], loc='lower left', framealpha=1.0, facecolor='white', fontsize=10)
    curretXticks = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())  
    ax.set_xticklabels([label.upper() for label in curretXticks])
    ax.set_xlabel('')

    plt.axhline(y=cPrct, color='dimgrey', linestyle='--', label='y=2')
    plt.axhline(y=sPrct, color='silver', linestyle='--', label='y=2')

    fullPerc = mergedPd[mergedPd['Accuracy'] >= cPrct].shape[0]/mergedPd.shape[0]
    partPerc = mergedPd[(mergedPd['Accuracy'] < cPrct) & (mergedPd['Accuracy'] >= sPrct)].shape[0]/mergedPd.shape[0]
    noPerc = mergedPd[mergedPd['Accuracy'] < sPrct].shape[0]/mergedPd.shape[0]

    x_offset = plt.xlim()[1] - .98

    plt.text(x=x_offset, y=cPrct + 0.01, s='Full ('+str(f"{fullPerc * 100:.1f}%")+')', color='black',
            verticalalignment='bottom', horizontalalignment='left', fontsize=12)
    plt.text(x=x_offset, y=(cPrct + sPrct) / 2, s='Partial ('+str(f"{partPerc * 100:.1f}%")+')', color='dimgrey',
            verticalalignment='center', horizontalalignment='left', fontsize=12)
    plt.text(x=x_offset, y=sPrct - 0.01, s='None ('+str(f"{noPerc * 100:.1f}%")+')', color='silver',
            verticalalignment='top', horizontalalignment='left', fontsize=12)

    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir, 'separation_tumor_nontumor'+suffix+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'separation_tumor_nontumor'+suffix+'.svg'),format='svg', dpi=600, bbox_inches='tight')
    plt.show()

# %%

if __name__ == "__main__":
    
    __main__()

# %%
