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

import yaml
import os
import cv2
import skimage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openslide as oSlide
import seaborn as sns

from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap
from sklearn.metrics.pairwise import cosine_distances

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

from Utils.Image_Utils import MaskFromXML
from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Visualization import apply_plot_settings
from Utils.Image_Utils import MaskFromXML

# %% Visualization settings, useful dictionaries.

figuresDir = files['FIGURES']
apply_plot_settings()


# %%

def __main__(saveFig=False):

    # comparea distances:
    profiles = np.load(os.path.join(figuresDir, 'tmaTesting_profilesMorphoITH.npy'))
    profiles_UNI = np.load(os.path.join(figuresDir, 'tmaTesting_profilesUNI.npy'))

    profiles_UNI = np.load(os.path.join(os.path.join(figuresDir,'profilesMayo_UNI.npy')))
    profiles = np.load(os.path.join(os.path.join(figuresDir,'profilesMayo.npy')))
        
    assert profiles.shape[0] == profiles_UNI.shape[0]
    numberOfProfiles = profiles.shape[0]

    np.random.seed(10)
    chosenIdxA = np.random.choice(numberOfProfiles, 300)
    np.random.seed(20)
    chosenIdxB = np.random.choice(numberOfProfiles, 300)

    distances = cosine_distances(profiles[chosenIdxA], profiles[chosenIdxB])
    distances_UNI = cosine_distances(profiles_UNI[chosenIdxA], profiles_UNI[chosenIdxB])
    row_idx, col_idx = np.triu_indices_from(distances, k=1)

    fig = plt.figure(figsize=(5,5))
    x = distances[row_idx, col_idx]
    y = distances_UNI[row_idx, col_idx]

    r, _ = pearsonr(x,y)
    plt.scatter(x=x,y=y,s=1)
    ax = fig.gca()
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va='top')

    plt.xlabel('MorphoITH encoder')
    plt.ylabel('UNI')
    plt.title('Cosine distances between\nsubset of feature vectors')
    plt.tight_layout()
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir, 'encoders_dinstances.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'encoders_dinstances.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()


    # compare separations:
    mutationsDf_UNI = pd.read_csv(os.path.join(figuresDir, 'normMS_UNI.csv'))
    circMutationsDf_UNI = pd.read_csv(os.path.join(figuresDir, 'circMS_UNI.csv'))
    splitMutationsDf_UNI = pd.read_csv(os.path.join(figuresDir, 'halfMS_UNI.csv'))

    mutationsDf = pd.read_csv(os.path.join(figuresDir, 'normMS.csv'))
    circMutationsDf = pd.read_csv(os.path.join(figuresDir, 'circMS.csv'))
    splitMutationsDf = pd.read_csv(os.path.join(figuresDir, 'halfMS.csv'))


    pairs = [(mutationsDf['Accuracy'].values, mutationsDf_UNI['Accuracy'].values),
            (circMutationsDf['Accuracy'].values, circMutationsDf_UNI['Accuracy'].values),
            (splitMutationsDf['Accuracy'].values, splitMutationsDf_UNI['Accuracy'].values),]
    labels = ['Separation', 'Contiguity baseline separation', 'Negative control separation']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)

    for cnt, (ax, (x, y)) in enumerate(zip(axes, pairs)):
        
        r, _ = pearsonr(x, y)

        sns.kdeplot(ax=ax, x=x, y=y, fill=False, thresh=0.05, levels=10)
            
        lo = round(min(x.min(),   y.min()),   1) - 0.05
        hi = round(max(x.max(),   y.max()),   1) + 0.05
        
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.plot([lo, hi], [lo, hi], ls='--', c='red')
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va='top')
        ax.set_title(labels[cnt])
        ax.set_xlabel('MorphoITH encoder')
        ax.set_ylabel('UNI')

    plt.tight_layout()
    if saveFig:
        fig.savefig(os.path.join(figuresDir, 'encoders_separation.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'encoders_separation.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()

    # compare clusters (2):
    dataset='bap1'
    cmap = ListedColormap(['white', 'lightgrey', 'black'])
    clustersNamesDict = {18625:(['white', 'black', 'lightgrey'],['white', 'black', 'lightgrey']),
                         19097:(['white', 'lightgrey', 'black'],['white', 'black', 'lightgrey']),
                         20273:(['white', 'black', 'lightgrey'],['white', 'black', 'lightgrey']),
                         19852:(['white', 'black', 'lightgrey'],['white', 'black', 'lightgrey'])}

    for name, (cmap1, cmap2) in clustersNamesDict.items():
        name = str(name)

        svsFile = os.path.join(files['SLIDES']['mayo'],name+'.svs')
        slide = oSlide.open_slide(svsFile)
        level = min(2,slide.level_count-1)
        imgLowRes = np.array(slide.read_region((0,0), level, slide.level_dimensions[level]))[:,:,range(3)]
    
        tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],dataset,name+'.npy')
        tumor = np.load(tissueMaskFile)
        tumor = np.uint8(SmoothResponseAndClasses(tumor,smoothSize=0)[0])
        tumor[tumor != 5] = 0
        tumor[tumor == 5] = 1
        tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
        tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))

        if dataset == 'bap1':
            annoFile = os.path.join(files['MASKS']['annotations'], dataset,name+'.xml')
            mask,_=MaskFromXML(annoFile,'Focal_BAP1',slideDim=(slide.dimensions),downSampleFactor=16)
        else:
            annoFile = os.path.join(files['MASKS']['annotations'], dataset,name+'.txt')
            mask,_=MaskFromXML(annoFile,[], slideDim=slide.dimensions,downSampleFactor=16)

        mask[mask != 0] = 1
        mask = cv2.resize(mask, (tumor.shape[1], tumor.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask += 1
        mask = mask*tumor
        
        clusterFile = os.path.join(files['MASKS']['clusters_mayo'],name+'_cluster2.npy')
        cluster = np.load(clusterFile)
        cluster= cluster * tumor

        clusterFile_UNI = os.path.join(files['MASKS']['clusters_mayo_uni'],name+'_cluster2.npy')
        cluster_UNI = np.load(clusterFile_UNI)
        cluster_UNI = cluster_UNI * tumor

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

        axes[0].imshow(imgLowRes)
        axes[0].set_title('ID: '+str(name)+' (WSI-3)')
        axes[0].set_xticks([],[])
        axes[0].set_yticks([],[])

        axes[1].imshow(mask,cmap=cmap,interpolation='none')
        axes[1].set_title('Ground truth')
        axes[1].set_xticks([],[])
        axes[1].set_yticks([],[])

        axes[2].imshow(cluster,cmap=ListedColormap(cmap1),interpolation='none')
        axes[2].set_title('MorphoITH encoder')
        axes[2].set_xticks([],[])
        axes[2].set_yticks([],[])

        axes[3].imshow(cluster_UNI,cmap=ListedColormap(cmap2),interpolation='none')
        axes[3].set_title('UNI')
        axes[3].set_xticks([],[])
        axes[3].set_yticks([],[])
        
        plt.tight_layout()
        fig.patch.set_alpha(0.0)
        if saveFig:
            fig.savefig(os.path.join(figuresDir, 'encoders_clusters_bap1_'+str(name)+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figuresDir, 'encoders_clusters_bap1_'+str(name)+'.svg'),format='svg',dpi=600,bbox_inches='tight')
        plt.show()

# %%

if __name__ == "__main__":
    
    __main__()