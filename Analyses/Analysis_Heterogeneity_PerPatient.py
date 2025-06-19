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
import tqdm
import skimage
import glob as glob
import seaborn as sns
import numpy as np
import pandas as pd
import openslide as oSlide
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.formula.api import ols
from scipy.stats import f_oneway

os.chdir(os.path.dirname(os.path.dirname(__file__)))

from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Visualization import apply_plot_settings
from Utils.Clusters import calculate_feature_representation, centroid_similarity


with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %% Visualization settings, helper functions, useful lists.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

# get input slides that have been annotated by pathologists:
samplesList = pd.read_csv(files['DATASETS']['pilot'])
samplesList = samplesList[samplesList['Sequencing']==True]['Name'].tolist()

# get paths to slides for pilot:
basePilotDir = files['SLIDES']['pilot']
pilotHneDirs = ['Batch1/HnE/','Batch2/HnE/','Batch3/']
fileToPathDict={}
for d in pilotHneDirs:
    pilotFiles=glob.glob(os.path.join(basePilotDir,d,'*.svs'))
    for f in pilotFiles:
        fileToPathDict[os.path.split(f)[-1].split('.')[0]]=f

# %%

def get_arrays(samplesList, uni=False):
    """Given a samples list, extract corresponding MorphoITH profiles, tissue classifier output, pen marks exclusions, and pathologists annotations.

    Args:
        samplesList (list): list of samples for which there are pathological annotations about areas with specific grade/architecture available.

    Returns:
        profilesDict (dictionary): arrays of feature vectors after tessellation (value) for each sample (key).
        masksDict (dictionary): arrays of tumor mask with excluded markers (value) for each sample (key).
        labelsDict (dictionary): arrays of pathologists annotations mask (value) for each sample (key).
        tumorDict (dictionary): arrays of tumor mask  (value) for each sample (key).
    """
    measureDir = {}
    sizesDir = {}
    dataset='pilot'

    for name in samplesList:
        name = str(name)

        svsFile = fileToPathDict[name]
 
        profilesFile = os.path.join(files['MASKS']['morphoith'],dataset,name+'.npy')
        if uni:
            profilesFile = os.path.join(files['MASKS']['morphoith'],dataset,name+'.npy')

        if os.path.exists(profilesFile):

            profiles = np.load(profilesFile)
            slide = oSlide.open_slide(svsFile)

            decX = int(slide.dimensions[1]/profiles.shape[0])
            decY = int(slide.dimensions[0]/profiles.shape[1])
            assert decX == decY

            tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],dataset,name+'.npy')
            tumor = np.load(tissueMaskFile)
            tumor = np.uint8(SmoothResponseAndClasses(tumor,smoothSize=0)[0])
            tumor[tumor != 5] = 0
            tumor[tumor == 5] = 1
            tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
            tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))

            maskFile = os.path.join(files['MASKS']['markers'][dataset], name+'.svs_mask.png')
            markers = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
            markers = cv2.resize(markers,(profiles.shape[1],profiles.shape[0]))
            markers[markers==3]=4
            markers[markers!=4]=0
            markers[markers==4]=1
            tumorNoMarkers = cv2.subtract(tumor, markers)

            n_cluster=10
            min_size=100
            clusterFile = os.path.join(files['MASKS']['clusters_pilot_3patients'],name+'_cluster'+str(n_cluster)+'.npy')

            if os.path.exists(clusterFile):

                idx1, idx2 = np.nonzero(tumorNoMarkers)                    
                clusterMap = np.load(clusterFile)
                clusterMap = clusterMap * tumorNoMarkers
            
                uniqueLabels, uniqueCnt = np.unique(clusterMap,return_counts=True)
                for uniqueLabel in uniqueLabels:
                    if uniqueCnt[uniqueLabels==uniqueLabel] < min_size:
                        clusterMap[clusterMap==uniqueLabel] = 0

                uniqueNew = np.unique(clusterMap)
                value_mapping = {old_value: new_value for new_value, old_value in enumerate(uniqueNew)}
                clusterMap = np.vectorize(value_mapping.get)(clusterMap)

                idx1, idx2 = np.where(clusterMap!=0)
                profilesFlat = profiles[idx1,idx2]
                clustersFlat = clusterMap[idx1,idx2]
                centroids, _, sizes = calculate_feature_representation(profilesFlat, clustersFlat)
                centroid_sim = centroid_similarity(centroids)
                measureDir[name] = centroid_sim
                sizesDir[name] = sizes

    return measureDir, sizesDir

def save_heterogeneity_dataframe(measureDir, sizesDir):
    """Saves as a DataFrame heterogeneity measure after filtering out cluster pairs below 5th percentile of cluster area size and above 95th.

    Args:
       measureDir (dictionary): nested directory with {pair: heterogeneity measure} (value) for sample (key)
       sizesDir (dictionary): nested directory with {pair: cluster size} (value) for sample (key)
    """    

    measurementsFlat = [(name, idx_pair[0], idx_pair[1], value) for name, idx_dict in measureDir.items() for idx_pair, value in idx_dict.items()]
    measurementsDf = pd.DataFrame(measurementsFlat, columns=['Name', 'Corrd1', 'Corrd2', 'Measurement'])

    sizesFlat = [(name, idx, value) for name, idx_dict in sizesDir.items() for idx, value in idx_dict.items()]
    sizesDf = pd.DataFrame(sizesFlat, columns=['Name', 'Cluster', 'Size'])

    merged1 = measurementsDf.merge(sizesDf, left_on=['Name','Corrd1'], right_on=['Name','Cluster'], suffixes=('', '_Corrd1'))
    merged2 = merged1.merge(sizesDf, left_on=['Name','Corrd2'], right_on=['Name','Cluster'], suffixes=('', '_Corrd2'))

    smallestSize = np.percentile(sizesDf['Size'].tolist(),5)
    biggestSize = np.percentile(sizesDf['Size'].tolist(),95)

    filtered_df = merged2[(merged2['Size'] > smallestSize) & (merged2['Size'] < biggestSize)  &
                          (merged2['Size_Corrd2'] > smallestSize) & (merged2['Size_Corrd2'] < biggestSize)]
    filtered_df = filtered_df[['Name','Corrd1', 'Corrd2','Measurement']]

    idx = filtered_df.groupby('Name')['Measurement'].idxmax()
    finalResultsDf = filtered_df.loc[idx]


    def exponential_weight(measurements):
        x_norm = (measurements - measurements.min()) / (measurements.max() - measurements.min())
        weights = np.exp(x_norm)
        weights /= weights.sum()
        return np.sum(measurements * weights)
 
    heterogeneity_df = (
    filtered_df.groupby('Name')['Measurement']
    .apply(exponential_weight)
    .reset_index(name='Heterogeneity'))

    finalResultsDf['Measurement heterogeneity'] = list(heterogeneity_df['Heterogeneity'])

    finalResultsDf = finalResultsDf.reset_index(drop=True)
    return finalResultsDf

# %%


def __main__(saveFig=False):

    measureDict, sizesDict = get_arrays(samplesList)
    heterogeneityDf = save_heterogeneity_dataframe(measureDict, sizesDict)
    heterogeneityDf['Patient'] = heterogeneityDf['Name'].apply(lambda x: x.split('-')[0])

    patientDir = {'KC01138':'Patient A', 'KC01888':'Patient B', 'KC02820':'Patient C'}
    heterogeneityDf['Patient name'] = heterogeneityDf['Patient'].map(patientDir)

    f_oneway(heterogeneityDf[heterogeneityDf['Patient']=='KC01138']['Measurement heterogeneity'],
             heterogeneityDf[heterogeneityDf['Patient']=='KC01888']['Measurement heterogeneity'],
             heterogeneityDf[heterogeneityDf['Patient']=='KC02820']['Measurement heterogeneity'])

    model = ols('Q("Measurement heterogeneity") ~ C(Patient)', data=heterogeneityDf).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    aov['eta_sq'] = aov['sum_sq'] / aov['sum_sq'].sum()
    print(aov)

    fig = plt.figure(figsize=(5,5))
    ax = sns.stripplot(heterogeneityDf,x='Patient name',y='Measurement heterogeneity',s=15)
    plt.xlabel('')
    plt.ylabel('Heterogeneity score')

    ax.text(0.03, 0.98,
            f"ANOVA:\np = {aov.loc['C(Patient)']['PR(>F)']:.3g}\n"
            f"η² = {aov.loc['C(Patient)']['eta_sq']:.1%}",
            transform=ax.transAxes,
            ha='left', va='top', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', ec='none', alpha=0))

    
    fig.patch.set_alpha(0.0)
    if saveFig:
        fig.savefig(os.path.join(figuresDir, 'heterogeneity_pilot_3patients.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'heterogeneity_pilot_3patients.svg'),dpi=600, bbox_inches='tight')
    plt.show()

# %%

if __name__ == "__main__":
    
    __main__()
    