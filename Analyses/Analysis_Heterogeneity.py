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
import tqdm
import cv2
import sys
import skimage
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from statannot import add_stat_annotation

os.chdir(os.path.dirname(os.path.dirname(__file__)))

from Utils.Tumor_Response import SmoothResponseAndClasses
from Utils.Clusters import calculate_feature_representation, centroid_similarity
from Utils.Visualization import apply_plot_settings


with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %% Visualization settings, useful lists.

slidesToVisualize = pd.read_csv(files['VISUALIZATION']['slides'])
figuresDir = files['FIGURES']
apply_plot_settings()

clusterCmap = ["#FFFFFF",  "#6A567A", "#3A7D8C", 
               "#9E3D3F", "#E9B84A", "#9CCB7A",  
               "#4C7C48", "#2D1C39", "#B4526F",
               "#D4B578", "#7C6A67"]

# %%

def heterogeneity_measure(samplesList, n_cluster=10, min_size=100):
    """Calculates heterogeneity measure based on cosine distance between the most distinct morphological clusters.

    Args:
        samplesList (list): list of samples for which heterogeneity measure is to be calculated.
        n_cluster (int, optional): number of morphological clusters that the cluster maps have. Defaults to 10.
        min_size (int, optional): minimal size of a morphological cluster, dictated of smallest tumor regions preserved. Defaults to 100.

    Returns:
       measureDir: nested directory with {pair: heterogeneity measure} (value) for sample (key)
       sizesDir: nested directory with {pair: cluster size} (value) for sample (key)
    """   

    measureDir = {}
    sizesDir = {}

    for name in tqdm.tqdm(samplesList):
            name = str(name)
            
            if name not in measureDir:
                tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],'mayo',name+'.npy')
                profilesFile = os.path.join(files['MASKS']['morphoith'],'mayo',name+'.npy')

                profiles = np.load(profilesFile)
                tumor = np.load(tissueMaskFile)
                tumor = np.uint8(SmoothResponseAndClasses(tumor,smoothSize=0)[0])
                tumor[tumor != 5] = 0
                tumor[tumor == 5] = 1
                tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
                tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))

                clusterSaveDir = files['MASKS']['clusters_mayo']
                clusterFile = os.path.join(clusterSaveDir,name+'_cluster'+str(n_cluster)+'.npy')

                if os.path.exists(clusterFile):

                    idx1, idx2 = np.nonzero(tumor)                    
                    clusterMap = np.load(clusterFile)
                    clusterMap = clusterMap * tumor
                
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

    focalPd = pd.read_csv(files['METADATA']['mayo'])
    focalPd['Name'] = focalPd['svs'].str.replace('.svs', '', regex=False)
    focalPd = focalPd[['Name', 'BAP1_Focal','PBRM1_Focal','SETD2_Focal']]

    measurementsFlat = [(name, idx_pair[0], idx_pair[1], value) for name, idx_dict in measureDir.items() for idx_pair, value in idx_dict.items()]
    measurementsDf = pd.DataFrame(measurementsFlat, columns=['Name', 'Corrd1', 'Corrd2', 'Measurement'])

    sizesFlat = [(name, idx, value) for name, idx_dict in sizesDir.items() for idx, value in idx_dict.items()]
    sizesDf = pd.DataFrame(sizesFlat, columns=['Name', 'Cluster', 'Size'])

    merged1 = measurementsDf.merge(sizesDf, left_on=['Name','Corrd1'], right_on=['Name','Cluster'], suffixes=('', '_Corrd1'))
    merged2 = merged1.merge(sizesDf, left_on=['Name','Corrd2'], right_on=['Name','Cluster'], suffixes=('', '_Corrd2'))

    smallestSize = np.percentile(sizesDf['Size'].tolist(),5)
    biggestSize = np.percentile(sizesDf['Size'].tolist(),95)

    filteredDf = merged2[(merged2['Size'] > smallestSize) & (merged2['Size'] < biggestSize)  &
                          (merged2['Size_Corrd2'] > smallestSize) & (merged2['Size_Corrd2'] < biggestSize)]
    filteredDf = filteredDf[['Name','Corrd1', 'Corrd2','Measurement']]

    finalResultsRawDf = pd.merge(filteredDf,focalPd, on='Name')
    idx = finalResultsRawDf.groupby('Name')['Measurement'].idxmax()
    finalResultsDf = finalResultsRawDf.loc[idx]

    def exponential_weight(measurements):
        x_norm = (measurements - measurements.min()) / (measurements.max() - measurements.min())
        beta = 1
        weights = np.exp(beta * x_norm)
        weights /= weights.sum()
        return np.sum(measurements * weights)
    
    heterogeneity_df = (
    finalResultsRawDf.groupby('Name')['Measurement']
    .apply(exponential_weight)
    .reset_index(name='Heterogeneity'))

    finalResultsDf['Measurement heterogeneity'] = list(heterogeneity_df['Heterogeneity'])
    finalResultsDf = finalResultsDf.reset_index(drop=True)

    finalResultsDf.to_csv(os.path.join(figuresDir, 'heterogeneity_Mayo.csv'))
    return finalResultsDf
    
# %%

def __main__(saveFig=False, runCal=False):

    if runCal:
        samplesList = pd.read_csv(files['DATASETS']['mayo'])
        samplesList = list(samplesList['Name'])
        measureDir, sizesDir = heterogeneity_measure(samplesList)
        finalResultsDf = save_heterogeneity_dataframe(measureDir, sizesDir)
        
    finalResultsDf = pd.read_csv(os.path.join(figuresDir, 'heterogeneity_Mayo.csv'),index_col=[0])

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    for cnt,driverMut in enumerate(['BAP1', 'SETD2', 'PBRM1']):

        sns.violinplot(ax=axs[cnt], data=finalResultsDf, x=driverMut + '_Focal', y='Measurement heterogeneity',palette=['grey','palevioletred'])
        axs[cnt].set_title(driverMut,style='italic')
        axs[cnt].set_ylim((0,0.04))
        finalResultsDf[driverMut + '_Focal'] = finalResultsDf[driverMut + '_Focal'].astype(str)
        add_stat_annotation(axs[cnt], data=finalResultsDf, x=driverMut + '_Focal', y='Measurement heterogeneity',
                            box_pairs=[('False', 'True')],
                            test='Mann-Whitney', text_format='simple', loc='inside', 
                            line_offset_to_box=0.1, verbose=0)
        axs[cnt].legend([],[], frameon=False)
        if cnt == 1:
            axs[cnt].set_xlabel('Focal')
        else:
            axs[cnt].set_xlabel('')
        axs[cnt].set_ylabel('Heterogeneity score')

        if cnt != 0:
            axs[cnt].set_ylabel('')
            axs[cnt].set_yticks([],[])

    plt.tight_layout()
    plt.subplots_adjust(top=0.83)
    plt.suptitle('Heterogeneity scores of focal cases')
    fig.patch.set_alpha(0.0)
    if saveFig:
        finalResultsDf[['Measurement','BAP1_Focal','SETD2_Focal','PBRM1_Focal']].to_csv(os.path.join(figuresDir, 'heterogeneity_focal.csv'))
        fig.savefig(os.path.join(figuresDir, 'heterogeneity_focal.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'heterogeneity_focal.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()

    # plot cosine distances as compared to TMA:
    tmaDistancesRep = np.load(os.path.join(figuresDir,'tmaDistancesMedian.npy'))

    fig = plt.figure(figsize=(6,5))
    sns.kdeplot(tmaDistancesRep, label='Within TMA cores', color=sns.color_palette()[0], linewidth=4)
    sns.kdeplot(finalResultsDf['Measurement heterogeneity'], label='Within WSIs', color=sns.color_palette()[2], linewidth=4)
    plt.title('Heterogeneity scores \nwithin TMAs and between clusters in WSIs\n')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize=15)
    plt.xlim(0,0.03)
    plt.xticks(np.arange(0, 0.031, 0.01))
    plt.xlabel('Heterogeneity score')
    fig.patch.set_alpha(0.0)
    if saveFig:
        pd.concat([pd.DataFrame({'TMA':tmaDistancesRep}),pd.DataFrame({'WSI':finalResultsDf['Measurement']})],axis=1).to_csv(os.path.join(figuresDir, 'heterogeneity_scores.csv'))
        fig.savefig(os.path.join(figuresDir, 'heterogeneity_scores.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figuresDir, 'heterogeneity_scores.svg'),format='svg',dpi=600,bbox_inches='tight')
    plt.show()

# %%

if __name__ == "__main__":
    
    __main__()
