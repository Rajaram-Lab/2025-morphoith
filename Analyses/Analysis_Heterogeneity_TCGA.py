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
import skimage
import numpy as np
import pandas as pd

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
                tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],'tcga',name+'.npy')
                profilesFile = os.path.join(files['MASKS']['morphoith'],'tcga',name+'.npy')

                profiles = np.load(profilesFile)
                tumor = np.load(tissueMaskFile)
                tumor = np.uint8(SmoothResponseAndClasses(tumor,smoothSize=0)[0])
                tumor[tumor != 5] = 0
                tumor[tumor == 5] = 1
                tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
                tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))

                clusterSaveDir = files['MASKS']['clusters_tcga']
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
    
    filteredDf['Size difference'] = filteredDf['Size'].astype('int')-filteredDf['Size_Corrd2'].astype('int')
    filteredDf['Size difference'] = filteredDf['Size difference'].abs()
    filteredDf = filteredDf[['Name','Corrd1', 'Corrd2','Measurement','Size difference']]

    def exponential_weight(measurements):
        x_norm = (measurements - measurements.min()) / (measurements.max() - measurements.min())
        beta = 1
        weights = np.exp(beta * x_norm)
        weights /= weights.sum()
        return np.sum(measurements * weights)

    heterogeneityDf = (
        filteredDf.groupby('Name')
        .apply(exponential_weight)
        .reset_index())
    
    heterogeneityDf = heterogeneityDf.reset_index(drop=True)
    heterogeneityDf.to_csv(os.path.join(figuresDir, 'heterogeneity_TCGA.csv'))
    
# %%

def __main__():

    samplesList = pd.read_csv(files['DATASETS']['tcga'])
    samplesList = list(samplesList['Name'])
    measureDir, sizesDir = heterogeneity_measure(samplesList)
    save_heterogeneity_dataframe(measureDir, sizesDir)
        

# %%

if __name__ == "__main__":
    
    __main__()


# %%
