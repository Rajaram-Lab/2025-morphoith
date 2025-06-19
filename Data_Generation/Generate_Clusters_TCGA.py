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
import scipy
import skimage
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

clusterSaveDir = files['MASKS']['clusters_tcga']

# %%

samplesList = pd.read_csv(files['DATASETS']['tcga'])
samplesList = list(samplesList['Name'])

for name in tqdm.tqdm(samplesList):
    
    name = str(name)

    tissueMaskFile = os.path.join(files['MASKS']['tissue_classifier'],'tcga',name+'.npy')
    profilesFile = os.path.join(files['MASKS']['morphoith'],'tcga',name+'.npy')
    profiles = np.load(profilesFile)
    tumor = np.load(tissueMaskFile)

    # only make the tumor area more smooth to mirror how annotations are made:
    tumor[:, :, 5] = scipy.ndimage.gaussian_filter(tumor[:, :, 5], 5)
    tumor = np.argmax(tumor, axis=-1)
    tumor[tumor != 5] = 0
    tumor[tumor == 5] = 1
    tumor = np.uint8(skimage.morphology.remove_small_objects(tumor == 1, 100))
    tumor = cv2.erode(np.uint8(tumor),np.ones((1,1),np.uint8))


    # are there marks to discard?
    maskFile = os.path.join(files['MASKS']['markers']['tcga'],name+'.svs_mask.png')
    markers = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    markers = cv2.resize(markers,(profiles.shape[1],profiles.shape[0]))
    markers[markers==3]=4
    markers[markers!=4]=0
    markers[markers==4]=1
    tumor = cv2.subtract(tumor, markers)
        
    idx1, idx2 = np.nonzero(tumor)
    idxSpatial = list(zip(idx1, idx2))
    idxSpatial= np.array(idxSpatial)
    profilesFlat = profiles[idx1,idx2]

    if np.sum(tumor) > 100:
        n_samples = len(idxSpatial)
        knn_graph = kneighbors_graph(idxSpatial, n_neighbors=8, mode='connectivity', include_self=False)
        knn_graph_lil = knn_graph.tolil()  
        n_components, component_labels = connected_components(knn_graph_lil)
        while n_components > 1:
            current_components = np.unique(component_labels)
            min_distance = np.inf
            closest_pair = (None, None)
            for i in range(len(current_components)):
                for j in range(i + 1, len(current_components)):
                    component_i_indices = np.where(component_labels == current_components[i])[0]
                    component_j_indices = np.where(component_labels == current_components[j])[0]
                    distances_between_components = distance.cdist(
                        np.array(idxSpatial)[component_i_indices],
                        np.array(idxSpatial)[component_j_indices],
                        metric='euclidean'
                    )
                    current_min_idx = np.unravel_index(np.argmin(distances_between_components), distances_between_components.shape)
                    current_distance = distances_between_components[current_min_idx]
                    if current_distance < min_distance:
                        min_distance = current_distance
                        closest_pair = (component_i_indices[current_min_idx[0]], component_j_indices[current_min_idx[1]])
            if closest_pair[0] is not None and closest_pair[1] is not None:
                knn_graph_lil[closest_pair[0], closest_pair[1]] = 1
                knn_graph_lil[closest_pair[1], closest_pair[0]] = 1
                n_components, component_labels = connected_components(knn_graph_lil)

        knn_graph_csr = knn_graph_lil.tocsr()
        knn_graph_csr.eliminate_zeros()
        profilesPCA = PCA(n_components=10, random_state=123).fit_transform(profilesFlat)
        
        clusterList = [5,10,15,20]

        for n_cluster in clusterList:
            clusFile = os.path.join(clusterSaveDir,name+'_cluster'+str(n_cluster)+'.npy')
            slideCluster = AgglomerativeClustering(n_clusters=n_cluster, connectivity=knn_graph_csr, linkage='ward')
            fitted = slideCluster.fit(profilesPCA)
            clus = fitted.labels_
            clusterMask = np.zeros_like(tumor)
            clusterMask[idx1, idx2] = clus+1
            
            np.save(clusFile,clusterMask)


# %%
