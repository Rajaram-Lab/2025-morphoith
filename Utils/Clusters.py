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

import numpy as np
from scipy.spatial import distance

# %%

def calculate_feature_representation(data, labels):
    """Featch feature profile closes to the mean representation of the feature for sample.

    Args:
        data (array): feature profiles.
        labels (array): sample labels corresponding to the feature profiles.

    Returns:
        representative_features (array): representative feature profiles.
        representative_indices (array): index of the representative feature profiles.
        area_sizes (array): sizes of the samples.
    """

    unique_labels = np.unique(labels)
    representative_features = {}
    representative_indices = {}
    area_sizes = {}
    
    for label in unique_labels:
        cluster_points = data[labels == label]
        cluster_indices = np.where(labels == label)[0]
        mean_feature = np.mean(cluster_points, axis=0)

        distances = np.linalg.norm(cluster_points - mean_feature, axis=1)
        representative_index = np.argmin(distances)
        
        representative_feature = cluster_points[representative_index]
        representative_features[label] = representative_feature
        representative_indices[label] = cluster_indices[representative_index]

        area_sizes[label] = cluster_points.shape[0]

    return representative_features, representative_indices, area_sizes

def centroid_similarity(centroids):
    """Calculate cosine distance between the profile features representative of the clusters.

    Args:
        centroids (dictionary): profile features (values) for pairs of clusters (keys).

    Returns:
        similarity (dictionary): cosine distances (values) for pairs of clusters (keys).
    """

    similarity = {}
    labels = list(centroids.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            centroid_i = centroids[labels[i]]
            centroid_j = centroids[labels[j]]
            sim = distance.cosine(centroid_i, centroid_j)
            similarity[(labels[i], labels[j])] = sim

    return similarity
