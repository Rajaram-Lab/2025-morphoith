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

from scipy import spatial
from skimage import measure
import numpy as np

# %% 

def MakeHalfMask(pairMask, numberOfRuns):
    """Creates true negative controls of a mask.

    Args:
        pairMask (array): mask with annotations.
        numberOfRuns (int): number of controls per slide.

    Returns:
       halfMasks (array): true negative controls (N=numberOfRuns).
       halfOverlapA (array): overlap between annotation A and the original mask.
       halfOverlapB (array): overlap between annotation B and the original mask.
    """

    maskLabeled = measure.label(pairMask, background=0)  
    degrees = int(360/numberOfRuns)
    halfMasks = []
    halfOverlapA = []
    halfOverlapB = []

    for run in range(numberOfRuns):

        degree = np.deg2rad(degrees*(run+1))
        halfMask = pairMask.copy()
        
        for i in np.unique(maskLabeled)[1:]:

            cluster_positions = np.argwhere(maskLabeled == i)
            mean_row = cluster_positions[:, 0].mean()
            mean_col = cluster_positions[:, 1].mean()

            slope = np.tan(degree)
            line_values = slope * (cluster_positions[:, 1] - mean_col) + mean_row
            
            above_mask = cluster_positions[:, 0] < line_values
            below_mask = cluster_positions[:, 0] >= line_values

            if degree < np.pi: 
                halfMask[cluster_positions[above_mask][:, 0], cluster_positions[above_mask][:, 1]] = 1
                halfMask[cluster_positions[below_mask][:, 0], cluster_positions[below_mask][:, 1]] = 2
            else: 
                halfMask[cluster_positions[above_mask][:, 0], cluster_positions[above_mask][:, 1]] = 2
                halfMask[cluster_positions[below_mask][:, 0], cluster_positions[below_mask][:, 1]] = 1

        halfMasks.append(halfMask)
        halfOverlapA.append(np.sum(halfMask*pairMask==4)/np.sum(pairMask==2))
        halfOverlapB.append(np.sum(halfMask*pairMask==1)/np.sum(pairMask==1))

    return halfMasks, halfOverlapA, halfOverlapB

def MakeCircleMask(pairMask, numberOfRuns):
    """Creates contiguity controls of a mask.

    Args:
        pairMask (array): mask with annotations.
        numberOfRuns (int): number of controls per slide.

    Returns:
       circleMasks (array): contiguity controls (N=numberOfRuns).
       circleOverlapA (array): overlap between annotation A and the original mask.
       circleOverlapB (array): overlap between annotation B and the original mask.
    """

    allTissue = pairMask.copy()
    allTissue[allTissue > 1] = 1

    maskLabeled = measure.label(pairMask>1, background=0)  
    u, count = np.unique(maskLabeled[maskLabeled > 0], return_counts=True)
    count_sort_ind = np.argsort(-count)
    uniqueOrdered = u[count_sort_ind]
    countsOrdered = count[count_sort_ind]

    circleMasks = []
    circleOverlapA = []
    circleOverlapB  = []

    for run in range(numberOfRuns):
        randomMask = allTissue.copy()

        for cnt, i in enumerate(uniqueOrdered):
            if countsOrdered[uniqueOrdered==i] > 1:
                (idx1, idx2) = np.nonzero(np.uint8(randomMask)==1)
                coords = list(zip(idx1, idx2))
                np.random.seed((cnt+1)*run)
                randomIdx = np.random.choice(idx1.shape[0])
                (idx1Random, idx2Random) = idx1[randomIdx], idx2[randomIdx]
                area = countsOrdered[cnt]
                tree = spatial.KDTree(coords)
                _, idxHit = tree.query([(idx1Random, idx2Random)], k=area)

                randomMask[np.asarray(coords)[idxHit[0]][:,0], np.asarray(coords)[idxHit[0]][:,1]] = cnt+2
        
        randomMask[randomMask>1] = 2

        circleMasks.append(randomMask)
        circleOverlapA.append(np.sum(randomMask*pairMask==4)/np.sum(pairMask==2))
        circleOverlapB.append(np.sum(randomMask*pairMask==1)/np.sum(pairMask==1))
            

    return circleMasks,circleOverlapA,circleOverlapB

def MakeCircleMaskOnTumor(pairMask, tumor, numberOfRuns):
    """Creates contiguity controls of a mask, adjusted for a mask that has tumor area besides the annotations.

    Args:
        pairMask (array): mask with annotations.
        tumor (array): maks with tumor area.
        numberOfRuns (int): number of controls per slide.

    Returns:
       circleMasks (array): contiguity controls (N=numberOfRuns).
       circleOverlapA (array): overlap between annotation A and the original mask.
       circleOverlapB (array): overlap between annotation B and the original mask.
    """

    maskLabeled = measure.label(pairMask, background=0)  
    u, count = np.unique(maskLabeled[maskLabeled > 0], return_counts=True)
    count_sort_ind = np.argsort(-count)
    uniqueOrdered = u[count_sort_ind]
    countsOrdered = count[count_sort_ind]

    circleMasks = []
    circleOverlapA = []
    circleOverlapB  = []

    for run in range(numberOfRuns):
        randomMask = tumor.copy()
        for cnt, i in enumerate(uniqueOrdered):

            if countsOrdered[uniqueOrdered==i] > 1:

                (idx1, idx2) = np.nonzero(np.uint8(randomMask)==1)
                coords = list(zip(idx1, idx2))
                np.random.seed((cnt+1)*(i+1)*(run+1))
                randomIdx = np.random.choice(idx1.shape[0])
                (idx1Random, idx2Random) = idx1[randomIdx], idx2[randomIdx]
                area = countsOrdered[cnt]
                tree = spatial.KDTree(coords)
                _, idxHit = tree.query([(idx1Random, idx2Random)], k=area)

                pointLabel = pairMask[maskLabeled==i]
                randomMask[np.asarray(coords)[idxHit[0]][:,0], np.asarray(coords)[idxHit[0]][:,1]] = np.unique(pointLabel).item()+1
        randomMask[randomMask>0] = randomMask[randomMask>0]-1

        circleMasks.append(randomMask)
        circleOverlapA.append(np.sum(randomMask*pairMask==4)/np.sum(pairMask==2))
        circleOverlapB.append(np.sum(randomMask*pairMask==1)/np.sum(pairMask==1))

    return circleMasks,circleOverlapA,circleOverlapB