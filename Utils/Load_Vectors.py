import os
import glob
import numpy as np

# %% Loading data for results visualization.

def load_vectors(saveDir):
    """Fetch profiles features in dictionary format.

    Args:
        saveDir (str): full path to where the features are saved.

    Returns:
        predScoresDir (dictionary): feature vectors (values) for specific samples (keys).
    """
    allNpyList = glob.glob(os.path.join(saveDir, '*.npy'))
    predScoresDir = {}

    for item in allNpyList:
        loadedDir = np.load(os.path.join(saveDir, item), allow_pickle=True).item()
        for key,value in loadedDir.items():
            predScoresDir[key] = np.asarray(list(value.values()))[0]
            
    return predScoresDir
