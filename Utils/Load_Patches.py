# %%

import os
import yaml
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

from . import Patch_Generation as pg

# %%

def load_patches_pilot(patientList, patchSaveDir):
    """ Load patches as arrays for specific samples.

    Args:
        patientList (list): list of patient names.
        patchSaveDir (str): full path to where the patches are saved.

    Returns:
        patchesDir (dictionary): array of patches (values) for samples (keys).
    """

    allSamplesFile = files['METADATA']['pilot']
    allSamplesInfo = pd.read_excel(allSamplesFile)
    isGood = np.logical_not(pd.isnull(allSamplesInfo['RNA_Map_Rate']))
    allSamplesInfo = allSamplesInfo[isGood].set_index('Patient.ID')
    
    patchesDir = {}
    for patient in patientList:
        patientInfo = allSamplesInfo.loc[[patient]]

        for uniqueSlide in pd.unique(patientInfo['Image.Flip'].dropna()):
            slideInfo = patientInfo[patientInfo['Image.Flip'] == uniqueSlide]
            slideId = uniqueSlide.strip('.svs')

            hdf5File = os.path.join(patchSaveDir, slideId + '.hdf5')
            patchData, patchClasses, classDict = pg.LoadPatchData([hdf5File], returnSampleNumbers=False)

            for i in range(slideInfo.shape[0]):
                sampleInfo = slideInfo.iloc[i]
                sampleId = sampleInfo['Sample.ID']

                annoName = sampleInfo['Punch.ID']
                isInAnno = patchClasses == classDict[annoName]
                patchesInfo = patchData[0][isInAnno]
                patchesDir[sampleId] = patchesInfo

    return patchesDir

