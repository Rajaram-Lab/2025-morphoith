# %%

import os
import yaml
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(__file__)))
with open('Utils/Global_Params.yaml') as file:
    files = yaml.full_load(file) 

# %%

def load_patients_info():
    """


    Returns
    -------
    allSamplesInfo : pandas DataFrame
        Table with patients informaiton.

    """

    allSamplesFile = files['METADATA']['pilot']
    allSamplesInfo = pd.read_excel(allSamplesFile)
    isGood = np.logical_not(pd.isnull(allSamplesInfo['RNA_Map_Rate']))
    allSamplesInfo = allSamplesInfo[isGood]
    allSamplesInfo = allSamplesInfo[allSamplesInfo.Location != 'Normal']


    return allSamplesInfo
