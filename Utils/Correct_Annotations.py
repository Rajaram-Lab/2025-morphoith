# %%

def CheckAnnotations(name, classDict, regionNames):
    """Manual corrections by pathologists to annotations provided as TXT files.
       Instead of using updated QuPath annotations, we corrected TXT files.
       Those changes were done prior to analyses.

    Args:
        name (str): slide name.
        classDict (dicitonary): information about annotations derived from TXT files.
        regionNames (dicitonary): information about annotations derived from TXT files.

    Returns:
        classDict (dictionary): information about annotations derived from TXT files, corrected.
        regionNames (dicitonary): information about annotations derived from TXT files, corrected.
    """

    if name == 'KC02000-T1-B8-Flip': 
        classDict[1] = '+Alveolar-2'
        regionNames[0] = '+Alveolar-2'
        
    if name == 'KC01165-T1-B4-Flip': 
        classDict[1] = '+LN-Alveolar-2' 
        regionNames[0] = '+LN-Alveolar-2'
        classDict[2] = '+LN-2' 
        regionNames[1] = '+LN-2'
        classDict[4] = '+LN-Trabecular-2'
        regionNames[3] = '+LN-Trabecular-2'

    if name == 'KC02000-T1-B5-Flip':
        classDict[6] = '+Solid-3'
        regionNames[5] = '+Solid-3'
        
    if name == 'KC01434-T1-A4-Flip':
        classDict[2] = '+SN-2'
        regionNames[1] = '+SN-2'

    if name=='10693': 
        classDict[3] = '+LN-Alveolar-3'
        regionNames[2] = '+LN-Alveolar-3'

    if name=='158913':
        classDict[1] = '+Solid-4'
        regionNames[0] = '+Solid-4'

    if name=='10758':
        for i in [1,3,7,8,10]:
            classDict[i] = '+Alveolar-papillary-3'
            regionNames[i-1] = '+Alveolar-papillary-3'
            classDict[11] = '+Solid-4'
            regionNames[10] = '+Solid-4'

    regionNames = [i.replace('SN', 'Small nest') for i in regionNames]
    regionNames = [i.replace('LN', 'Large nest') for i in regionNames]

    for i, j in classDict.items():
        if 'SN' in j:
            classDict[i] = j.replace('SN', 'Small nest')
        elif 'LN' in j:
            classDict[i] = j.replace('LN', 'Large nest')

    return classDict, regionNames
        