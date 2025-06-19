# Data generation

## Patch generation


We use encoders to generate image features from patches or whole-slide images.

| File    | Description |
| -------- | ------- |
| 1. [Generate_Patches_TMA1.py](Generate_Patches_TMA1.py)  | Generate TMA patches.| 
| 2. [Generate_DataSplit_TMA1.py](Generate_DataSplit_TMA1.py) | Split cohort "TMA training" into training and testing.|
| 3. [Generate_Patches_WSI1.py](Generate_Patches_WSI1.py) | Generate patches for cohort "WSI1".|

## Morphological clusters generation

| File    | Description |
| -------- | ------- |
| 1. [Generate_Clusters.py](Generate_Clusters.py)  | Generate morpholoigical clusters based on image profiles. | 
| 2. [Generate_Clusters_TCGA.py](Generate_Clusters_TCGA.py)  | Generate morpholoigical clusters based on image profiles - for cohort "TCGA KIRC". | 