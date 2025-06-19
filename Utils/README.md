# Utilities 

## Functions 

| File    | Description |
| -------- | ------- |
| 1. [Clusters.py](Clusters.py) | Calculate representative features from morphological clusters and make comparisons between the clusters. | 
| 2. [Controls.py](Controls.py) | Create full and partial negative controls.|
| 3. [Correct_Annotations.py](Correct_Annotations.py) | Corrections to annotations made by pathologists. |
| 4. [Feature_Extractor.py](Feature_Extractor.py) | Pipeline to extract features from nuclei. |
| 5. [Image_Utils.py](Image_Utils.py) | Pipeline for whole-slide image processing including annotations extraction. |
| 6. [Load_Patches.py](Load_Patches.py) | Load patch images to overlay them on t-SNES. |
| 7. [Load_PatientInfo.py](Load_PatientInfo.py) | Load patient information for per-patient analysis based on multiple samples. |
| 8. [Load_Vectors.py](Load_Vectors.py) | Load feature vectors for per-patient analysis based on multiple samples.|
| 9. [Normalization.py](Normalization.py) | Schemes for Macenko and Vahadane stain normalization using [staintools](https://github.com/Peter554/StainTools/). |
| 10. [Patch_Generation.py](Patch_Generation.py) | Pipeline for patch generation. |
| 11. [Pytorch_Dataset.py](Pytorch_Dataset.py) | Custom dataset function for Pytorch. |
| 12. [RetCCL_ResNet.py](RetCCL_ResNet.py) | Pipeline necessary for [RetCCL](https://github.com/Xiyue-Wang/RetCCL).|
| 13. [Tessellation.py](Tessellation.py) |Pipeline for WSI tessellation.|
| 14. [Transforms.py](Transforms.py) | Additional Pytorch Dataset transformations from [Augumentation-Pytorch-Transforms](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms). |
| 15. [Tumor_Response.py](Tumor_Response.py) | Function for tissue classifier response processing. |

## Parameters

| File    | Description |
| -------- | ------- |
| 1. [Slides_To_Visualize.csv](Slides_To_Visualize.csv) | List of WSI used in the figures. | 
| 2. [Global_Params.yaml](Global_Params.yaml) | YAML file with parameters. |

## Folders

| File    | Description |
| -------- | ------- |
| 1. [Datasets](Datasets) | CSV files with defined input WSIs for analyzes. | 
| 2. [MoCoV2](MoCoV2) | Pipeline for [MoCo-v2](https://github.com/facebookresearch/moco). | 
| 3. [tiler](tiler) | Pipeline for [tiler](https://github.com/the-lay/tiler). |

