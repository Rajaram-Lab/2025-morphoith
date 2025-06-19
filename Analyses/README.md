# Analyses

## Sampling

| File    | Description | Figures |
| -------- | ------- | ------- |
| 1. [Sampling_Mutations.py](Sampling_Mutations.py)  | Sample driver gene mutation status from cohort "WSI2". | |

## Retrieval

| File    | Description | Figures |
| -------- | ------- | ------- |
| 1. [Retrieval_TMA1.py](Retrieval_TMA1.py)  | Retrieval of patches from the same TMA core for cohort "TMA training".| |
| 2. [Retrieval_TMA2.py](Retrieval_TMA2.py) | Retrieval of patches from the same TMA core or with the same nuclear grade for cohort "TMA validation". Global representation in t-SNE form with grade overlay. | |

## Other

| File    | Description | Figures |
| -------- | ------- | ------- |
| 1. [Analysis_Annotations.py](Analysis_Annotations.py)  | Analysis of pathologists annotations of WSI based on their vascular architecture and nuclear grade in cohorts "WSI1", "WS2". | |
| 2. [Analysis_HandcrafedFeatures.py](Analysis_HandcrafedFeatures.py) | Comparison between MorphoITH-derived similarity measure with nuclear grade area sizes, vasculature density, and eosin intensity in cohorts "WSI1", "WS2".| |
| 3. [Analysis_Heterogeneity.py](Analysis_Heterogeneity.py) | Hetererogeneity score calculations of WSI in cohort "WSI3". | |
| 4. [Analysis_Heterogeneity_PerPatient.py](Analysis_Heterogeneity_PerPatient.py) | Hetererogeneity score calculations of WSI in cohort "WSI1". | |
| 5. [Analysis_Heterogeneity_TCGA.py](Analysis_Heterogeneity_TCGA.py) | Hetererogeneity score calculations of WSI in cohort 

## Encoders

We compare different encoders during the Retrieval Task.

| Encoder    | 
| -------- | 
| 1. [RetCCL](https://github.com/Xiyue-Wang/RetCCL)| 
| 2. [UNI](https://huggingface.co/MahmoodLab/UNI) and [UNIv2](https://huggingface.co/MahmoodLab/UNI2-h)|
| 3. [CONCH](https://huggingface.co/MahmoodLab/CONCH) and [CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5)|
| 4. [Virchow](https://huggingface.co/paige-ai/Virchow) and [Virchowv2](https://huggingface.co/paige-ai/Virchow2)|
| 5. [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) |
