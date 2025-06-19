# Model training

## Training data
We train on patches from cohort "TMA training" generated as seen [here](../Data_Generation/Generate_Patches_TMA1.py). Patches are used as input in pairs, either with a flag:
1. `same_patch`: same patch, augumented
2. `close_patch`: two patches from the same TMA core, augumented
3. `combied_patch`: combinaiton of previous approaches

Augumentations include transforms from [Augumentation-Pytorch-Transforms](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms). 

## Encoders
We train CNN (ResNet50) or Transformer (ViT-b) encoders using the following approaches: BYOL, MoCo-v2, Triplet. 


| File    | Description |
| -------- | ------- |
| 1. [Training_BYOL.py](Training_BYOL.py)  | BYOL approach for encoder training based on implementation by [Phil Wang](https://github.com/lucidrains/byol-pytorch).| 
| 2. [Training_MoCoV2.py](Training_MoCoV2.py) | MoCo-v2 approach for encoder training based on implementation by [Meta Research](https://github.com/facebookresearch/moco). |
| 3. [Training_Triplet.py](Training_Triplet.py)  | Triplet approach for encoder training based on implementation by [Kevin Musgrave](https://github.com/KevinMusgrave/pytorch-metric-learning). |