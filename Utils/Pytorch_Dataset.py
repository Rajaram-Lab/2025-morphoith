# %%

from torch.utils.data import Dataset

class Dataset_Simple(Dataset):
    """Dataset generator for the simple model.

    Args:
        Dataset (class): PyTorch class.
    """

    def __init__(self, patchData, permute_data):
        self.data = patchData
        self.numberOfPatches = patchData.shape[0]
        self.permute_data = permute_data

    def __len__(self):
        return int(self.numberOfPatches)

    def __getitem__(self, index):
        
        image = self.permute_data(self.data[index])
        return image
    