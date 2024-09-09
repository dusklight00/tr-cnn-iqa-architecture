import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class KadidDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        dmos_value = self.dataset.iloc[0, 2] / max(self.dataset.iloc[:, 2])
        print("DMOS value is : " + str(dmos_value))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        image = io.imread(img_name)
        dmos_value = self.dataset.iloc[idx, 2] / max(self.dataset.iloc[:, 2])


        sample = [image, dmos_value]

        if self.transform:
            image = self.transform(image)
            sample = [image, dmos_value]
        # print(sample)

        return sample