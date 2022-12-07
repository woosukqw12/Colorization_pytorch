import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DatasetImg(Dataset):
    def __init__(self, l, ab, input_transforms=[], output_transforms=[]):
        
        self.l = l
        self.ab = ab
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
    
    def __len__(self):
        return len(self.l)
    
    def __getitem__(self, idx):
        l = self.l[idx]
        ab = self.ab[idx]
        
        if self.input_transforms is not None:
            for fn in self.input_transforms:
                l = fn(l)
        
        if self.output_transforms is not None:
            for fn in self.output_transforms:
                ab = fn(ab)
                
        return l, ab
        return {'L': l, 'ab': ab}
    
    
def load_data(home, channels_first=True, train_percent=0.8):
    ab1 = np.load(os.path.join(home,"ab/ab", "ab1.npy"))
    ab2 = np.load(os.path.join(home, "ab/ab", "ab2.npy"))
    ab3 = np.load(os.path.join(home,"ab/ab", "ab3.npy"))
    ab = np.concatenate([ab1, ab2, ab3], axis=0).astype("float32")
    # ab = np.transpose(ab, [0, 3, 1, 2])
    l = np.load(os.path.join(home,"l/gray_scale.npy")).astype("float32")


    return train_test_split(ab,l, train_size=train_percent)
