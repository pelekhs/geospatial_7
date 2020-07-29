from torch.utils.data import Dataset
import random

class HyRank(Dataset):
    def __init__(self, X, y, transform=False, supervision=True):
        self.supervision = supervision
        self.transform = transform
        self.labels = y
        self.patches = X

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        original_patch = self.patches[idx]
        if self.supervision:
            if self.transform:
              key = random.choice(list(self.transform.keys()))
              return {'tensors': self.transform[key](original_patch).copy(),
                      'labels': self.labels[idx]}
            else:
              return {'tensors': original_patch.copy(),
                      'labels': self.labels[idx]}
        else:
            if self.transform:
                # choose transformation type randomly
                key = random.choice(list(self.transform.keys()))
                if key != "noise":
                    return {'tensors': self.transform[key](original_patch).copy(),
                            'labels': self.transform[key](original_patch).copy()}
                else: # denoise
                   return {'tensors': self.transform[key](original_patch).copy(),
                           'labels': original_patch.copy()}
            else:
                return {'tensors': original_patch.copy(),
                        'labels': original_patch.copy()}
