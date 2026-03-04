import torch
from torch.utils.data import Dataset, DataLoader

class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        eeg = torch.tensor(entry['eeg'], dtype=torch.float32)
        eye = torch.tensor(entry['eye'], dtype=torch.float32)
        emotion = torch.tensor(entry['emotion'], dtype=torch.float32)
        label = torch.tensor(entry['label'], dtype=torch.long)
        return {'eeg': eeg, 'eye': eye, 'emotion': emotion, 'label': label}
