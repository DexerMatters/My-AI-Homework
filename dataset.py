import torch
import utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dataset = datasets.ImageFolder(root_dir, transform=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    
class ImageSubset(Dataset):
    def __init__(self, dataset, indices, augmented=False):
        self.dataset = dataset
        self.indices = indices
        self.augmented = augmented

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return self.augment(image) if self.augmented else image, label
    
    def augment(self, image):
        param = utils.get_config()["augmentation"]
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=param["horizontal_flip_randomness"]),
            transforms.RandomVerticalFlip(p=param["vertical_flip_randomness"]),
            transforms.RandomRotation(param["rotation_range"], resample=False, expand=False, center=None),
        ])
        return trans(image)


class TrainDataloader:
    def __init__(self, root_dir, batch_size):
        self.dataset = ImageDataset(root_dir, augmented=True)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

class TestDataloader:
    def __init__(self, root_dir, batch_size):
        self.dataset = ImageDataset(root_dir, train=False, augmented=False)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
