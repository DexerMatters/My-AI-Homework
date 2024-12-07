import torch
import utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dataset = datasets.ImageFolder(
            root_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class ImageSubset(Dataset):
    def __init__(self, dataset, indices, augmented=False):
        self.dataset = dataset
        self.augmented = augmented
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return self.augment(image) if self.augmented else image, label

    def augment(self, image):
        param = utils.get_config()["data"]["augmentation"]
        trans = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=param["horizontal_flip_randomness"]),
                transforms.RandomVerticalFlip(p=param["vertical_flip_randomness"]),
                transforms.RandomRotation(
                    param["rotation_range"], expand=False, center=None
                ),
            ]
        )
        return trans(image)
