import torch
import os
import cv2
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class ImageDataset(Dataset):
    def __init__(self, root_dir, radio=0.8, augmented=False, train=True):
        self.root_dir = root_dir
        self.augmented = augmented
        self.file_names = os.listdir(self.root_dir)
        self.train_data = []
        self.test_data = []
        self.mode = 1 if train else 0
        m = dict()
        for i in range(len(self.file_names)):
            m.update({self.file_names[i]: i})
        for i in self.file_names:
            path = os.path.join(self.root_dir, i)
            sub_dirs = os.listdir(path)
            
            # Split the data into training and testing
            for j in range(len(sub_dirs)):
                x = sub_dirs[j]
                if x.split('.')[-1] == 'svg':
                    continue
                if j < len(sub_dirs) * radio:
                    self.train_data.append((os.path.join(path, x), m[i]))
                else:
                    self.test_data.append((os.path.join(path, x), m[i]))

    def __len__(self):
        if self.mode == 1:
            return len(self.train_data)
        return len(self.test_data)

    def __getitem__(self, idx):
        if self.mode == 1:
            path, label = self.train_data[idx]
        else:
            path, label = self.test_data[idx]
        return self.transform(cv2.imread(path)) , label
    
    def transform(self, image):
        # Read size of the image
        h, w, _ = image.shape
        # Resize the image with the same aspect ratio
        if h < w:
            new_h = 224
            new_w = int(w * new_h / h)
        else:
            new_w = 224
            new_h = int(h * new_w / w)
        image = cv2.resize(image, (new_w, new_h))
        # Crop the center of the image
        start_h = (new_h - 224) // 2
        start_w = (new_w - 224) // 2
        image = image[start_h:start_h + 224, start_w:start_w + 224, :]
        # Convert the image to a tensor
        image = ToTensor()(image)
        if self.augmented:
            image = self.augment(0.2, image)
        return image / 255.0

    
    def augment(self, randomness, image):
        # Data augmentation
        # Random horizontal flip
        if torch.rand(1) > randomness:
            image = image.flip(2)
        # Random vertical flip
        if torch.rand(1) > randomness:
            image = image.flip(1)
        # Random rotation
        angle = torch.randint(-10, 10, (1,)).item()
        image = image.rot90(angle // 90)
        # Random translation
        translation = torch.randint(-10, 10, (2,))
        image = torch.roll(image, translation.tolist(), (1, 2))
        # Normalization
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)
        return image
    

class TrainDataloader:
    def __init__(self, root_dir, batch_size):
        self.dataset = ImageDataset(root_dir, augmented=False)
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
