import typing

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from utils import round_to_nearest_odd
from config import IMG_SIZE, BATCH_SIZE


class BarlowTwinsDataset(Dataset):
    def __init__(self, images: np.array):
        super(BarlowTwinsDataset, self).__init__()

        self.images = images

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(round_to_nearest_odd(IMG_SIZE * 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        chosen_image = Image.fromarray(self.images[item])

        view1 = self.transform(chosen_image)
        view2 = self.transform(chosen_image)

        return view1, view2


def get_cifar10_loader() -> DataLoader:
    train_cifar10 = datasets.CIFAR10(root='./data', download=True, train=True)
    train_images = train_cifar10.data
    train_dataset = BarlowTwinsDataset(train_images)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader
