from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class HorseZebra(Dataset) :
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_imgs = os.listdir(root_zebra)
        self.horse_imgs = os.listdir(root_horse)

        self.length_dataset = max(len(self.zebra_imgs), len(self.horse_imgs))
        self.zebra_len = len(self.zebra_imgs)
        self.horse_len = len(self.horse_imgs)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_imgs[index % self.zebra_len]
        horse_img = self.horse_imgs[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform :
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img