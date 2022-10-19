import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods to deal with imbalanced data set
# 1. Oversampling
# 2. Class weighting

# Example of class weighting
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    class_weights = [1,50]
    sample_weights = [0] * len(dataset)

    for index, (data, lable) in enumerate(dataset) :
        class_weights[label]

def main():
    pass

if __name__ == "__main__" :
    main()
