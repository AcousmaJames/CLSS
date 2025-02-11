# loaddataset.py
from torchvision.datasets import CIFAR10
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from torchvision import transforms
unloader = transforms.ToPILImage()

class PreDataset(Dataset):
    def __init__(self, images_dir,transform=None):
        self.images_dir = images_dir
        # self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        # self.label_files = sorted([f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        global imgL, imgR
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        # label_path = os.path.join(self.labels_dir, self.label_files[idx])
        # Load image
        img = Image.open(img_path).convert("RGB")

        # # Load label
        # boxes = []
        # with open(label_path, 'r') as f:
        #     for line in f.readlines():
        #         class_id, x_center, y_center, width, height = map(float, line.strip().split())
        #         boxes.append([class_id, x_center, y_center, width, height])

        # boxes = torch.tensor(boxes)

        # Apply transformation to the image
        if self.transform:
            imgL = self.transform(img)
            imgR = self.transform(img)
            # imgR2 = imgR.cpu().clone()
            # imgR2 = unloader(imgR2)
            # imgR2.save('1.png')
            # imgL.save('2.png')

        return imgL, imgR



if __name__=="__main__":

    import config
    images_dir = r"G:\wkkSet\henanSet\allData\image_all"
    # labels_dir = 'dataset/labels'
    dataset = PreDataset(images_dir,  transform=config.train_transform)
    # train_data = PreDataset(root='dataset', train=True, transform=config.train_transform, download=True)
    print(dataset[0])
