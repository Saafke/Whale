import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, io, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class WhaleDataset(Dataset):
    """Whale dataset - Kaggle."""

    def __init__(self, csv_file, image_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.

            Data format: Anchor | Positive | Negative | Switch
        """

        self.whale_dataset = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform_all = transforms.Compose([
            transforms.Resize([224,224]),
            
            transforms.ToTensor()
        ])
        self.transform_switch = transforms.Compose([
            transforms.RandomHorizontalFlip(1), # flip for sure
            transforms.RandomGrayscale(0.5)
        ])



    def __len__(self):
        print("length:", self.whale_dataset.shape[0])
        return self.whale_dataset.shape[0]

    def __getitem__(self, idx):
        # -- Get Image filenames
        row = self.whale_dataset.iloc[idx, :]
        anchor_name = os.path.join(self.image_dir, row['Anchor'])
        positive_name = os.path.join(self.image_dir, row['Positive'])
        negative_name = os.path.join(self.image_dir, row['Negative'])
        switch = row['Switch']

        # -- Open images
        anchor_img = Image.open(anchor_name).convert('RGB')
        positive_img = Image.open(positive_name).convert('RGB')
        negative_img = Image.open(negative_name).convert('RGB')

        # transform images
        if switch: # if anchor == positive, flip and greyscale positive
            positive_img = self.transform_switch(positive_img)
        
        anchor_img = self.transform_all(anchor_img)
        positive_img = self.transform_all(positive_img)
        negative_img = self.transform_all(negative_img)

        ###just to visualize: tensors to pil images
        # a = anchor_img.numpy(); b = positive_img.numpy(); c = negative_img.numpy()
        # a = np.transpose(a, (1,2,0)); b = np.transpose(b, (1,2,0)); c = np.transpose(c, (1,2,0))

        # _, axarr = plt.subplots(1,3)
        # axarr[0].imshow(a); axarr[1].imshow(b); axarr[2].imshow(c)
        # plt.show()

        # input("Press Enter to continue...")
        ###

        sample = {'Anchor': anchor_img, 'Positive': positive_img, 'Negative': negative_img}
        
        return sample