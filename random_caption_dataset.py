import pandas as pd
import torch
import os
import random
from skimage import io
from torch.utils.data import Dataset


class RandomCaptionDataset(Dataset):
    """Image captioning dataset"""

    def __init__(self, root_dir, annotations_file, transform=None):
        """
        Args:
            csv_file(string): captions file
            root_dir(string): images directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.df_captions = pd.read_csv(annotations_file, sep=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Length is nb_captions / captions_per_image"""
        return int(len(self.df_captions) / 5)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        im_name = self.df_captions.iloc[index, 0]
        im_path = os.path.join(self.root_dir, im_name)
        image = io.imread(im_path)

        captions = list(self.df_captions.loc[self.df_captions['image_id'] == im_name].iloc[:, 1])
        caption = random.choice(captions)

        sample = {'image': image, 'caption': caption, 'im_path':im_path}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    dataset = RandomCaptionDataset('./flickr8k/images/train/', './flickr8k/annotations/annotations_image_id.csv')
    print(dataset[0])
    print(dataset[0]['image'].shape)
